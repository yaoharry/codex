//! Installs a completed prefix only after its exact preserved suffix has a durable checkpoint.

use super::*;
use crate::compact_retained_tool_outputs::is_retained_tool_output;
use std::collections::HashMap;
use std::collections::HashSet;

pub(crate) fn retain_for_body_recovery(envelope: &ResponseItemEnvelope) -> bool {
    match &envelope.item {
        ResponseItem::Message { role, .. } => role != "assistant",
        ResponseItem::AgentMessage { .. }
        | ResponseItem::AdditionalTools { .. }
        | ResponseItem::ConfigurationUpdate { .. }
        | ResponseItem::FunctionCallOutput { call_id: None, .. } => true,
        _ => is_retained_tool_output(envelope),
    }
}

/// Unknown and retained live groups stop selection; an earlier completed prefix remains eligible.
pub(crate) fn completed_prefix_ends(
    items: &[ResponseItemEnvelope],
    protected_start: usize,
) -> Vec<usize> {
    if protected_start > items.len() {
        return Vec::new();
    }
    enum PendingCall<'a> {
        Function(&'a str, Option<&'a str>),
        Custom(&'a str),
    }
    let mut pending = HashMap::new();
    let mut seen = HashSet::new();
    let mut ends = Vec::new();
    for (index, envelope) in items[..protected_start].iter().enumerate() {
        if is_retained_tool_output(envelope) {
            break;
        }
        match &envelope.item {
            ResponseItem::FunctionCall {
                call_id,
                name,
                namespace,
                status,
                ..
            } => {
                if status
                    .as_deref()
                    .is_some_and(|status| status != "completed")
                    || !seen.insert(call_id.as_str())
                {
                    break;
                }
                pending.insert(
                    call_id.as_str(),
                    PendingCall::Function(name, namespace.as_deref()),
                );
            }
            ResponseItem::CustomToolCall {
                call_id,
                name,
                status,
                ..
            } => {
                if status
                    .as_deref()
                    .is_some_and(|status| status != "completed")
                    || !seen.insert(call_id.as_str())
                {
                    break;
                }
                pending.insert(call_id.as_str(), PendingCall::Custom(name));
            }
            ResponseItem::FunctionCallOutput {
                call_id: Some(call_id),
                name,
                namespace,
                ..
            } => {
                let Some(PendingCall::Function(expected_name, expected_namespace)) =
                    pending.remove(call_id.as_str())
                else {
                    break;
                };
                if name.as_deref().is_some_and(|name| name != expected_name)
                    || namespace.as_deref().is_some_and(|namespace| {
                        namespace
                            != expected_namespace
                                .unwrap_or(codex_protocol::DEFAULT_FUNCTION_NAMESPACE)
                    })
                {
                    break;
                }
                if pending.is_empty() {
                    ends.push(index + 1);
                }
            }
            ResponseItem::CustomToolCallOutput { call_id, name, .. } => {
                let Some(PendingCall::Custom(expected_name)) = pending.remove(call_id.as_str())
                else {
                    break;
                };
                if name.as_deref().is_some_and(|name| name != expected_name) {
                    break;
                }
                if pending.is_empty() {
                    ends.push(index + 1);
                }
            }
            ResponseItem::Message { role, status, .. } if role == "assistant" => {
                if status
                    .as_deref()
                    .is_some_and(|status| status != "completed")
                {
                    break;
                }
                if pending.is_empty() {
                    ends.push(index + 1);
                }
            }
            ResponseItem::Compaction { .. } if pending.is_empty() => ends.push(index + 1),
            ResponseItem::Reasoning { .. } => {}
            _ if retain_for_body_recovery(envelope) => {}
            _ => break,
        }
    }
    ends
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn install_preserving_compaction_prefix(
        &self,
        step: &StepContext,
        expected: &ContextManager,
        protected_start: usize,
        prefix: CompactionPrefix,
        mut replacement_prefix: Vec<ResponseItemEnvelope>,
        compaction_response_id: String,
        cancellation: &CancellationToken,
    ) -> bool {
        if cancellation.is_cancelled()
            || compaction_response_id.is_empty()
            || prefix.len == 0
            || prefix.len > protected_start
            || !completed_prefix_ends(expected.annotated_items(), protected_start)
                .contains(&prefix.len)
            || expected.annotated_items()[..prefix.len]
                .iter()
                .any(|item| item.item.turn_id() == Some(step.turn.sub_id.as_str()))
        {
            return false;
        }
        let Some((summary, retained)) = replacement_prefix.split_last() else {
            return false;
        };
        if !matches!(&summary.item, ResponseItem::Compaction { encrypted_content, .. } if !encrypted_content.trim().is_empty())
            || !retained.iter().eq(expected.annotated_items()[..prefix.len]
                .iter()
                .filter(|item| retain_for_body_recovery(item)))
        {
            return false;
        }
        let _settings = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return false,
            settings = thread_settings::acquire_persistence_lock(self) => settings,
        };
        let settings_event = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return false,
            event = thread_settings::applied_event(self) => event,
        };
        let mut state = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return false,
            state = self.state.lock() => state,
        };
        let owner_is_current = |state: &SessionState| {
            let model = step.turn.capture_current_model_info();
            !cancellation.is_cancelled()
                && !state.shutting_down
                && state.last_started_turn_id.as_deref() == Some(step.turn.sub_id.as_str())
                && state.session_configuration.step_settings.as_ref() == step.settings.selected()
                && model.slug == step.settings.model_info.slug
                && model.comp_hash == step.settings.model_info.comp_hash
                && self.services.model_client.auth_owner_generation()
                    == prefix.auth_owner_generation
        };
        if !owner_is_current(&state)
            || state.history.history_version() != prefix.generation
            || state.history.annotated_items() != expected.annotated_items()
        {
            return false;
        }
        let Some(live_thread) = self.live_thread() else {
            return false;
        };
        let canonical = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return false,
            result = self.services.thread_store.load_latest_model_context(codex_thread_store::LoadThreadHistoryParams {
                thread_id: self.thread_id(),
                include_archived: true,
            }) => result,
        };
        let Ok(canonical) = canonical else {
            return false;
        };
        let replay = self
            .reconstruct_history_from_rollout(&step.turn, &canonical.items)
            .await;
        if !replay
            .history
            .iter()
            .map(|item| &item.item)
            .eq(expected.raw_items())
        {
            return false;
        }
        let world_state = state.history.world_state_checkpoint();
        let reference = step.to_turn_context_item();
        let (window_number, window_ids) = state.prepare_compaction_window();
        for envelope in &mut replacement_prefix {
            self.assign_missing_response_item_id(&mut envelope.item);
        }
        let mcp_revision = self.prepare_compaction_checkpoint_metadata(
            &mut replacement_prefix,
            &step.settings.model_info.comp_hash,
        );
        replacement_prefix.extend_from_slice(&expected.annotated_items()[prefix.len..]);
        let mut working = state.clone_history();
        working.checkpoint_sampling(&mut replacement_prefix);
        let replacement_history = replacement_prefix.clone();
        let invalidate_reviews = working.replace_compacted(replacement_prefix, None);
        working.set_reference_context_item(Some(reference.clone()));
        if let Some(world_state) = &world_state {
            working.set_world_state_baseline(WorldStateSnapshot::from(&world_state.state));
        }
        let checkpoint = self.compaction_checkpoint(
            &state,
            &working,
            replacement_history,
            CompactedHistoryMetadata {
                input_goal_ids: Default::default(),
                message: String::new(),
                window_number,
                window_ids: window_ids.clone(),
                compaction_response_id: Some(compaction_response_id),
                compaction_model_hash: step.settings.model_info.comp_hash.clone(),
                reviewer_compaction_hash: None,
            },
        );
        let items = std::iter::once(RolloutItem::Compacted(Box::new(checkpoint)))
            .chain(world_state.map(RolloutItem::WorldState))
            .chain([
                RolloutItem::TurnContext(reference),
                RolloutItem::EventMsg(settings_event),
            ])
            .collect::<Vec<_>>();
        // Before enqueue cancellation preserves the old state. A queued valid checkpoint may
        // finish after cancellation; native FIFO keeps later raw input after that checkpoint.
        if !owner_is_current(&state) {
            return false;
        }
        let acknowledged = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return false,
            result = live_thread.append_items(&items) => result,
        };
        if acknowledged.is_err() {
            return false;
        }
        if !owner_is_current(&state) {
            return false;
        }
        state.install_prepared_compaction(working, invalidate_reviews, window_ids.window_id);
        state.queue_pending_session_start_source(codex_hooks::SessionStartSource::Compact);
        if let Some(revision) = mcp_revision {
            self.services
                .executed_tool_calls
                .mark_mcp_attribution_persisted(revision);
        }
        true
    }

    pub(super) fn prepare_compaction_checkpoint_metadata(
        &self,
        items: &mut [ResponseItemEnvelope],
        compaction_model_hash: &Option<String>,
    ) -> Option<u64> {
        crate::model_search::SearchOwner::checkpoint(&self.services.thread_extension_data, items);
        let revision = self
            .services
            .executed_tool_calls
            .mcp_attribution_checkpoint(true)
            .and_then(|(attribution, revision)| {
                items.last_mut().map(|item| {
                    item.metadata.get_or_insert_default().mcp_attribution = Some(attribution);
                    revision
                })
            });
        if let Some(checkpoint) = items.iter_mut().rev().find(|item| {
            matches!(
                item.item,
                ResponseItem::Compaction { .. } | ResponseItem::ContextCompaction { .. }
            )
        }) {
            checkpoint
                .metadata
                .get_or_insert_default()
                .compaction_model_hash = compaction_model_hash.clone();
        }
        revision
    }

    pub(super) fn compaction_checkpoint(
        &self,
        state: &SessionState,
        history: &ContextManager,
        replacement_history: Vec<ResponseItemEnvelope>,
        metadata: CompactedHistoryMetadata,
    ) -> CompactedItem {
        CompactedItem {
            message: metadata.message,
            replacement_history: Some(replacement_history),
            guardian_history: history.guardian_history_checkpoint(),
            retained_context: Some(history.retained_context().clone()),
            mcp_resource_origins: self.services.mcp_runtime.resource_origin_checkpoint(),
            window_number: Some(metadata.window_number),
            first_window_id: Some(metadata.window_ids.first_window_id.to_string()),
            previous_window_id: metadata
                .window_ids
                .previous_window_id
                .map(|id| id.to_string()),
            window_id: Some(metadata.window_ids.window_id.to_string()),
            compaction_response_id: metadata.compaction_response_id,
            latest_token_usage_record: state.latest_token_usage_record.clone(),
            resume_metadata: Some(CompactionResumeMetadata {
                rejected_audio_file_ids: history.rejected_audio_file_ids(),
                dynamic_tools: Some(Box::new(DynamicToolsSnapshot {
                    thread_id: self.thread_id(),
                    tools: state.persisted_dynamic_tools.to_vec(),
                })),
                multi_agent_version: self.multi_agent_version(),
                last_started_turn_id: state.last_started_turn_id.clone(),
                turn_attribution: state.turn_attribution.clone(),
                previous_turn_settings: state.previous_turn_settings(),
            }),
        }
    }
}
