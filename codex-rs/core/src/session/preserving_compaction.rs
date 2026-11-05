//! Installs a completed prefix only after its exact preserved suffix has a durable checkpoint.

use super::body_recovery_summary::LocalSummaryCandidate;
use super::*;
use crate::compact_retained_tool_outputs::is_retained_tool_output;
use codex_protocol::models::ContentItemProvenance;
use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;

// Bound recovery's additional owner/settings hold, not the native interruption cleanup.
const BODY_RECOVERY_CHECKPOINT_TIMEOUT: Duration = Duration::from_secs(5);

/// Output success is a runtime hint omitted by FunctionCallOutputPayload serialization.
/// Compare every other typed item field and order; never modify the live source history.
pub(super) fn canonical_response_items_match(
    canonical: &[ResponseItemEnvelope],
    expected: &[ResponseItemEnvelope],
) -> bool {
    canonical.len() == expected.len()
        && canonical.iter().zip(expected).all(|(left, right)| {
            if left.item == right.item {
                return true;
            }
            let mut left = left.item.clone();
            let mut right = right.item.clone();
            for item in [&mut left, &mut right] {
                if let ResponseItem::FunctionCallOutput { output, .. }
                | ResponseItem::CustomToolCallOutput { output, .. } = item
                {
                    output.success = None;
                }
            }
            left == right
        })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum HistoricalUserPolicy {
    Retain,
    SummarizeKnownSampled,
}

pub(crate) struct PreservingSummary {
    item: ResponseItemEnvelope,
    message: String,
    response_id: String,
}

impl PreservingSummary {
    #[cfg(test)]
    pub(crate) fn encrypted(item: ResponseItemEnvelope, response_id: String) -> Option<Self> {
        if response_id.trim().is_empty()
            || !matches!(&item.item, ResponseItem::Compaction { encrypted_content, .. } if !encrypted_content.trim().is_empty())
        {
            return None;
        }
        Some(Self {
            item,
            message: String::new(),
            response_id,
        })
    }

    pub(crate) fn local(candidate: LocalSummaryCandidate) -> Self {
        let (item, message, response_id) = candidate.into_parts();
        Self {
            item,
            message,
            response_id,
        }
    }

    pub(crate) fn item(&self) -> &ResponseItemEnvelope {
        &self.item
    }

    pub(crate) fn assign_id(&mut self, session: &Session) {
        session.assign_missing_response_item_id(&mut self.item.item);
    }
}

pub(crate) struct PreservingWindow {
    source_number: u64,
    source_ids: crate::state::AutoCompactWindowIds,
    target_number: u64,
    target_ids: crate::state::AutoCompactWindowIds,
}

impl PreservingWindow {
    pub(crate) fn apply_metadata(
        &self,
        metadata: &mut crate::responses_metadata::CodexResponsesMetadata,
    ) {
        metadata.window_id = format!("{}:{}", metadata.thread_id, self.target_number);
        metadata.window_number = Some(self.target_number);
        metadata.context_window_id = Some(self.target_ids.window_id.clone());
    }
}

pub(crate) struct PreservingPrefixReceipt {
    pub(crate) history: ContextManager,
    pub(crate) protected_start: usize,
}

pub(crate) fn is_host_summary(envelope: &ResponseItemEnvelope) -> bool {
    let ResponseItem::Message {
        role,
        content,
        internal_chat_message_metadata_passthrough: Some(metadata),
        ..
    } = &envelope.item
    else {
        return false;
    };
    role == "user"
        && content.len() == 1
        && metadata.content_item_kinds.as_ref().is_some_and(|kinds| {
            kinds.len() == 1 && kinds[0].as_str() == crate::context::CompactionSummary::KIND
        })
        && metadata
            .content_item_metadata
            .as_ref()
            .is_some_and(|parts| {
                parts.len() == 1
                    && matches!(
                        parts[0].provenance,
                        ContentItemProvenance::Harness { hook: false }
                    )
            })
}

/// Stock compaction recreates these current-tagged context items before current user input.
/// They may cross a selected boundary only because the replacement retains them exactly.
pub(crate) fn is_retained_current_context(envelope: &ResponseItemEnvelope) -> bool {
    match &envelope.item {
        ResponseItem::Message { role, .. } if role == "developer" || role == "system" => true,
        ResponseItem::AdditionalTools { .. } | ResponseItem::ConfigurationUpdate { .. } => true,
        ResponseItem::Message {
            role,
            content,
            encrypted_content: None,
            internal_chat_message_metadata_passthrough: Some(metadata),
            ..
        } if role == "user"
            && !content.is_empty()
            && content.iter().all(|part| {
                matches!(part, codex_protocol::models::ContentItem::InputText { .. })
            })
            && !crate::context::is_user_authorization_message(&envelope.item) =>
        {
            match (
                &metadata.content_item_kinds,
                &metadata.content_item_metadata,
            ) {
                (Some(kinds), Some(parts))
                    if kinds.len() == content.len() && parts.len() == content.len() =>
                {
                    kinds.iter().zip(parts).all(|(kind, part)| {
                        matches!(
                            (kind.as_str(), &part.provenance),
                            (
                                "environments.environment_context",
                                ContentItemProvenance::Harness { hook: false }
                            ) | ("agents_md.instructions", ContentItemProvenance::AgentsMd)
                        )
                    })
                }
                _ => false,
            }
        }
        _ => false,
    }
}

pub(crate) fn protected_body_boundary(items: &[ResponseItemEnvelope], captured: usize) -> usize {
    let sampled = crate::context_manager::sampled_through(items);
    items
        .iter()
        .take(captured)
        .position(|item| {
            let authorization = crate::context::is_user_authorization_message(&item.item)
                || matches!(item.item, ResponseItem::AgentMessage { .. });
            authorization
                && !is_host_summary(item)
                && item
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.history_sequence)
                    .is_none_or(|sequence| sequence > sampled)
        })
        .unwrap_or(captured)
}

pub(crate) fn retain_for_policy(
    envelope: &ResponseItemEnvelope,
    policy: HistoricalUserPolicy,
    sampled: u64,
) -> bool {
    if policy == HistoricalUserPolicy::Retain {
        return retain_for_body_recovery(envelope);
    }
    if is_host_summary(envelope) {
        return false;
    }
    if let ResponseItem::Message {
        role,
        content,
        internal_chat_message_metadata_passthrough: Some(source),
        ..
    } = &envelope.item
        && role == "user"
        && !content.is_empty()
        && crate::context::UserGoalUpdate::message_text(&envelope.item).is_none()
        && source.content_item_kinds.as_ref().is_some_and(|kinds| {
            kinds.len() == content.len()
                && kinds.iter().all(|kind| kind.as_str().starts_with("user."))
        })
        && source.content_item_metadata.as_ref().is_some_and(|parts| {
            parts.len() == content.len()
                && parts
                    .iter()
                    .all(|part| matches!(part.provenance, ContentItemProvenance::User))
        })
        && envelope.metadata.as_ref().is_some_and(|metadata| {
            !metadata.inherited_user_message
                && metadata
                    .history_sequence
                    .is_some_and(|sequence| sequence <= sampled)
        })
    {
        return false;
    }
    retain_for_body_recovery(envelope)
}

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
            _ if is_host_summary(envelope) && pending.is_empty() => ends.push(index + 1),
            ResponseItem::Reasoning { .. } => {}
            _ if retain_for_body_recovery(envelope) => {}
            _ => break,
        }
    }
    ends
}

impl Session {
    pub(crate) async fn prepare_preserving_window(
        &self,
        expected: &ContextManager,
        cancellation: &CancellationToken,
    ) -> Option<PreservingWindow> {
        let state = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return None,
            state = self.state.lock() => state,
        };
        if state.history.history_version() != expected.history_version()
            || state.history.annotated_items() != expected.annotated_items()
        {
            return None;
        }
        let (target_number, target_ids) = state.prepare_compaction_window();
        Some(PreservingWindow {
            source_number: state.auto_compact_window_number(),
            source_ids: state.auto_compact_window_ids().clone(),
            target_number,
            target_ids,
        })
    }

    #[cfg(test)]
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
        let Some(item) = replacement_prefix.pop() else {
            return false;
        };
        let Some(summary) = PreservingSummary::encrypted(item, compaction_response_id) else {
            return false;
        };
        self.install_body_recovery_prefix(
            step,
            expected,
            protected_start,
            prefix,
            replacement_prefix,
            &summary,
            None,
            HistoricalUserPolicy::Retain,
            cancellation,
        )
        .await
        .is_some()
    }

    // The bounded owner hold keeps the canonical append/ACK and local adoption linearized.
    #[expect(
        clippy::await_holding_invalid_type,
        reason = "bounded durable ACK linearization preserves source ownership and native FIFO"
    )]
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn install_body_recovery_prefix(
        &self,
        step: &StepContext,
        expected: &ContextManager,
        protected_start: usize,
        prefix: CompactionPrefix,
        mut replacement_prefix: Vec<ResponseItemEnvelope>,
        summary: &PreservingSummary,
        window: Option<&PreservingWindow>,
        policy: HistoricalUserPolicy,
        cancellation: &CancellationToken,
    ) -> Option<PreservingPrefixReceipt> {
        let selected_boundary = if policy == HistoricalUserPolicy::SummarizeKnownSampled {
            protected_body_boundary(expected.annotated_items(), protected_start)
        } else {
            protected_start
        };
        let sampled = crate::context_manager::sampled_through(expected.annotated_items());
        if cancellation.is_cancelled()
            || prefix.generation != expected.history_version()
            || prefix.len == 0
            || prefix.len > selected_boundary
            || !completed_prefix_ends(expected.annotated_items(), selected_boundary)
                .contains(&prefix.len)
            || expected.annotated_items()[..prefix.len].iter().any(|item| {
                item.item.turn_id() == Some(step.turn.sub_id.as_str())
                    && !is_host_summary(item)
                    && !matches!(item.item, ResponseItem::Compaction { .. })
                    && !is_retained_current_context(item)
            })
        {
            return None;
        }
        if !replacement_prefix
            .iter()
            .eq(expected.annotated_items()[..prefix.len]
                .iter()
                .filter(|item| retain_for_policy(item, policy, sampled)))
        {
            return None;
        }
        let relocated_boundary = replacement_prefix.len() + 1 + protected_start - prefix.len;
        replacement_prefix.push(summary.item.clone());
        tokio::time::timeout(BODY_RECOVERY_CHECKPOINT_TIMEOUT, async {
            let _settings = tokio::select! {
                biased;
                _ = cancellation.cancelled() => return None,
                settings = thread_settings::acquire_persistence_lock(self) => settings,
            };
            let settings_event = tokio::select! {
                biased;
                _ = cancellation.cancelled() => return None,
                event = thread_settings::applied_event(self) => event,
            };
            let live_thread = self.live_thread()?;
            let canonical = tokio::select! {
                biased;
                _ = cancellation.cancelled() => return None,
                result = self.services.thread_store.load_latest_model_context(codex_thread_store::LoadThreadHistoryParams {
                    thread_id: self.thread_id(),
                    include_archived: true,
                }) => result,
            };
            let Ok(canonical) = canonical else {
                return None;
            };
            let replay = self
                .reconstruct_history_from_rollout(&step.turn, &canonical.items)
                .await;
            if !canonical_response_items_match(&replay.history, expected.annotated_items()) {
                return None;
            }
            let mut state = tokio::select! {
                biased;
                _ = cancellation.cancelled() => return None,
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
                return None;
            }
            let world_state = state.history.world_state_checkpoint();
            let reference = step.to_turn_context_item();
            let (window_number, window_ids) = match window {
                Some(window)
                    if window.source_number == state.auto_compact_window_number()
                        && &window.source_ids == state.auto_compact_window_ids() =>
                {
                    (window.target_number, window.target_ids.clone())
                }
                Some(_) => return None,
                None => state.prepare_compaction_window(),
            };
            for envelope in &mut replacement_prefix {
                self.assign_missing_response_item_id(&mut envelope.item);
            }
            replacement_prefix
                .last_mut()?
                .metadata
                .get_or_insert_default()
                .compaction_model_hash = step.settings.model_info.comp_hash.clone();
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
                    message: summary.message.clone(),
                    window_number,
                    window_ids: window_ids.clone(),
                    compaction_response_id: Some(summary.response_id.clone()),
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
                return None;
            }
            let acknowledged = tokio::select! {
                biased;
                _ = cancellation.cancelled() => return None,
                result = live_thread.append_items(&items) => result,
            };
            if acknowledged.is_err() {
                return None;
            }
            if !owner_is_current(&state) {
                return None;
            }
            let installed_history = working.clone();
            state.install_prepared_compaction(working, invalidate_reviews, window_ids.window_id);
            state.queue_pending_session_start_source(codex_hooks::SessionStartSource::Compact);
            if let Some(revision) = mcp_revision {
                self.services
                    .executed_tool_calls
                    .mark_mcp_attribution_persisted(revision);
            }
            Some(PreservingPrefixReceipt {
                history: installed_history,
                protected_start: relocated_boundary,
            })
        })
        .await
        .ok()
        .flatten()
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
