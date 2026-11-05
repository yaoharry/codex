//! One owned model-step recovery; only completed selected history may be summarized.

use super::body_recovery_collection::LocalSummaryCollectionError;
use super::body_recovery_collection::collect_local_summary_candidate;
use super::body_recovery_request::FrozenBody;
use super::preserving_compaction::*;
use super::*;
use crate::client::ModelClientSession;
use crate::responses_metadata::CodexResponsesMetadata;
use crate::responses_metadata::CompactionTurnMetadata;
use codex_api::EncodedJsonBody;
use codex_protocol::error::BodyRejectionRequestKind;
use codex_protocol::error::PreStreamBodyRejection;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

#[derive(Default)]
struct ProtectedBoundary {
    snapshot: StdMutex<Option<(u64, usize)>>,
    spent: AtomicBool,
}

pub(super) struct PreparedRetry {
    pub(super) body: EncodedJsonBody,
    pub(super) expected: ContextManager,
    pub(super) metadata: CodexResponsesMetadata,
    pub(super) owner: Option<u64>,
    pub(super) limit: u64,
    pub(super) rejected_bytes: usize,
}

pub(super) async fn capture_boundary(sess: &Session, turn: &TurnContext) {
    let history = sess.clone_history().await;
    let boundary = turn.extension_data.get_or_init(ProtectedBoundary::default);
    let mut boundary = boundary
        .snapshot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    boundary.get_or_insert((history.history_version(), history.annotated_items().len()));
}

pub(super) async fn validate_retry(
    sess: &Session,
    step: &StepContext,
    retry: &PreparedRetry,
    cancellation: &CancellationToken,
) -> bool {
    if retry.body.as_bytes().len() >= retry.rejected_bytes
        || retry.body.as_bytes().len() as u64 > retry.limit
    {
        return false;
    }
    let state = tokio::select! { biased;
        _ = cancellation.cancelled() => return false,
        state = sess.state.lock() => state,
    };
    let model = step.turn.capture_current_model_info();
    !cancellation.is_cancelled()
        && !state.shutting_down
        && state.last_started_turn_id.as_deref() == Some(step.turn.sub_id.as_str())
        && state.session_configuration.step_settings.as_ref() == step.settings.selected()
        && model.slug == step.settings.model_info.slug
        && model.comp_hash == step.settings.model_info.comp_hash
        && sess.services.model_client.auth_owner_generation() == retry.owner
        && state.history.history_version() == retry.expected.history_version()
        && state.history.annotated_items() == retry.expected.annotated_items()
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn prepare_retry(
    sess: &Arc<Session>,
    step: &Arc<StepContext>,
    expected: &ContextManager,
    metadata: &CodexResponsesMetadata,
    client: &mut ModelClientSession,
    rejection: &PreStreamBodyRejection,
    cancellation: &CancellationToken,
) -> CodexResult<Option<PreparedRetry>> {
    if rejection.request_kind() != BodyRejectionRequestKind::Generation
        || !client.supports_body_recovery()
        || sess
            .services
            .thread_extension_data
            .get::<crate::compact::AutoCompactionDisabled>()
            .is_some()
        || step
            .turn
            .config
            .features
            .enabled(codex_features::Feature::TokenBudget)
        || !matches!(
            step.turn.provider.capabilities().remote_compaction,
            codex_model_provider::RemoteCompactionSupport::Unsupported
        )
    {
        return Ok(None);
    }
    let Some(owner) = client.take_body_recovery_attempt() else {
        return Ok(None);
    };
    if cancellation.is_cancelled() || sess.services.model_client.auth_owner_generation() != owner {
        return Err(CodexErr::TurnAborted);
    }
    let boundary = step
        .turn
        .extension_data
        .get_or_init(ProtectedBoundary::default);
    // A new transport/model step does not refill the original turn's recovery budget.
    if boundary.spent.swap(true, Ordering::SeqCst) {
        return Ok(None);
    }
    let Some((generation, protected)) = *boundary
        .snapshot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    else {
        return Ok(None);
    };
    if generation != expected.history_version() || protected > expected.annotated_items().len() {
        return Ok(None);
    }
    let current_start = expected
        .annotated_items()
        .iter()
        .take(protected)
        .position(|item| {
            item.item.turn_id() == Some(step.turn.sub_id.as_str())
                && !is_host_summary(item)
                && !matches!(item.item, ResponseItem::Compaction { .. })
                && !is_retained_current_context(item)
        })
        .unwrap_or(protected);
    let protected_limit = protected_body_boundary(expected.annotated_items(), current_start);
    let Some(frame) = FrozenBody::capture(
        rejection.encoded_body(),
        expected,
        protected_limit,
        client,
        &step.settings.model_info,
        metadata,
    )?
    else {
        return Ok(None);
    };
    let ends = completed_prefix_ends(
        expected.annotated_items(),
        protected_limit.min(frame.aligned),
    );
    if ends.is_empty() {
        return Ok(None);
    }
    let mut compact_metadata = sess
        .compaction_responses_metadata(&step.turn, CompactionTurnMetadata::body_limit())
        .await;
    compact_metadata.tool_namespaces_info = metadata.tool_namespaces_info.clone();
    let prompt = sess.response_item_from_user_input(vec![UserInput::Text {
        text: step
            .turn
            .config
            .compact_prompt
            .clone()
            .unwrap_or_else(|| crate::compact::SUMMARIZATION_PROMPT.to_owned()),
        text_elements: Vec::new(),
    }]);
    let Some(prompt) =
        client.project_body_recovery_item(&prompt, &step.settings.model_info, &compact_metadata)?
    else {
        return Ok(None);
    };
    let trace = sess.services.rollout_thread_trace.inference_trace_context(
        &step.turn.sub_id,
        &step.settings.model_info.slug,
        &step.turn.provider.info().name,
    );
    let mut limit = rejection.limit_bytes();
    let mut upper = ends.len();
    for _ in 0..3 {
        let mut low = 0;
        let mut high = upper;
        let mut chosen = None;
        while low < high {
            let middle = low + (high - low) / 2;
            let body = frame.compaction(ends[middle], &prompt, &compact_metadata)?;
            if body.as_bytes().len() as u64 <= limit {
                chosen = Some((middle, body));
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        let Some((index, compact_body)) = chosen else {
            return Ok(None);
        };
        let selected = ends[index];
        let source_check = PreparedRetry {
            body: compact_body.clone(),
            expected: expected.clone(),
            metadata: compact_metadata.clone(),
            owner,
            limit,
            rejected_bytes: usize::MAX,
        };
        if !validate_retry(sess, step, &source_check, cancellation).await {
            return Ok(None);
        }
        let stream = tokio::select! { biased;
            _ = cancellation.cancelled() => return Err(CodexErr::TurnAborted),
            stream = client.stream_body_recovery_request(compact_body, &step.settings.model_info,
                &step.session_telemetry, &compact_metadata, owner, BodyRejectionRequestKind::PrefixCompaction, &trace) => stream,
        };
        let stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                if let Some(rejection) = error.pre_stream_body_rejection()
                    && rejection.request_kind() == BodyRejectionRequestKind::PrefixCompaction
                {
                    limit = limit.min(rejection.limit_bytes());
                    upper = index;
                    continue;
                }
                return Err(error);
            }
        };
        let collected =
            match collect_local_summary_candidate(&step.turn.sub_id, stream, cancellation).await {
                Ok(collected) => collected,
                Err(LocalSummaryCollectionError::Cancelled) => return Err(CodexErr::TurnAborted),
                Err(LocalSummaryCollectionError::Stream(error)) => {
                    if let CodexErrorDetails::ResponseIncomplete(incomplete) = error.details() {
                        sess.record_incomplete_response_usage(
                            &step.turn,
                            &step.settings,
                            incomplete,
                        )
                        .await;
                    }
                    return Err(*error);
                }
                Err(_) => return Ok(None),
            };
        let (response_id, candidate, usage, usage_metadata) = collected.into_parts();
        sess.record_observed_response_completed(
            &step.turn,
            &response_id,
            usage.as_ref(),
            usage_metadata.as_ref(),
        )
        .await;
        sess.record_token_usage_info(
            &step.turn,
            &step.settings,
            usage.as_ref(),
            TokenUsageContext::SpendingOnly,
        )
        .await?;
        if cancellation.is_cancelled() {
            return Err(CodexErr::TurnAborted);
        }
        let Some(candidate) = candidate else {
            return Ok(None);
        };
        let mut summary = PreservingSummary::local(candidate);
        summary.assign_id(sess);
        let Some(window) = sess.prepare_preserving_window(expected, cancellation).await else {
            return Ok(None);
        };
        let mut retry_metadata = metadata.clone();
        window.apply_metadata(&mut retry_metadata);
        let sampled = crate::context_manager::sampled_through(expected.annotated_items());
        let retained = expected.annotated_items()[..selected]
            .iter()
            .map(|item| {
                retain_for_policy(item, HistoricalUserPolicy::SummarizeKnownSampled, sampled)
            })
            .collect::<Vec<_>>();
        let Some(wire_summary) = client.project_body_recovery_item(
            &summary.item().item,
            &step.settings.model_info,
            &retry_metadata,
        )?
        else {
            return Ok(None);
        };
        let retry_body = frame.retry(selected, &retained, &wire_summary, &retry_metadata)?;
        if retry_body.as_bytes().len() as u64 > limit
            || retry_body.as_bytes().len() >= rejection.encoded_body().as_bytes().len()
        {
            return Ok(None);
        }
        let retained_items = expected.annotated_items()[..selected]
            .iter()
            .zip(&retained)
            .filter(|(_, keep)| **keep)
            .map(|(item, _)| item.clone())
            .collect();
        let Some(receipt) = sess
            .install_body_recovery_prefix(
                step,
                expected,
                protected,
                CompactionPrefix {
                    generation,
                    len: selected,
                    auth_owner_generation: owner,
                },
                retained_items,
                &summary,
                Some(&window),
                HistoricalUserPolicy::SummarizeKnownSampled,
                cancellation,
            )
            .await
        else {
            return Ok(None);
        };
        *boundary
            .snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some((receipt.history.history_version(), receipt.protected_start));
        let Some(installed) = receipt
            .history
            .annotated_items()
            .iter()
            .find(|item| item.item.id() == summary.item().item.id())
        else {
            return Ok(None);
        };
        let Some(installed_wire) = client.project_body_recovery_item(
            &installed.item,
            &step.settings.model_info,
            &retry_metadata,
        )?
        else {
            return Ok(None);
        };
        let body = frame.retry(selected, &retained, &installed_wire, &retry_metadata)?;
        let prepared = PreparedRetry {
            body,
            expected: receipt.history,
            metadata: retry_metadata,
            owner,
            limit,
            rejected_bytes: rejection.encoded_body().as_bytes().len(),
        };
        if !validate_retry(sess, step, &prepared, cancellation).await {
            return Ok(None);
        }
        return Ok(Some(prepared));
    }
    Ok(None)
}
