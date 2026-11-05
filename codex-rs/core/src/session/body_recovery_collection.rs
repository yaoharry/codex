//! Collects one owned local compaction response without publishing or installing it.

use super::body_recovery_summary::LocalSummaryAttempt;
use super::body_recovery_summary::LocalSummaryCandidate;
use super::body_recovery_summary::build_local_summary_candidate;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use codex_protocol::ResponseUsageMetadata;
use codex_protocol::error::CodexErr;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub(crate) enum LocalSummaryCollectionError {
    Cancelled,
    Eof,
    InvalidTurnId,
    InvalidResponseId,
    ResponseIdMismatch,
    UnexpectedOutput,
    Stream(Box<CodexErr>),
}

/// Observed completion accounting survives even when the output cannot be installed.
pub(crate) struct CollectedLocalSummary {
    response_id: String,
    candidate: Option<LocalSummaryCandidate>,
    token_usage: Option<TokenUsage>,
    usage_metadata: Option<ResponseUsageMetadata>,
}

impl CollectedLocalSummary {
    #[cfg(test)]
    pub(crate) fn response_id(&self) -> &str {
        &self.response_id
    }

    #[cfg(test)]
    pub(crate) fn candidate(&self) -> Option<&LocalSummaryCandidate> {
        self.candidate.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn token_usage(&self) -> Option<&TokenUsage> {
        self.token_usage.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn usage_metadata(&self) -> Option<&ResponseUsageMetadata> {
        self.usage_metadata.as_ref()
    }

    /// Return completed response ID, optional candidate, and unaggregated usage.
    pub(crate) fn into_parts(
        self,
    ) -> (
        String,
        Option<LocalSummaryCandidate>,
        Option<TokenUsage>,
        Option<ResponseUsageMetadata>,
    ) {
        (
            self.response_id,
            self.candidate,
            self.token_usage,
            self.usage_metadata,
        )
    }
}

pub(crate) async fn collect_local_summary_candidate(
    original_turn_id: &str,
    mut stream: ResponseStream,
    cancellation: &CancellationToken,
) -> Result<CollectedLocalSummary, LocalSummaryCollectionError> {
    if original_turn_id.trim().is_empty() {
        return Err(LocalSummaryCollectionError::InvalidTurnId);
    }
    let mut created_response_id: Option<String> = None;
    let mut output = Vec::new();
    loop {
        let event = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(LocalSummaryCollectionError::Cancelled),
            event = stream.next() => event,
        };
        let Some(event) = event else {
            return Err(LocalSummaryCollectionError::Eof);
        };
        match event.map_err(|error| LocalSummaryCollectionError::Stream(Box::new(error)))? {
            ResponseEvent::Created {
                response_id: Some(response_id),
            } => {
                if response_id.trim().is_empty() {
                    return Err(LocalSummaryCollectionError::InvalidResponseId);
                }
                if created_response_id
                    .as_ref()
                    .is_some_and(|created| created != &response_id)
                {
                    return Err(LocalSummaryCollectionError::ResponseIdMismatch);
                }
                created_response_id = Some(response_id);
            }
            ResponseEvent::OutputItemAdded(item) if !is_summary_output(&item) => {
                return Err(LocalSummaryCollectionError::UnexpectedOutput);
            }
            ResponseEvent::OutputItemDone(item) => {
                if !is_summary_output(&item) {
                    return Err(LocalSummaryCollectionError::UnexpectedOutput);
                }
                output.push(item);
            }
            ResponseEvent::ToolCallInputDelta { .. } => {
                return Err(LocalSummaryCollectionError::UnexpectedOutput);
            }
            ResponseEvent::Completed {
                response_id,
                token_usage,
                usage_metadata,
                end_turn,
            } => {
                if response_id.trim().is_empty() {
                    return Err(LocalSummaryCollectionError::InvalidResponseId);
                }
                if created_response_id
                    .as_ref()
                    .is_some_and(|created| created != &response_id)
                {
                    return Err(LocalSummaryCollectionError::ResponseIdMismatch);
                }
                return Ok(finish_completed_response(
                    original_turn_id,
                    response_id,
                    &output,
                    token_usage,
                    usage_metadata,
                    end_turn,
                    cancellation,
                ));
            }
            _ => {}
        }
    }
}

fn is_summary_output(item: &ResponseItem) -> bool {
    matches!(item, ResponseItem::Reasoning { .. })
        || matches!(item, ResponseItem::Message { role, .. } if role == "assistant")
}

fn finish_completed_response(
    original_turn_id: &str,
    response_id: String,
    output: &[ResponseItem],
    token_usage: Option<TokenUsage>,
    usage_metadata: Option<ResponseUsageMetadata>,
    end_turn: Option<bool>,
    cancellation: &CancellationToken,
) -> CollectedLocalSummary {
    let mut candidate = if cancellation.is_cancelled() || end_turn == Some(false) {
        None
    } else {
        build_local_summary_candidate(
            original_turn_id,
            LocalSummaryAttempt::Completed {
                response_id: &response_id,
                output,
            },
        )
    };
    if cancellation.is_cancelled() {
        candidate = None;
    }
    CollectedLocalSummary {
        response_id,
        candidate,
        token_usage,
        usage_metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact::SUMMARY_PREFIX;
    use codex_protocol::error::CodexErrorDetails;
    use codex_protocol::error::ResponseIncompleteError;
    use pretty_assertions::assert_eq;
    use serde_json::json;
    use std::task::Poll;
    use tokio::sync::mpsc;

    fn assistant(text: &str) -> ResponseItem {
        serde_json::from_value(json!({
            "type":"message", "role":"assistant", "status":"completed",
            "content":[{"type":"output_text", "text":text}],
        }))
        .expect("synthetic assistant output")
    }

    fn reasoning() -> ResponseItem {
        serde_json::from_value(json!({
            "type":"reasoning", "summary":[], "content":null, "encrypted_content":null,
        }))
        .expect("synthetic reasoning output")
    }

    fn usage() -> TokenUsage {
        TokenUsage {
            input_tokens: 9,
            output_tokens: 4,
            total_tokens: 13,
            ..Default::default()
        }
    }

    fn metadata() -> ResponseUsageMetadata {
        ResponseUsageMetadata {
            amount: Some("0.001".to_owned()),
            metadata: Some(json!({"synthetic":"compaction-usage"})),
        }
    }

    fn completed(response_id: &str) -> ResponseEvent {
        ResponseEvent::Completed {
            response_id: response_id.to_owned(),
            token_usage: Some(usage()),
            usage_metadata: Some(metadata()),
            end_turn: None,
        }
    }

    fn stream(events: Vec<Result<ResponseEvent, CodexErr>>) -> (ResponseStream, CancellationToken) {
        let (tx_event, rx_event) = mpsc::channel(events.len().max(1));
        for event in events {
            tx_event.try_send(event).expect("test channel capacity");
        }
        drop(tx_event);
        let consumer_dropped = CancellationToken::new();
        (
            ResponseStream {
                routing_affinity: None,
                rx_event,
                interrupt: None,
                consumer_dropped: consumer_dropped.clone(),
            },
            consumer_dropped,
        )
    }

    #[tokio::test]
    async fn completed_reasoning_and_summary_preserve_identity_metadata_and_accounting() {
        let (stream, dropped) = stream(vec![
            Ok(ResponseEvent::Created {
                response_id: Some("synthetic-response".to_owned()),
            }),
            Ok(ResponseEvent::OutputItemAdded(assistant(
                "Added content is not collected.",
            ))),
            Ok(ResponseEvent::OutputTextDelta(
                "Delta content is not collected.".to_owned(),
            )),
            Ok(ResponseEvent::ReasoningContentDelta {
                delta: "not forwarded".to_owned(),
                content_index: 0,
            }),
            Ok(ResponseEvent::OutputItemDone(reasoning())),
            Ok(ResponseEvent::OutputItemDone(assistant(
                "This attempt's summary.",
            ))),
            Ok(completed("synthetic-response")),
        ]);
        let collected =
            collect_local_summary_candidate("synthetic-turn", stream, &CancellationToken::new())
                .await
                .expect("owned completed summary");
        assert!(dropped.is_cancelled());
        assert_eq!(collected.response_id(), "synthetic-response");
        assert_eq!(collected.token_usage(), Some(&usage()));
        assert_eq!(collected.usage_metadata(), Some(&metadata()));
        let candidate = collected.candidate().expect("completed candidate");
        assert_eq!(candidate.response_id(), "synthetic-response");
        assert_eq!(
            candidate.message(),
            format!("{SUMMARY_PREFIX}\nThis attempt's summary.")
        );
        assert_eq!(candidate.item().item.turn_id(), Some("synthetic-turn"));
        let (response_id, candidate, token_usage, usage_metadata) = collected.into_parts();
        assert_eq!(response_id, "synthetic-response");
        assert_eq!(
            candidate.expect("owned candidate").response_id(),
            response_id
        );
        assert_eq!(token_usage, Some(usage()));
        assert_eq!(usage_metadata, Some(metadata()));
    }

    #[tokio::test]
    async fn added_and_delta_content_cannot_replace_missing_completed_output() {
        let (stream, _) = stream(vec![
            Ok(ResponseEvent::OutputItemAdded(assistant("Not done."))),
            Ok(ResponseEvent::OutputTextDelta(
                "Not authoritative summary text.".to_owned(),
            )),
            Ok(completed("synthetic-response")),
        ]);
        let collected =
            collect_local_summary_candidate("synthetic-turn", stream, &CancellationToken::new())
                .await
                .expect("observed completion accounting");
        assert!(collected.candidate().is_none());
        assert_eq!(collected.token_usage(), Some(&usage()));
    }

    #[tokio::test]
    async fn partial_done_output_followed_by_eof_never_produces_candidate() {
        let (stream, dropped) = stream(vec![Ok(ResponseEvent::OutputItemDone(assistant(
            "Partial summary.",
        )))]);
        assert!(matches!(
            collect_local_summary_candidate("synthetic-turn", stream, &CancellationToken::new())
                .await,
            Err(LocalSummaryCollectionError::Eof)
        ));
        assert!(dropped.is_cancelled());
    }

    #[tokio::test]
    async fn cancellation_wins_over_ready_output_and_completion() {
        let (stream, dropped) = stream(vec![
            Ok(ResponseEvent::OutputItemDone(assistant("Ready summary."))),
            Ok(completed("synthetic-response")),
        ]);
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert!(matches!(
            collect_local_summary_candidate("synthetic-turn", stream, &cancellation).await,
            Err(LocalSummaryCollectionError::Cancelled)
        ));
        assert!(dropped.is_cancelled());
    }

    #[tokio::test]
    async fn cancellation_after_partial_output_drops_the_owned_stream() {
        let (tx_event, rx_event) = mpsc::channel(1);
        tx_event
            .try_send(Ok(ResponseEvent::OutputItemDone(assistant(
                "Partial summary.",
            ))))
            .expect("test channel capacity");
        let dropped = CancellationToken::new();
        let stream = ResponseStream {
            routing_affinity: None,
            rx_event,
            interrupt: None,
            consumer_dropped: dropped.clone(),
        };
        let cancellation = CancellationToken::new();
        let mut collection = Box::pin(collect_local_summary_candidate(
            "synthetic-turn",
            stream,
            &cancellation,
        ));
        assert!(matches!(futures::poll!(collection.as_mut()), Poll::Pending));
        cancellation.cancel();
        assert!(matches!(
            collection.await,
            Err(LocalSummaryCollectionError::Cancelled)
        ));
        assert!(dropped.is_cancelled());
        drop(tx_event);
    }

    #[tokio::test]
    async fn preserves_stream_failure_without_using_partial_summary() {
        let (stream, _) = stream(vec![
            Ok(ResponseEvent::OutputItemDone(assistant("Partial summary."))),
            Err(CodexErr::Stream("synthetic stream failure".to_owned())),
        ]);
        let result =
            collect_local_summary_candidate("synthetic-turn", stream, &CancellationToken::new())
                .await;
        let Err(LocalSummaryCollectionError::Stream(error)) = result else {
            panic!("original stream error must survive");
        };
        assert!(
            matches!(error.details(), CodexErrorDetails::Stream(message) if message == "synthetic stream failure")
        );
    }

    #[tokio::test]
    async fn preserves_failed_and_incomplete_error_categories_and_incomplete_usage() {
        for error in [
            CodexErr::ContextWindowExceeded,
            CodexErr::new(CodexErrorDetails::ResponseIncomplete(Box::new(
                ResponseIncompleteError {
                    partial_output: vec![json!({"type":"message", "status":"incomplete"})],
                    reason: "max_output_tokens".to_owned(),
                    response_id: Some("synthetic-incomplete-response".to_owned()),
                    token_usage: Some(usage()),
                },
            ))),
        ] {
            let (stream, _) = stream(vec![Err(error)]);
            let result = collect_local_summary_candidate(
                "synthetic-turn",
                stream,
                &CancellationToken::new(),
            )
            .await;
            let Err(LocalSummaryCollectionError::Stream(error)) = result else {
                panic!("upstream failed/incomplete error must survive");
            };
            match error.details() {
                CodexErrorDetails::ContextWindowExceeded => {}
                CodexErrorDetails::ResponseIncomplete(incomplete) => {
                    assert_eq!(
                        incomplete.response_id.as_deref(),
                        Some("synthetic-incomplete-response")
                    );
                    assert_eq!(incomplete.token_usage, Some(usage()));
                    assert_eq!(
                        incomplete.partial_output,
                        vec![json!({"type":"message", "status":"incomplete"})]
                    );
                }
                _ => panic!("error category changed"),
            }
        }
    }

    #[tokio::test]
    async fn every_tool_indication_rejects_added_done_and_delta_signals() {
        for value in [
            json!({"type":"function_call", "name":"synthetic_tool", "call_id":"synthetic-call", "arguments":"{}"}),
            json!({"type":"function_call_output", "call_id":"synthetic-call", "output":"synthetic result"}),
            json!({"type":"custom_tool_call", "name":"synthetic_tool", "call_id":"synthetic-call", "input":"synthetic input"}),
            json!({"type":"custom_tool_call_output", "call_id":"synthetic-call", "output":"synthetic result"}),
        ] {
            let item: ResponseItem = serde_json::from_value(value).expect("synthetic tool signal");
            for event in [
                ResponseEvent::OutputItemAdded(item.clone()),
                ResponseEvent::OutputItemDone(item),
            ] {
                let (stream, dropped) = stream(vec![
                    Ok(ResponseEvent::OutputItemDone(assistant(
                        "Otherwise valid summary.",
                    ))),
                    Ok(event),
                    Ok(completed("synthetic-response")),
                ]);
                assert!(matches!(
                    collect_local_summary_candidate(
                        "synthetic-turn",
                        stream,
                        &CancellationToken::new()
                    )
                    .await,
                    Err(LocalSummaryCollectionError::UnexpectedOutput)
                ));
                assert!(dropped.is_cancelled());
            }
        }
        let (stream, _) = stream(vec![Ok(ResponseEvent::ToolCallInputDelta {
            item_id: "synthetic-item".to_owned(),
            call_id: Some("synthetic-call".to_owned()),
            delta: "{}".to_owned(),
        })]);
        assert!(matches!(
            collect_local_summary_candidate("synthetic-turn", stream, &CancellationToken::new())
                .await,
            Err(LocalSummaryCollectionError::UnexpectedOutput)
        ));
    }

    #[tokio::test]
    async fn empty_hidden_and_unfinished_completed_output_preserve_usage_without_candidate() {
        for output in [
            Vec::new(),
            vec![reasoning()],
            vec![assistant(" \t\n")],
            vec![assistant(
                "<oai-mem-citation>synthetic-source</oai-mem-citation>",
            )],
            vec![assistant(
                "<proposed_plan>\n- synthetic hidden plan\n</proposed_plan>",
            )],
        ] {
            let events = output
                .into_iter()
                .map(|item| Ok(ResponseEvent::OutputItemDone(item)))
                .chain(std::iter::once(Ok(completed("synthetic-response"))))
                .collect();
            let (stream, _) = stream(events);
            let collected = collect_local_summary_candidate(
                "synthetic-turn",
                stream,
                &CancellationToken::new(),
            )
            .await
            .expect("completed accounting");
            assert!(collected.candidate().is_none());
            assert_eq!(collected.token_usage(), Some(&usage()));
            assert_eq!(collected.usage_metadata(), Some(&metadata()));
        }
        let mut completion = completed("synthetic-response");
        let ResponseEvent::Completed { end_turn, .. } = &mut completion else {
            unreachable!()
        };
        *end_turn = Some(false);
        let (stream, _) = stream(vec![
            Ok(ResponseEvent::OutputItemDone(assistant(
                "Response did not finish its turn.",
            ))),
            Ok(completion),
        ]);
        let collected =
            collect_local_summary_candidate("synthetic-turn", stream, &CancellationToken::new())
                .await
                .expect("completed accounting");
        assert!(collected.candidate().is_none());
        assert_eq!(collected.token_usage(), Some(&usage()));
    }

    #[tokio::test]
    async fn response_identity_must_match_created_when_available() {
        for events in [
            vec![
                Ok(ResponseEvent::Created {
                    response_id: Some("synthetic-first".to_owned()),
                }),
                Ok(completed("synthetic-other")),
            ],
            vec![
                Ok(ResponseEvent::Created {
                    response_id: Some("synthetic-first".to_owned()),
                }),
                Ok(ResponseEvent::Created {
                    response_id: Some("synthetic-other".to_owned()),
                }),
            ],
            vec![
                Ok(ResponseEvent::Created {
                    response_id: Some("synthetic-first".to_owned()),
                }),
                Ok(ResponseEvent::Created { response_id: None }),
                Ok(completed("synthetic-other")),
            ],
        ] {
            let (stream, _) = stream(events);
            assert!(matches!(
                collect_local_summary_candidate(
                    "synthetic-turn",
                    stream,
                    &CancellationToken::new()
                )
                .await,
                Err(LocalSummaryCollectionError::ResponseIdMismatch)
            ));
        }
    }

    #[tokio::test]
    async fn rejects_blank_created_completed_and_original_turn_ids() {
        for id in ["", " \t\n"] {
            for event in [
                ResponseEvent::Created {
                    response_id: Some(id.to_owned()),
                },
                completed(id),
            ] {
                let (stream, _) = stream(vec![Ok(event)]);
                assert!(matches!(
                    collect_local_summary_candidate(
                        "synthetic-turn",
                        stream,
                        &CancellationToken::new()
                    )
                    .await,
                    Err(LocalSummaryCollectionError::InvalidResponseId)
                ));
            }
            let (stream, _) = stream(vec![Ok(completed("synthetic-response"))]);
            assert!(matches!(
                collect_local_summary_candidate(id, stream, &CancellationToken::new()).await,
                Err(LocalSummaryCollectionError::InvalidTurnId)
            ));
        }
    }

    #[tokio::test]
    async fn permits_legacy_missing_created_identity_and_duplicate_matching_created() {
        for created in [
            Vec::new(),
            vec![ResponseEvent::Created { response_id: None }],
            vec![
                ResponseEvent::Created {
                    response_id: Some("synthetic-response".to_owned()),
                },
                ResponseEvent::Created {
                    response_id: Some("synthetic-response".to_owned()),
                },
            ],
        ] {
            let events = created
                .into_iter()
                .map(Ok)
                .chain([
                    Ok(ResponseEvent::OutputItemDone(assistant("Legacy summary."))),
                    Ok(completed("synthetic-response")),
                ])
                .collect();
            let (stream, _) = stream(events);
            let collected = collect_local_summary_candidate(
                "synthetic-turn",
                stream,
                &CancellationToken::new(),
            )
            .await
            .expect("legacy completed response");
            assert_eq!(
                collected.candidate().expect("candidate").response_id(),
                "synthetic-response"
            );
        }
    }

    #[test]
    fn cancellation_after_observed_completion_retains_accounting_without_candidate() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let collected = finish_completed_response(
            "synthetic-turn",
            "synthetic-response".to_owned(),
            &[assistant("Completed summary.")],
            Some(usage()),
            Some(metadata()),
            Some(true),
            &cancellation,
        );
        assert!(collected.candidate().is_none());
        assert_eq!(collected.response_id(), "synthetic-response");
        assert_eq!(collected.token_usage(), Some(&usage()));
        assert_eq!(collected.usage_metadata(), Some(&metadata()));
    }
}
