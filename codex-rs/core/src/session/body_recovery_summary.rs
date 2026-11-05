//! Builds an owned text summary from one completed local compaction attempt.

use crate::compact::SUMMARY_PREFIX;
use crate::context::CompactionSummary;
use crate::context::ContextualUserFragment;
use crate::session::turn::get_last_assistant_message_from_turn;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_utils_stream_parser::strip_proposed_plan_blocks;

/// The collector owns response identity and provenance for this single attempt.
pub(crate) enum LocalSummaryAttempt<'a> {
    Completed {
        response_id: &'a str,
        output: &'a [ResponseItem],
    },
    #[cfg(test)]
    Incomplete,
    #[cfg(test)]
    Failed,
    #[cfg(test)]
    Eof,
    #[cfg(test)]
    Cancelled,
}

/// Validated, host-classified summary; prefix selection and installation stay separate.
pub(crate) struct LocalSummaryCandidate {
    response_id: String,
    message: String,
    item: ResponseItemEnvelope,
}

impl LocalSummaryCandidate {
    #[cfg(test)]
    pub(crate) fn response_id(&self) -> &str {
        &self.response_id
    }

    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.message
    }

    #[cfg(test)]
    pub(crate) fn item(&self) -> &ResponseItemEnvelope {
        &self.item
    }

    /// Return the summary envelope, framed checkpoint message, and completed response ID.
    pub(crate) fn into_parts(self) -> (ResponseItemEnvelope, String, String) {
        (self.item, self.message, self.response_id)
    }
}

pub(crate) fn build_local_summary_candidate(
    original_turn_id: &str,
    attempt: LocalSummaryAttempt<'_>,
) -> Option<LocalSummaryCandidate> {
    let (response_id, output) = match attempt {
        LocalSummaryAttempt::Completed {
            response_id,
            output,
        } => (response_id, output),
        #[cfg(test)]
        _ => return None,
    };
    if original_turn_id.trim().is_empty() || response_id.trim().is_empty() {
        return None;
    }
    for item in output {
        match item {
            ResponseItem::Reasoning { .. } => {}
            ResponseItem::Message {
                role,
                status,
                encrypted_content,
                content,
                ..
            } if role == "assistant"
                && status.as_deref().is_none_or(|status| status == "completed")
                && encrypted_content.is_none()
                && content
                    .iter()
                    .all(|part| matches!(part, ContentItem::OutputText { .. })) => {}
            _ => return None,
        }
    }
    let summary = get_last_assistant_message_from_turn(output.iter())?;
    let summary = strip_proposed_plan_blocks(&summary);
    if summary.trim().is_empty()
        || summary.trim() == "(no summary available)"
        || summary.trim() == SUMMARY_PREFIX.trim()
    {
        return None;
    }
    let message = format!("{SUMMARY_PREFIX}\n{summary}");
    let mut item = ContextualUserFragment::into(CompactionSummary::new(message.clone()));
    item.set_turn_id_if_missing(original_turn_id);
    Some(LocalSummaryCandidate {
        response_id: response_id.to_owned(),
        message,
        item: ResponseItemEnvelope::new(item),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::is_user_authorization_message;
    use codex_protocol::models::ContentItemKind;
    use codex_protocol::models::ContentItemMetadata;
    use codex_protocol::models::InternalChatMessageMetadataPassthrough;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    fn assistant(text: &str) -> ResponseItem {
        serde_json::from_value(json!({
            "type": "message",
            "id": "synthetic-source-message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }))
        .expect("synthetic assistant output")
    }

    fn reasoning() -> ResponseItem {
        serde_json::from_value(json!({
            "type": "reasoning",
            "summary": [],
            "content": null,
            "encrypted_content": null,
        }))
        .expect("synthetic reasoning output")
    }

    fn completed(output: &[ResponseItem]) -> Option<LocalSummaryCandidate> {
        build_local_summary_candidate(
            "synthetic-original-turn",
            LocalSummaryAttempt::Completed {
                response_id: "synthetic-completed-response",
                output,
            },
        )
    }

    #[test]
    fn completed_plaintext_summary_has_exact_host_metadata_and_response_binding() {
        let output = [
            reasoning(),
            assistant("Completed work.\nKeep the next step."),
        ];
        let original = output.clone();
        let candidate = completed(&output).expect("completed plaintext summary");
        let expected_message = format!("{SUMMARY_PREFIX}\nCompleted work.\nKeep the next step.");
        assert_eq!(candidate.response_id(), "synthetic-completed-response");
        assert_eq!(candidate.message(), expected_message);
        assert_eq!(candidate.item().metadata, None);
        let ResponseItem::Message {
            id,
            role,
            status,
            phase,
            encrypted_content,
            content,
            internal_chat_message_metadata_passthrough,
        } = &candidate.item().item
        else {
            panic!("local summary must be a classified message");
        };
        assert_eq!(id, &None);
        assert_eq!(role, "user");
        assert_eq!(status, &None);
        assert_eq!(phase, &None);
        assert_eq!(encrypted_content, &None);
        assert_eq!(
            content,
            &vec![ContentItem::InputText {
                text: expected_message.clone()
            }]
        );
        assert_eq!(
            internal_chat_message_metadata_passthrough,
            &Some(InternalChatMessageMetadataPassthrough {
                turn_id: Some("synthetic-original-turn".to_owned()),
                content_item_kinds: Some(vec![ContentItemKind("compaction.summary".to_owned())]),
                content_item_metadata: Some(vec![ContentItemMetadata::harness()]),
                ..Default::default()
            })
        );
        assert!(!is_user_authorization_message(&candidate.item().item));
        assert_eq!(output, original);
        let item = candidate.item().clone();
        let (owned_item, message, response_id) = candidate.into_parts();
        assert_eq!(owned_item, item);
        assert_eq!(message, expected_message);
        assert_eq!(response_id, "synthetic-completed-response");
    }

    #[test]
    fn candidate_owns_this_attempt_id_and_output_without_session_fallback() {
        let mut response_id = "synthetic-owned-response".to_owned();
        let mut output = vec![assistant("This attempt's summary.")];
        let candidate = build_local_summary_candidate(
            "synthetic-original-turn",
            LocalSummaryAttempt::Completed {
                response_id: &response_id,
                output: &output,
            },
        )
        .expect("owned candidate");
        response_id.clear();
        output.clear();
        assert_eq!(candidate.response_id(), "synthetic-owned-response");
        assert_eq!(
            candidate.message(),
            format!("{SUMMARY_PREFIX}\nThis attempt's summary.")
        );
        assert!(completed(&output).is_none());
    }

    #[test]
    fn completed_summary_uses_only_usable_assistant_text_from_this_response() {
        let output = [
            assistant("First summary."),
            assistant("Final summary.<oai-mem-citation>synthetic-source</oai-mem-citation>"),
            reasoning(),
        ];
        let candidate = completed(&output).expect("usable current response output");
        assert_eq!(
            candidate.message(),
            format!("{SUMMARY_PREFIX}\nFinal summary.")
        );
    }

    #[test]
    fn rejects_empty_reasoning_citation_hidden_and_placeholder_only_output() {
        assert!(completed(&[]).is_none());
        assert!(completed(&[reasoning()]).is_none());
        for text in [
            "",
            " \t\n",
            "<oai-mem-citation>synthetic-source</oai-mem-citation>",
            "<proposed_plan>\n- synthetic hidden plan\n</proposed_plan>",
            "(no summary available)",
            " \n(no summary available)\t",
            SUMMARY_PREFIX,
        ] {
            assert!(
                completed(&[assistant(text)]).is_none(),
                "unusable output: {text:?}"
            );
        }
    }

    #[test]
    fn rejects_incomplete_failed_eof_and_cancelled_attempts() {
        for attempt in [
            LocalSummaryAttempt::Incomplete,
            LocalSummaryAttempt::Failed,
            LocalSummaryAttempt::Eof,
            LocalSummaryAttempt::Cancelled,
        ] {
            assert!(build_local_summary_candidate("synthetic-original-turn", attempt).is_none());
        }
    }

    #[test]
    fn requires_nonblank_original_turn_and_completed_response_ids() {
        let output = [assistant("Usable summary.")];
        for missing in ["", " \t\n"] {
            assert!(
                build_local_summary_candidate(
                    missing,
                    LocalSummaryAttempt::Completed {
                        response_id: "synthetic-response",
                        output: &output
                    },
                )
                .is_none()
            );
            assert!(
                build_local_summary_candidate(
                    "synthetic-turn",
                    LocalSummaryAttempt::Completed {
                        response_id: missing,
                        output: &output
                    },
                )
                .is_none()
            );
        }
    }

    #[test]
    fn rejects_wrong_roles_incomplete_message_status_and_media_output() {
        for role in ["user", "developer", "system", "tool"] {
            let mut item = assistant("Not an assistant summary.");
            let ResponseItem::Message {
                role: item_role, ..
            } = &mut item
            else {
                unreachable!()
            };
            *item_role = role.to_owned();
            assert!(
                completed(std::slice::from_ref(&item)).is_none(),
                "role: {role}"
            );
            assert!(completed(&[assistant("Otherwise valid summary."), item]).is_none());
        }
        for status in [
            "in_progress",
            "incomplete",
            "failed",
            "cancelled",
            "unknown",
            "",
        ] {
            let mut item = assistant("Unfinished output.");
            let ResponseItem::Message {
                status: item_status,
                ..
            } = &mut item
            else {
                unreachable!()
            };
            *item_status = Some(status.to_owned());
            assert!(completed(&[item]).is_none(), "status: {status}");
        }
        let mut item = assistant("Text alongside media is not a plaintext summary.");
        let ResponseItem::Message { content, .. } = &mut item else {
            unreachable!()
        };
        content.push(
            serde_json::from_value(json!({
                "type": "input_image", "image_url": "https://example.invalid/synthetic.png",
            }))
            .expect("synthetic image"),
        );
        assert!(completed(&[item]).is_none());
        let mut item = assistant("Plaintext cannot authorize an opaque message.");
        let ResponseItem::Message {
            encrypted_content, ..
        } = &mut item
        else {
            unreachable!()
        };
        *encrypted_content = Some("synthetic opaque content".to_owned());
        assert!(completed(&[item]).is_none());
    }

    #[test]
    fn rejects_executable_and_tool_output_items_even_with_valid_summary_text() {
        for value in [
            json!({"type":"function_call", "name":"synthetic_tool", "call_id":"synthetic-call", "arguments":"{}"}),
            json!({"type":"function_call_output", "call_id":"synthetic-call", "output":"synthetic result"}),
            json!({"type":"custom_tool_call", "name":"synthetic_tool", "call_id":"synthetic-call", "input":"synthetic input"}),
            json!({"type":"custom_tool_call_output", "call_id":"synthetic-call", "output":"synthetic result"}),
            json!({"type":"compaction", "encrypted_content":"synthetic opaque content"}),
        ] {
            let item: ResponseItem = serde_json::from_value(value).expect("synthetic tool output");
            assert!(completed(std::slice::from_ref(&item)).is_none());
            assert!(completed(&[assistant("Otherwise valid summary."), item]).is_none());
        }
    }

    #[test]
    fn accepts_legacy_missing_message_status_without_copying_provider_metadata() {
        let mut item = assistant("Legacy completed response summary.");
        let ResponseItem::Message {
            status,
            internal_chat_message_metadata_passthrough,
            ..
        } = &mut item
        else {
            unreachable!()
        };
        *status = None;
        *internal_chat_message_metadata_passthrough =
            Some(InternalChatMessageMetadataPassthrough {
                turn_id: Some("synthetic-foreign-turn".to_owned()),
                content_item_kinds: Some(vec![ContentItemKind("user.text".to_owned())]),
                content_item_metadata: Some(vec![ContentItemMetadata::user()]),
                ..Default::default()
            });
        let candidate = completed(&[item]).expect("completed legacy message");
        assert_eq!(
            candidate.item().item.turn_id(),
            Some("synthetic-original-turn")
        );
        assert!(!is_user_authorization_message(&candidate.item().item));
    }

    #[test]
    fn preserves_large_visible_summary_text_without_a_user_retention_budget() {
        let text = "synthetic 雪🧪\n".repeat(25_000);
        let output = [assistant(&text)];
        let candidate = completed(&output).expect("summary size is the installer's concern");
        assert_eq!(candidate.message(), format!("{SUMMARY_PREFIX}\n{text}"));
        assert_eq!(output, [assistant(&text)]);
    }
}
