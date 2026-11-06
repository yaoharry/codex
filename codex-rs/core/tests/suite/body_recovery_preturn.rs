//! Successful stock pre-turn compaction must leave an eligible owned body-recovery boundary.

use std::fs;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;
use codex_core::ForkSnapshot;
use codex_core::TurnInputRequest;
use codex_core::TurnInputSubmission;
use codex_core::compact::SUMMARIZATION_PROMPT;
use codex_core::compact::SUMMARY_PREFIX;
use codex_protocol::models::ContentItemProvenance;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_rollout::RolloutItem;
use codex_utils_string::approx_token_count;
use core_test_support::responses;
use core_test_support::wait_for_event_with_timeout;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use test_case::test_case;
use tokio::time::timeout;
use wiremock::Mock;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path_regex;

use super::body_rejection_recovery::ToolStarts;
use super::body_rejection_recovery::body_recovery_test_builder;
use super::body_rejection_recovery::completed_sse;
use super::body_rejection_recovery::finish_without_error;
use super::body_rejection_recovery::input;
use super::body_rejection_recovery::item_contains_text;
use super::body_rejection_recovery::text_input;

const AUTO_COMPACT_LIMIT: i64 = 80_000;
const SEED_TOKENS: i64 = 120_000;
const SEED_MARKER: &str = "PRETURN_COMPLETED_HISTORICAL_USER";
const SEED_CALL: &str = "preturn-completed-plan";
const SEED_DONE: &str = "PRETURN_SEED_COMPLETE";
const STOCK_MARKER: &str = "PRETURN_STOCK_COMPLETED_SUMMARY";
const STOCK_RESPONSE: &str = "preturn-stock-summary-response";
const CURRENT_MARKER: &str = "PRETURN_CURRENT_MULTIBLOCK_INPUT";
const CURRENT_TAIL: &str = "Current final block: naïve 東京 🦀 stays intact.";
const RECOVERY_SUMMARY: &str =
    "Completed historical work is summarized; keep the complete current request.";
const RECOVERY_RESPONSE: &str = "preturn-owned-summary-response";
const FINAL_RESPONSE: &str = "preturn-recovered-generation";
const FINAL: &str = "PRETURN_BODY_RECOVERY_OK";

fn stock_summary() -> String {
    format!(
        "{STOCK_MARKER}\n{}",
        "Completed synthetic historical work; preserve its established state.\n".repeat(512),
    )
}

#[derive(Default)]
struct RecordedRequests {
    lengths: Vec<usize>,
    summary_requests: Vec<bool>,
    limit: Option<usize>,
}

struct PreturnResponder {
    status: u16,
    recorded: Arc<Mutex<RecordedRequests>>,
}

impl Respond for PreturnResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        assert!(request.headers.get("content-encoding").is_none());
        let body: Value = serde_json::from_slice(&request.body).expect("uncompressed native JSON");
        let summary = body["tools"] == json!([]);
        let bytes = request.body.len();
        let mut recorded = self.recorded.lock().expect("recorded requests mutex");
        let index = recorded.lengths.len();
        recorded.lengths.push(bytes);
        recorded.summary_requests.push(summary);

        match index {
            0 => {
                assert!(!summary);
                assert!(
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, SEED_MARKER))
                );
                completed_sse(
                    "preturn-seed-tool-response",
                    responses::ev_function_call(SEED_CALL, "update_plan", &json!({
                        "plan": [{"step": "Completed synthetic historical work.", "status": "completed"}]
                    }).to_string()),
                )
            }
            1 => {
                assert!(!summary);
                assert_eq!(
                    input(&body)
                        .iter()
                        .filter(|item| {
                            item["type"] == "function_call_output" && item["call_id"] == SEED_CALL
                        })
                        .count(),
                    1
                );
                responses::sse_response(responses::sse(vec![
                    responses::ev_response_created("preturn-seed-final-response"),
                    responses::ev_assistant_message("preturn-seed-final", SEED_DONE),
                    responses::ev_completed_with_tokens("preturn-seed-final-response", SEED_TOKENS),
                ]))
            }
            2 => {
                let metadata: Value = body["client_metadata"]["x-codex-turn-metadata"]
                    .as_str()
                    .and_then(|value| serde_json::from_str(value).ok())
                    .unwrap_or(Value::Null);
                assert!(
                    summary,
                    "child first request: kind={:?} phase={:?} reason={:?} tools_array={} prompt={} current={} trigger={}",
                    metadata["request_kind"],
                    metadata["compaction"]["phase"],
                    metadata["compaction"]["reason"],
                    body["tools"].is_array(),
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, SUMMARIZATION_PROMPT)),
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, CURRENT_MARKER)),
                    input(&body)
                        .iter()
                        .any(|item| item["type"] == "compaction_trigger")
                );
                assert!(
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, SEED_MARKER))
                );
                assert!(
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, SUMMARIZATION_PROMPT))
                );
                assert!(
                    !input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, CURRENT_MARKER))
                );
                completed_sse(
                    STOCK_RESPONSE,
                    responses::ev_assistant_message("preturn-stock-summary", &stock_summary()),
                )
            }
            3 => {
                assert!(!summary);
                assert!(
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, CURRENT_MARKER))
                );
                assert!(
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, STOCK_MARKER))
                );
                assert!(
                    !input(&body).iter().any(|item| {
                        item["role"] == "assistant"
                            || item["type"] == "function_call"
                            || item["type"] == "function_call_output"
                    }),
                    "stock compaction leaves no synthetic assistant/tool endpoint"
                );
                let limit = bytes.checked_sub(1).expect("nonempty owned request");
                let mut irreducible = body.clone();
                irreducible["input"] = Value::Array(
                    input(&body)
                        .iter()
                        .filter(|item| {
                            !item_contains_text(item, SEED_MARKER)
                                && !item_contains_text(item, STOCK_MARKER)
                        })
                        .cloned()
                        .collect(),
                );
                assert!(
                    serde_json::to_vec(&irreducible)
                        .expect("serialize irreducible request")
                        .len()
                        + 4096
                        < limit,
                    "the full current input and mandatory fields can fit the cap",
                );
                recorded.limit = Some(limit);
                ResponseTemplate::new(self.status).set_body_json(json!({
                    "error_code": "BAD_REQUEST",
                    "message": format!(
                        "Server received a request which exceeds maximum allowed content length. RequestSize(bytes): {bytes}, Limit(bytes): {limit}"
                    )
                }))
            }
            4 => {
                assert!(
                    summary,
                    "body recovery must remain eligible after pre-turn compaction"
                );
                assert!(bytes <= recorded.limit.expect("measured body cap"));
                assert_eq!(body["tool_choice"], "none");
                assert_eq!(body["parallel_tool_calls"], false);
                assert!(
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, STOCK_MARKER))
                );
                assert!(
                    !input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, CURRENT_MARKER))
                );
                completed_sse(
                    RECOVERY_RESPONSE,
                    responses::ev_assistant_message("preturn-owned-summary", RECOVERY_SUMMARY),
                )
            }
            5 => {
                assert!(!summary);
                assert!(bytes <= recorded.limit.expect("measured body cap"));
                assert!(
                    bytes < recorded.lengths[3],
                    "the retry makes actual byte progress"
                );
                assert!(
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, CURRENT_MARKER))
                );
                completed_sse(
                    FINAL_RESPONSE,
                    responses::ev_assistant_message("preturn-final", FINAL),
                )
            }
            _ => completed_sse(
                "preturn-unexpected-request",
                responses::ev_assistant_message(
                    "preturn-unexpected",
                    "UNEXPECTED_EXTRA_PRETURN_REQUEST",
                ),
            ),
        }
    }
}

#[test_case(400; "http_400")]
#[test_case(413; "http_413")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn successful_preturn_compaction_keeps_body_recovery_eligible(status: u16) -> Result<()> {
    let server = responses::start_mock_server().await;
    let recorded = Arc::new(Mutex::new(RecordedRequests::default()));
    Mock::given(method("POST"))
        .and(path_regex(".*/responses$"))
        .respond_with(PreturnResponder {
            status,
            recorded: Arc::clone(&recorded),
        })
        .mount(&server)
        .await;
    let starts = Arc::new(ToolStarts::default());
    let test = body_recovery_test_builder(&server, Arc::clone(&starts))
        .build_with_auto_env(&server)
        .await?;
    let seed_text = format!(
        "{SEED_MARKER}\n{}",
        (0..8192)
            .map(|index| format!("Historical row {index:05}: completed synthetic work.\n"))
            .collect::<String>(),
    );
    assert!(approx_token_count(&seed_text) > AUTO_COMPACT_LIMIT as usize + 4096);
    let seed = test
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![text_input(
            seed_text.clone(),
        )]))
        .await?;
    let TurnInputSubmission::Started {
        turn_id: seed_turn, ..
    } = seed
    else {
        panic!("seed must start a user turn: {seed:?}");
    };
    assert_eq!(
        finish_without_error(&test.codex, &seed_turn).await,
        SEED_DONE
    );
    assert_eq!(starts.snapshot(), vec![SEED_CALL.to_owned()]);
    test.codex.flush_rollout().await?;
    let parent_path = test.codex.rollout_path().expect("parent canonical rollout");
    let parent_original = fs::read(&parent_path)?;
    let (mut options, settings) = test
        .thread_manager
        .fork_options_from_parent(test.session_configured.thread_id)
        .await?;
    options.config.model_auto_compact_token_limit = Some(AUTO_COMPACT_LIMIT);
    options.config.model_post_turn_compact_threshold_percent = 0;
    let child = test
        .thread_manager
        .fork_legacy_thread(ForkSnapshot::Interrupted, options, parent_path.clone())
        .await?;
    child.thread.restore_thread_settings(settings).await?;
    let restored_usage = child
        .thread
        .token_usage_info()
        .await
        .expect("restored provider usage")
        .last_token_usage
        .total_tokens;
    assert_eq!(restored_usage, SEED_TOKENS);
    assert!(restored_usage > AUTO_COMPACT_LIMIT);
    child.thread.flush_rollout().await?;
    let child_path = child
        .thread
        .rollout_path()
        .expect("child canonical rollout");
    let child_original = fs::read(&child_path)?;
    assert!(String::from_utf8_lossy(&child_original).contains(SEED_MARKER));

    let current_text = format!(
        "{CURRENT_MARKER}\n{}",
        "Current café 東京 🦀 input must remain exact.\n".repeat(2048),
    );
    assert!(approx_token_count(&current_text) > 20_000);
    assert!(approx_token_count(&current_text) < 30_000);
    assert!(approx_token_count(&stock_summary()) < 10_000);
    let submission = child
        .thread
        .start_or_steer_turn(TurnInputRequest::user_input(vec![
            text_input(current_text.clone()),
            text_input(String::new()),
            text_input(CURRENT_TAIL.to_owned()),
        ]))
        .await?;
    let TurnInputSubmission::Started {
        turn_id: current_turn,
        ..
    } = submission
    else {
        panic!("child input must start a user turn: {submission:?}");
    };
    let mut turn_starts = 0;
    let mut completed_responses = Vec::new();
    let completed = timeout(
        Duration::from_secs(30),
        wait_for_event_with_timeout(
            &child.thread,
            |event| {
                match event {
                    EventMsg::TurnStarted(started) => {
                        assert_eq!(started.turn_id, current_turn);
                        turn_starts += 1;
                    }
                    EventMsg::RawResponseCompleted(response) => {
                        completed_responses.push(response.response_id.clone());
                    }
                    EventMsg::Error(error) => panic!("pre-turn body recovery failed: {error:?}"),
                    EventMsg::TurnAborted(_) => panic!("recovery must finish the admitted turn"),
                    EventMsg::StreamError(_) => panic!("no unchanged transport replay"),
                    _ => {}
                }
                matches!(event, EventMsg::TurnComplete(_))
            },
            Duration::from_secs(10),
        ),
    )
    .await
    .expect("pre-turn recovery must complete within the fixed deadline");
    let EventMsg::TurnComplete(completed) = completed else {
        unreachable!()
    };
    assert_eq!(completed.turn_id, current_turn);
    assert!(completed.error.is_none());
    assert_eq!(completed.last_agent_message.as_deref(), Some(FINAL));
    assert_eq!(turn_starts, 1);
    assert_eq!(
        completed_responses,
        vec![STOCK_RESPONSE, RECOVERY_RESPONSE, FINAL_RESPONSE],
    );
    assert_eq!(
        starts.snapshot(),
        vec![SEED_CALL.to_owned()],
        "old tools never run again"
    );

    let requests = responses::received_responses_requests(&server).await;
    assert_eq!(requests.len(), 6);
    let (lengths, summary_requests, limit) = {
        let recorded = recorded.lock().expect("recorded requests mutex");
        (
            recorded.lengths.clone(),
            recorded.summary_requests.clone(),
            recorded.limit.expect("measured body cap"),
        )
    };
    assert_eq!(
        summary_requests,
        vec![false, false, true, false, true, false]
    );
    for (request, bytes) in requests.iter().zip(&lengths) {
        assert_eq!(request.body_bytes().len(), *bytes);
    }
    assert_eq!(lengths[3], limit + 1);
    assert!(lengths[4] <= limit && lengths[5] <= limit && lengths[5] < lengths[3]);
    for (index, phase, reason) in [
        (2, "pre_turn", "context_limit"),
        (4, "mid_turn", "request_body_limit"),
    ] {
        let metadata: Value = serde_json::from_str(
            &requests[index]
                .header("x-codex-turn-metadata")
                .expect("compactor metadata"),
        )?;
        assert_eq!(metadata["request_kind"], "compaction");
        assert_eq!(metadata["compaction"]["phase"], phase);
        assert_eq!(metadata["compaction"]["reason"], reason);
    }

    let before = requests[3].body_json();
    let stock = requests[2].body_json();
    let recovery = requests[4].body_json();
    let retry = requests[5].body_json();
    let raw_inputs = [3, 4, 5].map(|index| {
        let fields: std::collections::BTreeMap<String, Box<serde_json::value::RawValue>> =
            serde_json::from_slice(&requests[index].body_bytes()).expect("owned JSON fields");
        serde_json::from_str::<Vec<Box<serde_json::value::RawValue>>>(fields["input"].get())
            .expect("owned raw input")
    });
    let mut mandatory_context = 0;
    for (value, raw) in input(&before).iter().zip(&raw_inputs[0]) {
        if value["role"] != "developer" && value["role"] != "system" {
            continue;
        }
        mandatory_context += 1;
        for target in &raw_inputs[1..] {
            assert_eq!(
                target
                    .iter()
                    .filter(|item| item.get().as_bytes() == raw.get().as_bytes())
                    .count(),
                1,
                "mandatory context remains byte-identical in summary and retry"
            );
        }
    }
    assert!(mandatory_context >= 2);
    for compactor in [&stock, &recovery] {
        assert!(
            !input(compactor)
                .iter()
                .any(|item| item_contains_text(item, CURRENT_MARKER))
        );
    }
    let current = input(&before)
        .iter()
        .find(|item| item_contains_text(item, CURRENT_MARKER))
        .expect("original current input");
    assert_eq!(
        input(&retry)
            .iter()
            .filter(|item| item_contains_text(item, CURRENT_MARKER))
            .collect::<Vec<_>>(),
        vec![current],
    );
    let current_blocks = current["content"]
        .as_array()
        .expect("native JSON array")
        .iter()
        .filter_map(|part| part["text"].as_str())
        .collect::<Vec<_>>();
    assert!(current_blocks.contains(&current_text.as_str()));
    assert!(current_blocks.contains(&""));
    assert!(current_blocks.contains(&CURRENT_TAIL));
    for item in input(&before)
        .iter()
        .filter(|item| item["role"] == "developer")
    {
        assert!(
            input(&retry).contains(item),
            "mandatory developer input changed"
        );
    }
    for field in [
        "model",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "text",
    ] {
        assert_eq!(
            before.get(field),
            retry.get(field),
            "generation field {field}"
        );
    }
    assert!(input(&retry).iter().any(|item| {
        item_contains_text(item, &format!("{SUMMARY_PREFIX}\n{RECOVERY_SUMMARY}"))
    }));
    assert!(
        !input(&retry)
            .iter()
            .any(|item| item_contains_text(item, STOCK_MARKER))
    );
    let retained_seed = input(&before)
        .iter()
        .find(|item| item_contains_text(item, SEED_MARKER))
        .expect("stock-retained mixed-source input");
    assert_eq!(
        input(&retry)
            .iter()
            .filter(|item| item_contains_text(item, SEED_MARKER))
            .collect::<Vec<_>>(),
        vec![retained_seed],
        "mixed-source stock input remains exact"
    );
    assert!(
        approx_token_count(&serde_json::to_string(&retry)?) + 1024 < AUTO_COMPACT_LIMIT as usize,
        "the irreducible current request fits without another token compaction",
    );

    let history = child.thread.conversation_history_snapshot().await;
    let history = serde_json::to_value(history.items().collect::<Vec<_>>())?;
    let retained_current = history
        .as_array()
        .expect("native JSON array")
        .iter()
        .filter(|item| item_contains_text(item, CURRENT_MARKER))
        .collect::<Vec<_>>();
    assert_eq!(retained_current.len(), 1);
    assert_eq!(retained_current[0]["content"], current["content"]);
    child.thread.flush_rollout().await?;
    let child_rollout = fs::read(child_path)?;
    assert!(
        child_rollout.starts_with(&child_original),
        "child original canonical bytes changed"
    );
    assert_eq!(
        fs::read(parent_path)?,
        parent_original,
        "fork source bytes changed"
    );
    let mut checkpoints = Vec::new();
    let mut raw_current = Vec::new();
    for line in std::str::from_utf8(&child_rollout)?.lines() {
        match codex_rollout::parse_rollout_line(line)?.item {
            RolloutItem::Compacted(checkpoint) => checkpoints.push(checkpoint),
            RolloutItem::ResponseItem(envelope) => {
                let item = serde_json::to_value(&envelope.item)?;
                if item_contains_text(&item, CURRENT_MARKER) {
                    raw_current.push(item["content"].clone());
                }
            }
            _ => {}
        }
    }
    assert_eq!(checkpoints.len(), 2);
    assert_eq!(
        checkpoints[0].compaction_response_id.as_deref(),
        Some(STOCK_RESPONSE)
    );
    assert_eq!(
        checkpoints[1].compaction_response_id.as_deref(),
        Some(RECOVERY_RESPONSE)
    );
    let stock_history = checkpoints[0]
        .replacement_history
        .as_ref()
        .expect("stock replacement history");
    let recovered_history = checkpoints[1]
        .replacement_history
        .as_ref()
        .expect("recovery replacement history");
    let retained_seed = stock_history
        .iter()
        .find(|item| {
            serde_json::to_value(&item.item)
                .is_ok_and(|value| item_contains_text(&value, SEED_MARKER))
        })
        .expect("canonical mixed-source input");
    let (kinds, sources) = match &retained_seed.item {
        ResponseItem::Message {
            internal_chat_message_metadata_passthrough: Some(metadata),
            ..
        } => (
            metadata.content_item_kinds.clone(),
            metadata.content_item_metadata.as_ref().map(|parts| {
                parts
                    .iter()
                    .map(|part| part.provenance.clone())
                    .collect::<Vec<_>>()
            }),
        ),
        _ => (None, None),
    };
    let kinds = kinds.expect("stock input kinds");
    let sources = sources.expect("stock input sources");
    assert_eq!(kinds.len(), sources.len());
    assert!(
        kinds
            .iter()
            .zip(&sources)
            .any(|(kind, source)| kind.as_str() == "unknown"
                && matches!(source, ContentItemProvenance::Harness { hook: false })),
        "stock truncation adds a host notice, so this mixed-source input must remain exact"
    );
    assert_eq!(
        recovered_history
            .iter()
            .filter(|item| item.item == retained_seed.item)
            .count(),
        1,
        "mixed-source input remains typed and exact across recovery"
    );
    let mut environment_context = 0;
    for envelope in stock_history {
        let ResponseItem::Message {
            role,
            content,
            internal_chat_message_metadata_passthrough: Some(metadata),
            ..
        } = &envelope.item
        else {
            continue;
        };
        if !metadata.content_item_kinds.as_ref().is_some_and(|kinds| {
            kinds
                .iter()
                .any(|kind| kind.as_str() == "environments.environment_context")
        }) {
            continue;
        }
        environment_context += 1;
        assert_eq!(role, "user");
        let kinds = metadata.content_item_kinds.as_ref().expect("host kinds");
        let sources = metadata
            .content_item_metadata
            .as_ref()
            .expect("host sources");
        assert_eq!(kinds.len(), content.len());
        assert_eq!(sources.len(), content.len());
        assert!(sources.iter().all(|part| matches!(
            part.provenance,
            ContentItemProvenance::Harness { hook: false }
        )));
        assert_eq!(
            recovered_history
                .iter()
                .filter(|item| item.item == envelope.item)
                .count(),
            1,
            "typed mandatory context remains exact in the next checkpoint"
        );
        // Custom providers remove internal attribution from wire items, while canonical items retain it.
        let mut wire = envelope.item.clone();
        wire.clear_internal_chat_message_metadata_passthrough();
        let wire = serde_json::to_value(wire)?;
        let positions = input(&before)
            .iter()
            .enumerate()
            .filter(|(_, item)| **item == wire)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(
            positions.len(),
            1,
            "canonical host context projects to one original wire item"
        );
        let raw = &raw_inputs[0][positions[0]];
        for target in &raw_inputs[1..] {
            assert_eq!(
                target
                    .iter()
                    .filter(|item| item.get().as_bytes() == raw.get().as_bytes())
                    .count(),
                1,
                "environment context stays byte-identical in summary and retry"
            );
        }
    }
    assert_eq!(environment_context, 1);
    assert!(
        !stock_history.iter().any(|envelope| {
            matches!(&envelope.item, ResponseItem::Message { role, .. } if role == "assistant")
                || matches!(
                    &envelope.item,
                    ResponseItem::FunctionCall { .. } | ResponseItem::FunctionCallOutput { .. }
                )
        }),
        "stock history has no invented closed assistant/tool endpoint"
    );
    let host_summary = stock_history
        .iter()
        .find(|envelope| {
            serde_json::to_value(&envelope.item)
                .is_ok_and(|item| item_contains_text(&item, STOCK_MARKER))
        })
        .expect("completed stock summary");
    let ResponseItem::Message {
        role,
        internal_chat_message_metadata_passthrough: Some(metadata),
        ..
    } = &host_summary.item
    else {
        panic!("stock summary must retain typed host provenance")
    };
    assert_eq!(role, "user");
    let kinds = metadata
        .content_item_kinds
        .as_ref()
        .expect("host content kinds");
    assert_eq!(kinds.len(), 1);
    assert_eq!(kinds[0].as_str(), "compaction.summary");
    let parts = metadata
        .content_item_metadata
        .as_ref()
        .expect("host provenance");
    assert_eq!(parts.len(), 1);
    assert!(matches!(
        parts[0].provenance,
        ContentItemProvenance::Harness { hook: false }
    ));
    assert_eq!(raw_current, vec![current["content"].clone()]);
    Ok(())
}
