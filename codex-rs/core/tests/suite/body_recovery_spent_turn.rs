//! A completed model step cannot refill recovery already spent by the same user turn.

use std::fs;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_core::TurnInputSubmission;
use codex_core::compact::SUMMARIZATION_PROMPT;
use codex_core::compact::SUMMARY_PREFIX;
use codex_protocol::protocol::EventMsg;
use codex_rollout::RolloutItem;
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

const SEED_MARKER: &str = "SPENT_TURN_COMPLETED_HISTORY";
const SEED_DONE: &str = "SPENT_TURN_SEED_DONE";
const SEED_CALL: &str = "spent-turn-seed-plan";
const CURRENT_MARKER: &str = "SPENT_TURN_CURRENT_INPUT";
const CURRENT_TAIL: &str = "Current final block: naïve 東京 🦀 remains exact.";
const SUMMARY: &str = "The synthetic historical work is complete; keep the current request intact.";
const SUMMARY_RESPONSE: &str = "spent-turn-local-summary";
const RECOVERED_RESPONSE: &str = "spent-turn-recovered-generation";
const NEW_CALL: &str = "spent-turn-new-plan";

#[derive(Default)]
struct RecordedRequests {
    lengths: Vec<usize>,
    summaries: Vec<bool>,
    first_limit: Option<usize>,
    second_limit: Option<usize>,
}

struct SpentTurnResponder {
    status: u16,
    recorded: Arc<Mutex<RecordedRequests>>,
}

impl Respond for SpentTurnResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        assert!(request.headers.get("content-encoding").is_none());
        let body: Value = serde_json::from_slice(&request.body).expect("native JSON request");
        let bytes = request.body.len();
        let summary = body["tools"] == json!([]);
        let mut recorded = self.recorded.lock().expect("recorded requests mutex");
        let index = recorded.lengths.len();
        recorded.lengths.push(bytes);
        recorded.summaries.push(summary);

        match index {
            0 => {
                assert!(!summary);
                assert!(
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, SEED_MARKER))
                );
                completed_sse(
                    "spent-turn-seed-tool-response",
                    responses::ev_function_call(SEED_CALL, "update_plan", &plan_arguments("seed")),
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
                    1,
                );
                completed_sse(
                    "spent-turn-seed-final-response",
                    responses::ev_assistant_message("spent-turn-seed-final", SEED_DONE),
                )
            }
            2 => {
                assert!(!summary);
                assert!(
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, CURRENT_MARKER))
                );
                let limit = bytes.checked_sub(1).expect("nonempty generation body");
                recorded.first_limit = Some(limit);
                body_rejection(self.status, bytes, limit)
            }
            3 => {
                assert!(summary, "the first cap must trigger local summarization");
                assert!(
                    bytes <= recorded.first_limit.expect("first measured body cap"),
                    "summary fits the measured cap"
                );
                assert_eq!(body["tool_choice"], "none");
                assert_eq!(body["parallel_tool_calls"], false);
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
                    SUMMARY_RESPONSE,
                    responses::ev_assistant_message("spent-turn-summary-message", SUMMARY),
                )
            }
            4 => {
                assert!(!summary);
                assert!(
                    bytes <= recorded.first_limit.expect("first measured body cap"),
                    "recovered generation fits"
                );
                assert!(
                    input(&body)
                        .iter()
                        .any(|item| item_contains_text(item, CURRENT_MARKER))
                );
                completed_sse(
                    RECOVERED_RESPONSE,
                    responses::ev_function_call(NEW_CALL, "update_plan", &plan_arguments("new")),
                )
            }
            5 => {
                assert!(!summary, "this must be the ordinary tool-result followup");
                for kind in ["function_call", "function_call_output"] {
                    assert_eq!(
                        input(&body)
                            .iter()
                            .filter(|item| { item["type"] == kind && item["call_id"] == NEW_CALL })
                            .count(),
                        1,
                    );
                }
                let limit = bytes.checked_sub(1).expect("nonempty followup body");
                assert!(
                    limit < recorded.first_limit.expect("first measured body cap"),
                    "independent authoritative cap"
                );
                recorded.second_limit = Some(limit);
                body_rejection(self.status, bytes, limit)
            }
            _ => completed_sse(
                "spent-turn-unexpected-extra-response",
                responses::ev_assistant_message(
                    "spent-turn-unexpected-extra-message",
                    "UNEXPECTED_EXTRA_RECOVERY_REQUEST",
                ),
            ),
        }
    }
}

fn plan_arguments(label: &str) -> String {
    json!({
        "plan": [{
            "step": format!("Completed synthetic {label} work."),
            "status": "completed"
        }]
    })
    .to_string()
}

fn body_rejection(status: u16, bytes: usize, limit: usize) -> ResponseTemplate {
    assert!(bytes > limit && limit > 0);
    ResponseTemplate::new(status).set_body_json(json!({
        "error_code": "BAD_REQUEST",
        "message": format!(
            "Server received a request which exceeds maximum allowed content length. RequestSize(bytes): {bytes}, Limit(bytes): {limit}"
        )
    }))
}

#[test_case(400; "http_400")]
#[test_case(413; "http_413")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completed_tool_step_does_not_refill_same_turn_body_recovery(status: u16) -> Result<()> {
    let server = responses::start_mock_server().await;
    let recorded = Arc::new(Mutex::new(RecordedRequests::default()));
    Mock::given(method("POST"))
        .and(path_regex(".*/responses$"))
        .respond_with(SpentTurnResponder {
            status,
            recorded: Arc::clone(&recorded),
        })
        .mount(&server)
        .await;
    let starts = Arc::new(ToolStarts::default());
    let test = body_recovery_test_builder(&server, Arc::clone(&starts))
        .build_with_auto_env(&server)
        .await?;

    let seed = test
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![text_input(format!(
            "{SEED_MARKER}\n{}",
            "completed historical evidence ".repeat(4096)
        ))]))
        .await?;
    let TurnInputSubmission::Started {
        turn_id: seed_turn, ..
    } = seed
    else {
        panic!("seed must start one admitted turn: {seed:?}");
    };
    assert_eq!(
        finish_without_error(&test.codex, &seed_turn).await,
        SEED_DONE
    );
    assert_eq!(starts.snapshot(), vec![SEED_CALL.to_owned()]);
    test.codex.flush_rollout().await?;
    let rollout_path = test.codex.rollout_path().expect("canonical rollout");
    let original_rollout = fs::read(&rollout_path)?;

    let current_text = format!(
        "{CURRENT_MARKER}\n{}",
        "Current café 東京 🦀 remains exact.\n".repeat(1024)
    );
    assert!(current_text.len() > SUMMARIZATION_PROMPT.len() + 8192);
    let submission = test
        .codex
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
        panic!("current input must start one admitted turn: {submission:?}");
    };
    assert_ne!(current_turn, seed_turn);
    let mut turn_starts = 0;
    let mut errors = Vec::new();
    let mut completed_responses = Vec::new();
    let event = timeout(
        Duration::from_secs(30),
        wait_for_event_with_timeout(
            &test.codex,
            |event| {
                match event {
                    EventMsg::TurnStarted(started) => {
                        assert_eq!(started.turn_id, current_turn, "no replacement user turn");
                        turn_starts += 1;
                    }
                    EventMsg::Error(error) => errors.push(error.message.clone()),
                    EventMsg::RawResponseCompleted(completed) => {
                        completed_responses.push(completed.response_id.clone());
                    }
                    EventMsg::TurnAborted(_) => {
                        panic!("second cap must fail the original turn, not abort")
                    }
                    EventMsg::StreamError(_) => {
                        panic!("spent recovery must not replay the request")
                    }
                    _ => {}
                }
                matches!(event, EventMsg::TurnComplete(_))
            },
            Duration::from_secs(10),
        ),
    )
    .await
    .expect("same-turn recovery must settle within the fixed deadline");
    let EventMsg::TurnComplete(completed) = event else {
        unreachable!()
    };
    assert_eq!(completed.turn_id, current_turn);
    assert_eq!(turn_starts, 1);
    assert_eq!(errors.len(), 1, "only the second cap is a terminal failure");
    assert!(
        completed.error.is_some(),
        "the second cap must remain visible"
    );
    assert_eq!(
        completed_responses,
        vec![SUMMARY_RESPONSE, RECOVERED_RESPONSE]
    );
    assert_eq!(
        starts.snapshot(),
        vec![SEED_CALL.to_owned(), NEW_CALL.to_owned()]
    );

    let requests = responses::received_responses_requests(&server).await;
    assert_eq!(
        requests.len(),
        6,
        "two seed requests, one summary, three current generations"
    );
    let (lengths, summaries, first_limit, second_limit) = {
        let recorded = recorded.lock().expect("recorded requests mutex");
        (
            recorded.lengths.clone(),
            recorded.summaries.clone(),
            recorded.first_limit.expect("first owned cap"),
            recorded.second_limit.expect("second owned cap"),
        )
    };
    assert_eq!(summaries, vec![false, false, false, true, false, false]);
    for (request, bytes) in requests.iter().zip(&lengths) {
        assert_eq!(request.body_bytes().len(), *bytes);
    }
    assert_eq!(first_limit + 1, lengths[2]);
    assert_eq!(second_limit + 1, lengths[5]);
    assert!(lengths[3] <= first_limit && lengths[4] <= first_limit);
    let second_cap = format!(
        "RequestSize(bytes): {}, Limit(bytes): {second_limit}",
        lengths[5]
    );
    assert!(errors[0].contains(&second_cap));
    assert!(
        completed
            .error
            .as_ref()
            .expect("terminal body rejection")
            .message
            .contains(&second_cap)
    );

    let original = requests[2].body_json();
    let summary = requests[3].body_json();
    let retry = requests[4].body_json();
    let followup = requests[5].body_json();
    let current = input(&original)
        .iter()
        .find(|item| item_contains_text(item, CURRENT_MARKER))
        .expect("submitted current input");
    for generation in [&retry, &followup] {
        assert_eq!(
            input(generation)
                .iter()
                .filter(|item| item_contains_text(item, CURRENT_MARKER))
                .collect::<Vec<_>>(),
            vec![current],
            "every generation keeps the full current input",
        );
    }
    let blocks = current["content"]
        .as_array()
        .expect("native JSON array")
        .iter()
        .filter_map(|block| block["text"].as_str())
        .collect::<Vec<_>>();
    assert!(blocks.contains(&current_text.as_str()));
    assert!(blocks.contains(&""));
    assert!(blocks.contains(&CURRENT_TAIL));
    assert!(
        input(&followup)
            .iter()
            .any(|item| item_contains_text(item, &format!("{SUMMARY_PREFIX}\n{SUMMARY}")))
    );

    // Another summary could fit: refusal must come from the spent turn budget.
    let mut another_summary = summary.clone();
    let mut another_input = input(&followup)
        .iter()
        .filter(|item| !item_contains_text(item, CURRENT_MARKER) && item["call_id"] != NEW_CALL)
        .cloned()
        .collect::<Vec<_>>();
    another_input.push(input(&summary).last().expect("summary prompt").clone());
    another_summary["input"] = Value::Array(another_input);
    assert!(serde_json::to_vec(&another_summary)?.len() + 1024 < second_limit);

    let history = test.codex.conversation_history_snapshot().await;
    let history = serde_json::to_value(history.items().collect::<Vec<_>>())?;
    let history = history.as_array().expect("native JSON array");
    let retained_current = history
        .iter()
        .filter(|item| item_contains_text(item, CURRENT_MARKER))
        .collect::<Vec<_>>();
    assert_eq!(retained_current.len(), 1);
    assert_eq!(retained_current[0]["content"], current["content"]);
    for kind in ["function_call", "function_call_output"] {
        assert_eq!(
            history
                .iter()
                .filter(|item| item["type"] == kind && item["call_id"] == NEW_CALL)
                .count(),
            1
        );
    }

    test.codex.flush_rollout().await?;
    let rollout = fs::read(rollout_path)?;
    assert!(
        rollout.starts_with(&original_rollout),
        "original canonical records changed"
    );
    let mut checkpoints = 0;
    let mut raw_current = Vec::new();
    let mut raw_new_calls = 0;
    let mut raw_new_outputs = 0;
    for line in std::str::from_utf8(&rollout)?.lines() {
        match codex_rollout::parse_rollout_line(line)?.item {
            RolloutItem::Compacted(checkpoint) => {
                checkpoints += 1;
                assert_eq!(
                    checkpoint.compaction_response_id.as_deref(),
                    Some(SUMMARY_RESPONSE)
                );
            }
            RolloutItem::ResponseItem(envelope) => {
                let item = serde_json::to_value(&envelope.item)?;
                if item_contains_text(&item, CURRENT_MARKER) {
                    raw_current.push(item["content"].clone());
                }
                if item["call_id"] == NEW_CALL {
                    raw_new_calls += usize::from(item["type"] == "function_call");
                    raw_new_outputs += usize::from(item["type"] == "function_call_output");
                }
            }
            _ => {}
        }
    }
    assert_eq!(
        checkpoints, 1,
        "the second cap must not install another recovery"
    );
    assert_eq!(raw_current, vec![current["content"].clone()]);
    assert_eq!(raw_new_calls, 1);
    assert_eq!(raw_new_outputs, 1);
    Ok(())
}
