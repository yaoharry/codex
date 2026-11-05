//! Normal-turn recovery from a synthetic, measured HTTP request-body limit.

use std::fs;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;
use codex_core::CodexThread;
use codex_core::TurnInputRequest;
use codex_core::TurnInputSubmission;
use codex_core::compact::SUMMARIZATION_PROMPT;
use codex_core::compact::SUMMARY_PREFIX;
use codex_core::config::Config;
use codex_extension_api::ExtensionRegistryBuilder;
use codex_extension_api::ToolLifecycleContributor;
use codex_extension_api::ToolLifecycleFuture;
use codex_extension_api::ToolStartInput;
use codex_features::Feature;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::models::ImageReference;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use codex_utils_string::approx_token_count;
use core_test_support::responses;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use test_case::test_case;
use wiremock::Mock;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path_regex;

const HISTORICAL_TURNS: usize = 16;
const CURRENT_MARKER: &str = "CURRENT_BODY_RECOVERY_MULTIBLOCK";
const BASE_INSTRUCTIONS: &str = "Preserve every current input block and obey the supplied tools.";
const SUMMARY: &str = "Synthetic historical work is complete; preserve the remaining work.";
const FINAL: &str = "BODY_RECOVERY_NORMAL_TURN_OK";
const IMAGE_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

#[derive(Default)]
pub(super) struct ToolStarts(Mutex<Vec<String>>);

impl ToolStarts {
    pub(super) fn snapshot(&self) -> Vec<String> {
        self.0.lock().expect("recorded requests mutex").clone()
    }
}

impl ToolLifecycleContributor for ToolStarts {
    fn on_tool_start<'a>(&'a self, input: ToolStartInput<'a>) -> ToolLifecycleFuture<'a> {
        Box::pin(async move {
            self.0
                .lock()
                .expect("recorded requests mutex")
                .push(input.call_id.to_owned());
        })
    }
}

#[derive(Debug)]
struct CappedAttempt {
    bytes: usize,
    summary: bool,
    rejected: bool,
}

#[derive(Default)]
struct BodyCapState {
    armed: bool,
    seed_requests: usize,
    limit: Option<usize>,
    attempts: Vec<CappedAttempt>,
}

struct OwnedBodyCap {
    status: u16,
    tighten_summary_cap: bool,
    state: Arc<Mutex<BodyCapState>>,
}

impl Respond for OwnedBodyCap {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        assert!(request.headers.get("content-encoding").is_none());
        let body: Value = serde_json::from_slice(&request.body).expect("uncompressed request JSON");
        let mut state = self.state.lock().expect("recorded requests mutex");
        if !state.armed {
            let index = state.seed_requests / 2;
            assert!(index < HISTORICAL_TURNS, "unexpected seed request");
            assert!(contains_text(&body, &historical_marker(index)));
            let response_id = format!("seed-response-{}", state.seed_requests);
            let output = if state.seed_requests.is_multiple_of(2) {
                responses::ev_function_call(&call_id(index), "update_plan", &plan_arguments(index))
            } else {
                responses::ev_assistant_message(
                    &format!("seed-message-{index}"),
                    &format!("HISTORICAL_DONE_{index:02}"),
                )
            };
            state.seed_requests += 1;
            return completed_sse(&response_id, output);
        }

        let summary = body["tools"] == json!([]);
        if summary {
            assert_eq!(body["tool_choice"], "none");
            assert_eq!(body["parallel_tool_calls"], false);
            assert!(contains_text(&body, SUMMARIZATION_PROMPT));
            assert!(!contains_text(&body, CURRENT_MARKER));
        } else {
            assert!(contains_text(&body, CURRENT_MARKER));
        }
        let bytes = request.body.len();
        let limit = *state.limit.get_or_insert_with(|| {
            assert!(!summary, "the ordinary turn must encounter the body limit");
            bytes * 3 / 4
        });
        let limit = if summary
            && self.tighten_summary_cap
            && !state.attempts.iter().any(|attempt| attempt.summary)
        {
            assert!(
                bytes <= limit,
                "the first selected prefix fits the original cap"
            );
            let reduced = bytes - 1;
            state.limit = Some(reduced);
            reduced
        } else {
            limit
        };
        assert!(state.attempts.len() < 2 * HISTORICAL_TURNS + 4);
        let rejected = bytes > limit;
        state.attempts.push(CappedAttempt {
            bytes,
            summary,
            rejected,
        });
        if rejected {
            return ResponseTemplate::new(self.status).set_body_json(json!({
                "error_code": "BAD_REQUEST",
                "message": format!(
                    "Server received a request which exceeds maximum allowed content length. RequestSize(bytes): {bytes}, Limit(bytes): {limit}"
                ),
            }));
        }
        if summary {
            completed_sse(
                "owned-local-summary",
                responses::ev_assistant_message("owned-summary-message", SUMMARY),
            )
        } else {
            assert!(
                state
                    .attempts
                    .iter()
                    .any(|attempt| attempt.summary && !attempt.rejected)
            );
            completed_sse(
                "owned-recovered-generation",
                responses::ev_assistant_message("owned-final-message", FINAL),
            )
        }
    }
}

pub(super) fn completed_sse(response_id: &str, output: Value) -> ResponseTemplate {
    responses::sse_response(responses::sse(vec![
        responses::ev_response_created(response_id),
        output,
        responses::ev_completed(response_id),
    ]))
}

fn historical_marker(index: usize) -> String {
    format!("HISTORICAL_USER_{index:02}")
}

fn call_id(index: usize) -> String {
    format!("historical-plan-{index:02}")
}

fn plan_arguments(index: usize) -> String {
    let arguments = json!({
        "plan": [{
            "step": format!("Completed synthetic group {index}: {}", "valid step ".repeat(700)),
            "status": "completed",
        }],
    })
    .to_string();
    assert!(arguments.len() < 8 * 1024);
    arguments
}

pub(super) fn text_input(text: String) -> UserInput {
    UserInput::Text {
        text,
        text_elements: Vec::new(),
    }
}

pub(super) fn image_input() -> UserInput {
    UserInput::Image {
        image: ImageReference::Inline {
            image_url: IMAGE_URL.to_owned(),
        },
        detail: None,
    }
}

pub(super) fn item_contains_text(item: &Value, needle: &str) -> bool {
    item["content"].as_array().is_some_and(|parts| {
        parts.iter().any(|part| {
            part["text"]
                .as_str()
                .is_some_and(|text| text.contains(needle))
        })
    })
}

fn contains_text(body: &Value, needle: &str) -> bool {
    input(body)
        .iter()
        .any(|item| item_contains_text(item, needle))
}

pub(super) fn input(body: &Value) -> &[Value] {
    body["input"].as_array().expect("request input")
}

fn is_historical(item: &Value) -> bool {
    item["call_id"]
        .as_str()
        .is_some_and(|id| id.starts_with("historical-plan-"))
        || item_contains_text(item, "HISTORICAL_USER_")
        || item_contains_text(item, "HISTORICAL_DONE_")
}

pub(super) async fn finish_without_error(codex: &CodexThread, admitted_turn_id: &str) -> String {
    let event = wait_for_event(codex, |event| {
        matches!(
            event,
            EventMsg::Error(_) | EventMsg::TurnComplete(_) | EventMsg::TurnAborted(_)
        )
    })
    .await;
    match event {
        EventMsg::TurnComplete(completed) => {
            assert_eq!(
                completed.turn_id, admitted_turn_id,
                "recovery must complete the admitted original turn"
            );
            assert!(completed.error.is_none(), "terminal error: {completed:?}");
            completed.last_agent_message.expect("assistant completion")
        }
        event => panic!("normal turn emitted a terminal error: {event:?}"),
    }
}

pub(super) fn body_recovery_test_builder(
    server: &wiremock::MockServer,
    starts: Arc<ToolStarts>,
) -> core_test_support::test_codex::TestCodexBuilder {
    let mut extensions = ExtensionRegistryBuilder::<Config>::new();
    extensions.tool_lifecycle_contributor(starts);
    let mut provider = built_in_model_providers(None)["openai"].clone();
    provider.name = "Synthetic body-cap provider".to_owned();
    provider.base_url = Some(format!("{}/v1", server.uri()));
    provider.supports_websockets = false;
    provider.request_max_retries = Some(0);
    provider.stream_max_retries = Some(0);
    test_codex()
        .with_extensions(Arc::new(extensions.build()))
        .with_config(move |config| {
            config.model_provider = provider;
            config.model_context_window = Some(20_000_000);
            config.model_auto_compact_token_limit = Some(20_000_000);
            config.base_instructions = Some(BASE_INSTRUCTIONS.to_owned());
            config.compact_prompt = Some(SUMMARIZATION_PROMPT.to_owned());
            config.update_plan_enabled = true;
            for feature in [
                Feature::TokenBudget,
                Feature::AsyncCompaction,
                Feature::CodeModeOnly,
                Feature::CodeMode,
            ] {
                config
                    .features
                    .disable(feature)
                    .expect("disable unrelated compaction/tool modes");
            }
        })
}

#[test_case(400, false; "http_400")]
#[test_case(413, false; "http_413")]
#[test_case(400, true; "http_400_compaction_cap_decreases")]
#[test_case(413, true; "http_413_compaction_cap_decreases")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn normal_turn_recovers_body_rejection_without_losing_current_input(
    status: u16,
    tighten_summary_cap: bool,
) -> Result<()> {
    let server = responses::start_mock_server().await;
    let state = Arc::new(Mutex::new(BodyCapState::default()));
    Mock::given(method("POST"))
        .and(path_regex(".*/responses$"))
        .respond_with(OwnedBodyCap {
            status,
            tighten_summary_cap,
            state: Arc::clone(&state),
        })
        .mount(&server)
        .await;
    let starts = Arc::new(ToolStarts::default());
    let test = body_recovery_test_builder(&server, Arc::clone(&starts))
        .build_with_auto_env(&server)
        .await?;

    for index in 0..HISTORICAL_TURNS {
        let mut seed = vec![text_input(format!(
            "{}\n{}",
            historical_marker(index),
            "synthetic historical context ".repeat(1800),
        ))];
        if index == 0 {
            seed.push(image_input());
        }
        let submission = test
            .codex
            .start_or_steer_turn(TurnInputRequest::user_input(seed))
            .await?;
        let turn_id = match submission {
            TurnInputSubmission::Started { turn_id, .. }
            | TurnInputSubmission::Steered { turn_id, .. } => turn_id,
            other => panic!("seed input was not admitted: {other:?}"),
        };
        assert_eq!(
            finish_without_error(&test.codex, &turn_id).await,
            format!("HISTORICAL_DONE_{index:02}")
        );
    }
    let expected_calls = (0..HISTORICAL_TURNS).map(call_id).collect::<Vec<_>>();
    assert_eq!(
        *starts.0.lock().expect("recorded requests mutex"),
        expected_calls
    );
    test.codex.flush_rollout().await?;
    let rollout_path = test.codex.rollout_path().expect("canonical rollout");
    let original_rollout = fs::read(&rollout_path)?;
    assert!(String::from_utf8_lossy(&original_rollout).contains(IMAGE_URL));
    {
        let mut state = state.lock().expect("recorded requests mutex");
        assert_eq!(state.seed_requests, 2 * HISTORICAL_TURNS);
        state.armed = true;
    }

    let current_text = format!(
        "{CURRENT_MARKER}\n{}",
        (0..4096)
            .map(|index| format!("line {index:05}: café 東京 🦀 preserve this exact block.\n"))
            .collect::<String>(),
    );
    assert!(approx_token_count(&current_text) > 20_000);
    let tail = "CURRENT_FINAL_BLOCK: keep naïve résumé, 東京 and 🦀 intact.";
    let submission = test
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![
            text_input(current_text.clone()),
            text_input(String::new()),
            image_input(),
            text_input(tail.to_owned()),
        ]))
        .await?;
    let current_turn_id = match submission {
        TurnInputSubmission::Started { turn_id, .. }
        | TurnInputSubmission::Steered { turn_id, .. } => turn_id,
        other => panic!("current input was not admitted: {other:?}"),
    };
    assert_eq!(
        finish_without_error(&test.codex, &current_turn_id).await,
        FINAL
    );
    assert_eq!(
        *starts.0.lock().expect("recorded requests mutex"),
        expected_calls,
        "historical tools must not run again"
    );

    let requests = responses::received_responses_requests(&server).await;
    let recovery = &requests[2 * HISTORICAL_TURNS..];
    let (limit, summary_index) = {
        let state = state.lock().expect("recorded requests mutex");
        let limit = state.limit.expect("activated cap");
        assert_eq!(recovery.len(), state.attempts.len());
        let generation = state
            .attempts
            .iter()
            .filter(|attempt| !attempt.summary)
            .collect::<Vec<_>>();
        assert_eq!(
            generation.len(),
            2,
            "one rejected generation and one same-turn retry"
        );
        assert!(generation[0].rejected && generation[0].bytes > limit);
        assert!(!generation[1].rejected && generation[1].bytes <= limit);
        let successful_summary = state
            .attempts
            .iter()
            .enumerate()
            .filter(|(_, attempt)| attempt.summary && !attempt.rejected)
            .collect::<Vec<_>>();
        assert_eq!(successful_summary.len(), 1);
        assert_eq!(
            state
                .attempts
                .iter()
                .filter(|attempt| attempt.summary && attempt.rejected)
                .count(),
            usize::from(tighten_summary_cap)
        );
        let summary_index = successful_summary[0].0;
        assert!(summary_index > 0 && summary_index + 1 == recovery.len() - 1);
        for (request, attempt) in recovery.iter().zip(&state.attempts) {
            assert_eq!(request.body_bytes().len(), attempt.bytes);
            assert_eq!(attempt.rejected, attempt.bytes > limit);
        }
        (limit, summary_index)
    };

    let rejected = recovery[0].body_json();
    let summary = recovery[summary_index].body_json();
    let retry = recovery.last().expect("final captured request").body_json();
    if tighten_summary_cap {
        assert_eq!(
            summary_index, 2,
            "one smaller selected prefix after authoritative summary rejection"
        );
        let rejected_summary = recovery[1].body_json();
        let selected = input(&summary);
        let previous = input(&rejected_summary);
        assert!(selected.len() < previous.len());
        assert_eq!(
            &selected[..selected.len() - 1],
            &previous[..selected.len() - 1],
            "shrinking changes the selected END only"
        );
        assert_eq!(
            selected.last(),
            previous.last(),
            "same summary request prompt"
        );
        assert!(recovery[summary_index].body_bytes().len() < recovery[1].body_bytes().len());
    }
    let current = input(&rejected)
        .iter()
        .find(|item| item_contains_text(item, CURRENT_MARKER))
        .expect("current input in captured request");
    assert!(item_contains_text(current, &current_text));
    assert!(item_contains_text(current, tail));
    let current_text_blocks = current["content"]
        .as_array()
        .expect("native JSON array")
        .iter()
        .filter_map(|part| part["text"].as_str())
        .collect::<Vec<_>>();
    assert!(current_text_blocks.contains(&current_text.as_str()));
    assert!(current_text_blocks.contains(&""));
    assert!(current_text_blocks.contains(&tail));
    assert!(
        current["content"]
            .as_array()
            .expect("native JSON array")
            .iter()
            .any(|part| part["image_url"] == IMAGE_URL)
    );
    assert_eq!(
        input(&retry)
            .iter()
            .filter(|item| item_contains_text(item, CURRENT_MARKER))
            .collect::<Vec<_>>(),
        vec![current]
    );
    for field in [
        "model",
        "instructions",
        "stream",
        "service_tier",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "store",
        "stream_options",
        "include",
        "text",
        "access_programs",
    ] {
        assert_eq!(
            retry.get(field),
            rejected.get(field),
            "generation field {field}"
        );
    }
    for item in input(&rejected)
        .iter()
        .filter(|item| item["role"] == "developer")
    {
        assert!(
            input(&retry).contains(item),
            "mandatory developer instructions changed"
        );
    }
    assert!(recovery[0].instructions_text().contains(BASE_INSTRUCTIONS));
    assert!(
        recovery
            .last()
            .expect("final captured request")
            .instructions_text()
            .contains(BASE_INSTRUCTIONS)
    );

    let selected_calls = input(&summary)
        .iter()
        .filter(|item| item["type"] == "function_call")
        .map(|item| {
            item["call_id"]
                .as_str()
                .expect("JSON text field")
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert!(!selected_calls.is_empty() && selected_calls.len() < HISTORICAL_TURNS);
    assert_eq!(selected_calls, expected_calls[..selected_calls.len()]);
    for id in &selected_calls {
        assert_eq!(
            input(&summary)
                .iter()
                .filter(|item| item["type"] == "function_call_output" && item["call_id"] == *id)
                .count(),
            1
        );
    }
    let selected_end = input(&rejected)
        .iter()
        .rposition(|item| is_historical(item) && input(&summary).contains(item))
        .expect("selected historical boundary")
        + 1;
    let original_suffix = &input(&rejected)[selected_end..];
    assert!(
        input(&retry).ends_with(original_suffix),
        "nonselected suffix/current changed"
    );
    assert!(input(&summary).iter().any(|item| {
        item_contains_text(item, &historical_marker(0))
            && item["content"]
                .as_array()
                .expect("native JSON array")
                .iter()
                .any(|part| part["image_url"] == IMAGE_URL)
    }));
    assert!(!contains_text(&summary, CURRENT_MARKER));
    assert!(contains_text(
        &retry,
        &format!("{SUMMARY_PREFIX}\n{SUMMARY}")
    ));
    assert!(!contains_text(&retry, &historical_marker(0)));
    let history = test.codex.conversation_history_snapshot().await;
    let history = serde_json::to_value(history.items().collect::<Vec<_>>())?;
    let retained_current = history
        .as_array()
        .expect("native JSON array")
        .iter()
        .filter(|item| item_contains_text(item, CURRENT_MARKER))
        .collect::<Vec<_>>();
    assert_eq!(retained_current.len(), 1);
    assert_eq!(retained_current[0]["content"], current["content"]);

    // The fixed cap permits the complete current request; no oversized single group is required.
    let mut irreducible = rejected.clone();
    irreducible["input"] = Value::Array(
        input(&rejected)
            .iter()
            .filter(|item| !is_historical(item))
            .cloned()
            .collect(),
    );
    assert!(serde_json::to_vec(&irreducible)?.len() + 64 * 1024 < limit);
    for index in 0..HISTORICAL_TURNS {
        let group_bytes = input(&rejected)
            .iter()
            .filter(|item| {
                item_contains_text(item, &historical_marker(index))
                    || item_contains_text(item, &format!("HISTORICAL_DONE_{index:02}"))
                    || item["call_id"] == call_id(index)
            })
            .map(|item| {
                serde_json::to_vec(item)
                    .expect("serialize captured input item")
                    .len()
            })
            .sum::<usize>();
        assert!(group_bytes > 0 && group_bytes < limit);
    }
    let mut full_summary = summary.clone();
    let mut full_input = input(&rejected)
        .iter()
        .filter(|item| !item_contains_text(item, CURRENT_MARKER))
        .cloned()
        .collect::<Vec<_>>();
    full_input.push(json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": SUMMARIZATION_PROMPT}]}));
    full_summary["input"] = Value::Array(full_input);
    assert!(
        serde_json::to_vec(&full_summary)?.len() > limit,
        "a smaller selected boundary is necessary"
    );
    test.codex.flush_rollout().await?;
    assert!(
        fs::read(rollout_path)?.starts_with(&original_rollout),
        "original canonical records changed"
    );
    Ok(())
}
