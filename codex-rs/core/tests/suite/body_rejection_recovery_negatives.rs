//! Refusal and cancellation controls through ordinary native turns and owned HTTP responses.

use std::fs;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;
use codex_core::CodexThread;
use codex_core::TurnInputRequest;
use codex_core::TurnInputSubmission;
use codex_core::compact::SUMMARIZATION_PROMPT;
use codex_history::RolloutItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use core_test_support::responses;
use core_test_support::test_codex::TestCodex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use test_case::test_case;
use tokio::sync::Notify;
use tokio::time::timeout;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path_regex;

use super::body_rejection_recovery::ToolStarts;
use super::body_rejection_recovery::body_recovery_test_builder;
use super::body_rejection_recovery::completed_sse;
use super::body_rejection_recovery::finish_without_error;
use super::body_rejection_recovery::image_input;
use super::body_rejection_recovery::input;
use super::body_rejection_recovery::item_contains_text;
use super::body_rejection_recovery::text_input;
use super::client::ProviderAuthCommandFixture;

const SEED_MARKER: &str = "NEGATIVE_COMPLETED_HISTORICAL_USER";
const SEED_DONE: &str = "NEGATIVE_COMPLETED_HISTORICAL_ASSISTANT";
const SEED_CALL: &str = "negative-completed-plan";
const CURRENT_MARKER: &str = "NEGATIVE_CURRENT_UTF8_MULTIBLOCK";
const CURRENT_TAIL: &str = "NEGATIVE_CURRENT_TAIL: naïve 東京 🦀 stays exact.";
const SUMMARY_FRAGMENT: &str = "UNUSABLE_SYNTHETIC_SUMMARY";
const FOLLOWUP: &str = "NEGATIVE_FOLLOWUP_AFTER_CANCEL";
const FOLLOWUP_DONE: &str = "NEGATIVE_FOLLOWUP_OK";
const WAIT_BOUND: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug)]
enum NegativeCase {
    NoCompletedPrefix,
    ImpossibleCurrent,
    NestedEnvelope,
    MalformedGrammar,
    WrongStatus,
    MismatchedBytes,
    SummaryEof,
    SummaryTool,
    SummaryFailed,
    SummaryIncomplete,
    CancelSummary,
    AutoDisabled,
    TokenBudget,
    AuthRetry,
}

impl NegativeCase {
    fn has_seed(self) -> bool {
        !matches!(self, Self::NoCompletedPrefix)
    }

    fn reaches_summary(self) -> bool {
        matches!(
            self,
            Self::SummaryEof
                | Self::SummaryTool
                | Self::SummaryFailed
                | Self::SummaryIncomplete
                | Self::CancelSummary
        )
    }
}

#[derive(Default)]
struct SummaryGate {
    entered: Notify,
    returned: Notify,
    released: Mutex<bool>,
    release_changed: Condvar,
}

impl SummaryGate {
    fn wait_for_release(&self) {
        self.entered.notify_one();
        let (released, wait) = self
            .release_changed
            .wait_timeout_while(
                self.released.lock().expect("summary gate mutex"),
                WAIT_BOUND,
                |released| !*released,
            )
            .expect("summary gate wait");
        assert!(
            *released && !wait.timed_out(),
            "summary release barrier timed out"
        );
        self.returned.notify_one();
    }

    fn release(&self) {
        *self.released.lock().expect("summary gate mutex") = true;
        self.release_changed.notify_all();
    }
}

struct ReleaseOnDrop(Arc<SummaryGate>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

#[derive(Default)]
struct RecordedRequests {
    seed: usize,
    kinds: Vec<&'static str>,
    limit: Option<usize>,
}

struct NegativeResponder {
    case: NegativeCase,
    requests: Arc<Mutex<RecordedRequests>>,
    gate: Arc<SummaryGate>,
}

impl Respond for NegativeResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        assert!(request.headers.get("content-encoding").is_none());
        let body: Value =
            serde_json::from_slice(&request.body).expect("uncompressed native request");
        let mut recorded = self.requests.lock().expect("recorded requests mutex");
        if self.case.has_seed() && recorded.seed < 2 {
            assert!(
                input(&body)
                    .iter()
                    .any(|item| item_contains_text(item, SEED_MARKER))
            );
            let output = if recorded.seed == 0 {
                responses::ev_function_call(SEED_CALL, "update_plan", &plan_arguments())
            } else {
                responses::ev_assistant_message("negative-seed-message", SEED_DONE)
            };
            let id = format!("negative-seed-response-{}", recorded.seed);
            recorded.seed += 1;
            return completed_sse(&id, output);
        }

        let summary = body["tools"] == json!([]);
        let kind = if summary { "summary" } else { "generation" };
        recorded.kinds.push(kind);
        let attempt = recorded.kinds.len();
        if matches!(self.case, NegativeCase::AuthRetry) && attempt <= 2 {
            assert!(!summary);
            assert!(
                input(&body)
                    .iter()
                    .any(|item| item_contains_text(item, CURRENT_MARKER))
            );
            let token = request
                .headers
                .get("authorization")
                .expect("command auth")
                .to_str()
                .expect("auth header text");
            if attempt == 1 {
                assert_eq!(token, "Bearer first-token");
                return ResponseTemplate::new(401).set_body_string("unauthorized");
            }
            assert_eq!(token, "Bearer first-token");
            recorded.limit = Some(declared_limit(self.case, request.body.len()));
            return rejected_body(self.case, request.body.len());
        }
        if attempt == 1 {
            assert!(!summary);
            assert!(
                input(&body)
                    .iter()
                    .any(|item| item_contains_text(item, CURRENT_MARKER))
            );
            recorded.limit = Some(declared_limit(self.case, request.body.len()));
            return rejected_body(self.case, request.body.len());
        }
        if summary && attempt == 2 && self.case.reaches_summary() {
            assert!(
                request.body.len() <= recorded.limit.expect("measured body cap"),
                "the synthetic summary request must fit"
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
            assert!(
                input(&body)
                    .iter()
                    .any(|item| item["call_id"] == SEED_CALL
                        && item["type"] == "function_call_output")
            );
            drop(recorded);
            if matches!(self.case, NegativeCase::CancelSummary) {
                // The request is in flight; no summary completion is released until after Interrupt.
                self.gate.wait_for_release();
                return completed_sse(
                    "negative-late-summary",
                    responses::ev_assistant_message("negative-late-message", SUMMARY_FRAGMENT),
                );
            }
            return unusable_summary(self.case);
        }
        if matches!(self.case, NegativeCase::CancelSummary)
            && attempt == 3
            && !summary
            && input(&body)
                .iter()
                .any(|item| item_contains_text(item, FOLLOWUP))
        {
            return completed_sse(
                "negative-followup-response",
                responses::ev_assistant_message("negative-followup-message", FOLLOWUP_DONE),
            );
        }
        ResponseTemplate::new(500).set_body_json(json!({"error": "unexpected native retry"}))
    }
}

fn plan_arguments() -> String {
    json!({"plan": [{"step": "Complete harmless synthetic work", "status": "completed"}]})
        .to_string()
}

fn declared_limit(case: NegativeCase, actual_bytes: usize) -> usize {
    if matches!(case, NegativeCase::ImpossibleCurrent) {
        1
    } else {
        actual_bytes - 1
    }
}

fn rejected_body(case: NegativeCase, actual_bytes: usize) -> ResponseTemplate {
    let limit = declared_limit(case, actual_bytes);
    let declared = actual_bytes + usize::from(matches!(case, NegativeCase::MismatchedBytes));
    let message = format!(
        "Server received a request which exceeds maximum allowed content length. RequestSize(bytes): {declared}, Limit(bytes): {limit}"
    );
    let body = match case {
        NegativeCase::NestedEnvelope => {
            json!({"error": {"error_code": "BAD_REQUEST", "message": message}})
        }
        NegativeCase::MalformedGrammar => json!({
            "error_code": "BAD_REQUEST",
            "message": format!("Request exceeds maximum allowed content length. RequestSize(bytes): {declared} Limit(bytes): {limit}"),
        }),
        _ => json!({"error_code": "BAD_REQUEST", "message": message}),
    };
    let status = if matches!(case, NegativeCase::WrongStatus) {
        422
    } else {
        400
    };
    ResponseTemplate::new(status).set_body_json(body)
}

fn unusable_summary(case: NegativeCase) -> ResponseTemplate {
    let mut events = vec![
        responses::ev_response_created("negative-summary-response"),
        responses::ev_assistant_message("negative-summary-fragment", SUMMARY_FRAGMENT),
    ];
    match case {
        NegativeCase::SummaryEof => {}
        NegativeCase::SummaryTool => {
            events.push(responses::ev_function_call("must-not-dispatch", "update_plan", &plan_arguments()));
            events.push(responses::ev_completed("negative-summary-response"));
        }
        NegativeCase::SummaryFailed => events.push(json!({
            "type": "response.failed",
            "response": {"id": "negative-summary-response", "error": {"code": "server_error", "message": "synthetic summary failure"}},
        })),
        NegativeCase::SummaryIncomplete => events.push(json!({
            "type": "response.incomplete",
            "response": {"id": "negative-summary-response", "incomplete_details": {"reason": "max_output_tokens"}},
        })),
        _ => panic!("not an unusable summary case: {case:?}"),
    }
    responses::sse_response(responses::sse(events))
}

fn admitted_turn_id(submission: TurnInputSubmission) -> String {
    match submission {
        TurnInputSubmission::Started { turn_id, .. } => turn_id,
        other => panic!("idle native turn was not started: {other:?}"),
    }
}

async fn build_negative_case(
    case: NegativeCase,
) -> Result<(
    TestCodex,
    MockServer,
    Arc<ToolStarts>,
    Arc<Mutex<RecordedRequests>>,
    Arc<SummaryGate>,
    Option<ProviderAuthCommandFixture>,
)> {
    let server = responses::start_mock_server().await;
    let requests = Arc::new(Mutex::new(RecordedRequests::default()));
    let gate = Arc::new(SummaryGate::default());
    Mock::given(method("POST"))
        .and(path_regex(".*/responses$"))
        .respond_with(NegativeResponder {
            case,
            requests: Arc::clone(&requests),
            gate: Arc::clone(&gate),
        })
        .mount(&server)
        .await;
    let starts = Arc::new(ToolStarts::default());
    let mut builder = body_recovery_test_builder(&server, Arc::clone(&starts));
    let auth_fixture = if matches!(case, NegativeCase::AuthRetry) {
        let fixture = ProviderAuthCommandFixture::fixed_token("first-token")?;
        let auth = fixture.auth();
        builder = builder.with_config(move |config| {
            config.model_provider.requires_openai_auth = false;
            config.model_provider.auth = Some(auth);
        });
        Some(fixture)
    } else {
        None
    };
    if matches!(case, NegativeCase::TokenBudget) {
        builder = builder.with_config(|config| {
            config
                .features
                .enable(codex_features::Feature::TokenBudget)
                .expect("enable reset-only policy");
        });
    }
    let mut test = builder.build_with_auto_env(&server).await?;
    if case.has_seed() {
        let turn_id = admitted_turn_id(
            test.codex
                .start_or_steer_turn(TurnInputRequest::user_input(vec![text_input(format!(
                    "{SEED_MARKER}\n{}",
                    "completed historical context ".repeat(1024)
                ))]))
                .await?,
        );
        assert_eq!(finish_without_error(&test.codex, &turn_id).await, SEED_DONE);
        assert_eq!(starts.snapshot(), vec![SEED_CALL.to_owned()]);
    }
    if matches!(case, NegativeCase::AutoDisabled) {
        test.codex.flush_rollout().await?;
        let (mut options, settings) = test
            .thread_manager
            .fork_options_from_parent(test.session_configured.thread_id)
            .await?;
        options.config.ephemeral = false;
        options
            .thread_extension_init
            .insert(codex_core::compact::AutoCompactionDisabled);
        let child = test
            .thread_manager
            .fork_legacy_thread(
                codex_core::ForkSnapshot::Interrupted,
                options,
                test.codex.rollout_path().expect("seed rollout"),
            )
            .await?;
        child.thread.restore_thread_settings(settings).await?;
        test.codex = child.thread;
    }
    Ok((test, server, starts, requests, gate, auth_fixture))
}

async fn submit_current(test: &TestCodex) -> Result<String> {
    Ok(admitted_turn_id(
        test.codex
            .start_or_steer_turn(TurnInputRequest::user_input(vec![
                text_input(format!(
                    "{CURRENT_MARKER}\n{}",
                    "current café 東京 🦀 stays exact\n".repeat(256)
                )),
                text_input(String::new()),
                image_input(),
                text_input(CURRENT_TAIL.to_owned()),
            ]))
            .await?,
    ))
}

async fn finish_with_preserved_failure(codex: &CodexThread, expected_turn_id: &str) {
    let mut errors = 0;
    let event = wait_for_event(codex, |event| {
        match event {
            EventMsg::Error(_) => errors += 1,
            EventMsg::ContextCompacted(_) => panic!("failed recovery published a checkpoint"),
            EventMsg::StreamError(_) => panic!("failed recovery retried a stream"),
            EventMsg::TurnAborted(_) => panic!("refusal unexpectedly aborted"),
            _ => {}
        }
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let EventMsg::TurnComplete(completed) = event else {
        unreachable!()
    };
    assert_eq!(completed.turn_id, expected_turn_id);
    assert!(
        completed.error.is_some(),
        "a refusal must remain a visible terminal failure"
    );
    assert_eq!(errors, 1, "one final error, without duplicate failures");
}

async fn history_json(codex: &CodexThread) -> Result<Vec<Value>> {
    let history = codex.conversation_history_snapshot().await;
    Ok(serde_json::to_value(history.items().collect::<Vec<_>>())?
        .as_array()
        .expect("native JSON array")
        .clone())
}

async fn assert_no_checkpoint_or_input_loss(
    test: &TestCodex,
    server: &MockServer,
    original_history: &[Value],
    original_rollout: &[u8],
    seeded: bool,
) -> Result<()> {
    let requests = responses::received_responses_requests(server).await;
    let generation = requests[if seeded { 2 } else { 0 }].body_json();
    let current = input(&generation)
        .iter()
        .find(|item| item_contains_text(item, CURRENT_MARKER))
        .expect("current input in captured request");
    assert!(item_contains_text(current, CURRENT_TAIL));
    let history = history_json(&test.codex).await?;
    assert!(
        history.starts_with(original_history),
        "original historical records changed"
    );
    let current_items = history
        .iter()
        .filter(|item| item_contains_text(item, CURRENT_MARKER))
        .collect::<Vec<_>>();
    assert_eq!(current_items.len(), 1);
    assert_eq!(current_items[0]["content"], current["content"]);
    assert!(
        !history
            .iter()
            .any(|item| item_contains_text(item, SUMMARY_FRAGMENT))
    );
    test.codex.flush_rollout().await?;
    let rollout = fs::read(test.codex.rollout_path().expect("canonical rollout"))?;
    assert!(
        rollout.starts_with(original_rollout),
        "original canonical bytes changed"
    );
    for line in std::str::from_utf8(&rollout)?.lines() {
        assert!(
            !matches!(
                codex_rollout::parse_rollout_line(line)?.item,
                RolloutItem::Compacted(_)
            ),
            "unusable recovery installed a checkpoint"
        );
    }
    Ok(())
}

#[test_case(NegativeCase::NoCompletedPrefix; "no_completed_prefix")]
#[test_case(NegativeCase::ImpossibleCurrent; "irreducible_current_cannot_fit")]
#[test_case(NegativeCase::NestedEnvelope; "nested_envelope")]
#[test_case(NegativeCase::MalformedGrammar; "malformed_grammar")]
#[test_case(NegativeCase::WrongStatus; "wrong_http_status")]
#[test_case(NegativeCase::MismatchedBytes; "declared_bytes_do_not_match_wire")]
#[test_case(NegativeCase::SummaryEof; "summary_eof")]
#[test_case(NegativeCase::SummaryTool; "summary_contains_executable_tool")]
#[test_case(NegativeCase::SummaryFailed; "summary_failed")]
#[test_case(NegativeCase::SummaryIncomplete; "summary_incomplete")]
#[test_case(NegativeCase::AutoDisabled; "automatic_compaction_disabled")]
#[test_case(NegativeCase::TokenBudget; "token_budget_preserves_reset_only_policy")]
#[test_case(NegativeCase::AuthRetry; "outer_auth_refresh_refuses_body_recovery")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn normal_turn_recovery_refuses_without_checkpoint_or_generation_retry(
    case: NegativeCase,
) -> Result<()> {
    let (test, server, starts, recorded, _gate, _auth_fixture) = build_negative_case(case).await?;
    let original_history = history_json(&test.codex).await?;
    test.codex.ensure_rollout_materialized().await;
    test.codex.flush_rollout().await?;
    let original_rollout = fs::read(test.codex.rollout_path().expect("canonical rollout"))?;
    let expected_tools = starts.snapshot();
    let turn_id = submit_current(&test).await?;
    finish_with_preserved_failure(&test.codex, &turn_id).await;
    assert_eq!(
        starts.snapshot(),
        expected_tools,
        "recovery must not dispatch historical or summary tools"
    );
    let expected = if matches!(case, NegativeCase::AuthRetry) {
        vec!["generation", "generation"]
    } else if case.reaches_summary() {
        vec!["generation", "summary"]
    } else {
        vec!["generation"]
    };
    assert_eq!(
        recorded.lock().expect("recorded requests mutex").kinds,
        expected
    );
    let requests = responses::received_responses_requests(&server).await;
    assert_eq!(
        requests.len(),
        usize::from(case.has_seed()) * 2 + expected.len()
    );
    if matches!(case, NegativeCase::AuthRetry) {
        assert_eq!(
            requests[2].body_json(),
            requests[3].body_json(),
            "auth refresh preserves every submitted field despite metadata key ordering"
        );
    }
    assert_no_checkpoint_or_input_loss(
        &test,
        &server,
        &original_history,
        &original_rollout,
        case.has_seed(),
    )
    .await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupt_during_summary_http_discards_late_completion_and_never_retries_old_turn()
-> Result<()> {
    let (test, server, starts, recorded, gate, _auth_fixture) =
        build_negative_case(NegativeCase::CancelSummary).await?;
    let _release_on_drop = ReleaseOnDrop(Arc::clone(&gate));
    let original_history = history_json(&test.codex).await?;
    test.codex.flush_rollout().await?;
    let original_rollout = fs::read(test.codex.rollout_path().expect("canonical rollout"))?;
    let turn_id = submit_current(&test).await?;
    timeout(WAIT_BOUND, gate.entered.notified())
        .await
        .expect("local summary request never arrived");
    test.codex.submit(Op::Interrupt).await?;
    let aborted = wait_for_event(&test.codex, |event| {
        assert!(
            !matches!(event, EventMsg::Error(_) | EventMsg::ContextCompacted(_)),
            "cancelled recovery must not fail or install a checkpoint: {event:?}"
        );
        matches!(event, EventMsg::TurnAborted(_))
    })
    .await;
    let EventMsg::TurnAborted(aborted) = aborted else {
        unreachable!()
    };
    assert_eq!(aborted.turn_id.as_deref(), Some(turn_id.as_str()));
    assert!(aborted.error.is_none());
    gate.release();
    timeout(WAIT_BOUND, gate.returned.notified())
        .await
        .expect("late summary release did not return");
    let followup_id = admitted_turn_id(
        test.codex
            .start_or_steer_turn(TurnInputRequest::user_input(vec![text_input(
                FOLLOWUP.to_owned(),
            )]))
            .await?,
    );
    assert_eq!(
        finish_without_error(&test.codex, &followup_id).await,
        FOLLOWUP_DONE
    );
    assert_eq!(starts.snapshot(), vec![SEED_CALL.to_owned()]);
    assert_eq!(
        recorded.lock().expect("recorded requests mutex").kinds,
        vec!["generation", "summary", "generation"]
    );
    assert_eq!(
        responses::received_responses_requests(&server).await.len(),
        5
    );
    let history = history_json(&test.codex).await?;
    assert_eq!(
        history
            .iter()
            .filter(|item| item_contains_text(item, FOLLOWUP))
            .count(),
        1
    );
    assert_no_checkpoint_or_input_loss(&test, &server, &original_history, &original_rollout, true)
        .await?;
    Ok(())
}
