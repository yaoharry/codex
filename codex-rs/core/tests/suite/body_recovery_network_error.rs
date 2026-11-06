//! Closing the owned summary transport before headers cannot trigger another inference send.

use std::fs;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_core::TurnInputSubmission;
use codex_core::compact::SUMMARIZATION_PROMPT;
use codex_protocol::protocol::EventMsg;
use codex_rollout::RolloutItem;
use core_test_support::responses;
use core_test_support::streaming_sse::read_http_request;
use core_test_support::streaming_sse::read_request_body;
use core_test_support::streaming_sse::write_http_response;
use core_test_support::wait_for_event_with_timeout;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use test_case::test_case;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::time::timeout;

use super::body_rejection_recovery::ToolStarts;
use super::body_rejection_recovery::body_recovery_test_builder;
use super::body_rejection_recovery::finish_without_error;
use super::body_rejection_recovery::input;
use super::body_rejection_recovery::item_contains_text;
use super::body_rejection_recovery::text_input;

const SEED_MARKER: &str = "NETWORK_ERROR_COMPLETED_HISTORY";
const SEED_DONE: &str = "NETWORK_ERROR_SEED_DONE";
const SEED_CALL: &str = "network-error-seed-plan";
const CURRENT_MARKER: &str = "NETWORK_ERROR_CURRENT_INPUT";
const CURRENT_TAIL: &str = "Current final block: naïve 東京 🦀 remains intact.";

#[test_case(400; "http_400")]
#[test_case(413; "http_413")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn summary_transport_close_before_headers_is_never_replayed(status: u16) -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}/v1", listener.local_addr()?);
    let (shutdown, mut shutdown_requested) = oneshot::channel();
    let server_task = tokio::spawn(async move {
        let mut requests: Vec<Vec<u8>> = Vec::new();
        let mut unexpected_routes = Vec::new();
        loop {
            tokio::select! {
                biased;
                _ = &mut shutdown_requested => break,
                handled = async {
                    let (mut stream, _) = listener.accept().await?;
                    let (headers, body_prefix) = read_http_request(&mut stream).await;
                    let bytes = read_request_body(&mut stream, &headers, body_prefix).await?;
                    if headers.starts_with("GET /v1/models") {
                        return write_http_response(
                            &mut stream, 200,
                            &json!({"data": [], "object": "list"}).to_string(),
                            "application/json",
                        ).await;
                    }
                    if !headers.starts_with("POST /v1/responses ") {
                        unexpected_routes.push(headers.lines().next().unwrap_or_default().to_owned());
                        return write_http_response(&mut stream, 404, "unexpected route", "text/plain").await;
                    }
                    assert!(!headers.to_ascii_lowercase().contains("\r\ncontent-encoding:"));
                    let body: Value = serde_json::from_slice(&bytes).expect("uncompressed native JSON");
                    let index = requests.len();
                    requests.push(bytes);
                    if index < 2 {
                        assert!(body["tools"] != json!([]));
                        assert!(input(&body).iter().any(|item| item_contains_text(item, SEED_MARKER)));
                        let (response_id, output) = if index == 0 {
                            (
                                "network-error-seed-tool-response",
                                responses::ev_function_call(
                                    SEED_CALL, "update_plan",
                                    &json!({"plan": [{
                                        "step": "Completed synthetic seed work.",
                                        "status": "completed"
                                    }]}).to_string(),
                                ),
                            )
                        } else {
                            assert_eq!(input(&body).iter().filter(|item| {
                                item["type"] == "function_call_output" && item["call_id"] == SEED_CALL
                            }).count(), 1);
                            (
                                "network-error-seed-final-response",
                                responses::ev_assistant_message("network-error-seed-final", SEED_DONE),
                            )
                        };
                        let response = responses::sse(vec![
                            responses::ev_response_created(response_id),
                            output,
                            responses::ev_completed(response_id),
                        ]);
                        write_http_response(&mut stream, 200, &response, "text/event-stream").await?;
                    } else if index == 2 {
                        assert!(body["tools"] != json!([]));
                        assert!(input(&body).iter().any(|item| item_contains_text(item, CURRENT_MARKER)));
                        let bytes = requests[index].len();
                        let limit = bytes.checked_sub(1).expect("nonempty owned request");
                        let response = json!({
                            "error_code": "BAD_REQUEST",
                            "message": format!(
                                "Server received a request which exceeds maximum allowed content length. RequestSize(bytes): {bytes}, Limit(bytes): {limit}"
                            )
                        }).to_string();
                        write_http_response(&mut stream, i64::from(status), &response, "application/json").await?;
                    } else {
                        if index == 3 {
                            assert_eq!(body["tools"], json!([]));
                            assert_eq!(body["tool_choice"], "none");
                            assert_eq!(body["parallel_tool_calls"], false);
                            assert!(requests[index].len() < requests[2].len());
                            assert!(input(&body).iter().any(|item| item_contains_text(item, SUMMARIZATION_PROMPT)));
                            assert!(!input(&body).iter().any(|item| item_contains_text(item, CURRENT_MARKER)));
                        }
                        // The full body was received; no response status line or headers are sent.
                        drop(stream);
                    }
                    Ok::<(), std::io::Error>(())
                } => handled?,
            }
        }
        Ok::<_, std::io::Error>((requests, unexpected_routes))
    });

    let dummy_server = responses::start_mock_server().await;
    let starts = Arc::new(ToolStarts::default());
    let test = body_recovery_test_builder(&dummy_server, Arc::clone(&starts))
        .with_config(move |config| {
            config.model_provider.base_url = Some(base_url);
            config.model_provider.request_max_retries = Some(2);
            config.model_provider.stream_max_retries = Some(2);
        })
        .build_with_auto_env(&dummy_server)
        .await?;
    let seed = test
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![text_input(format!(
            "{SEED_MARKER}\n{}",
            "completed synthetic historical evidence ".repeat(4096)
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
    let seed_history = test.codex.conversation_history_snapshot().await;
    let seed_history = serde_json::to_value(seed_history.items().collect::<Vec<_>>())?;
    let seed_history = seed_history.as_array().expect("native JSON array").clone();
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
    let mut errors = Vec::new();
    let mut turn_starts = 0;
    let terminal = timeout(
        Duration::from_secs(30),
        wait_for_event_with_timeout(
            &test.codex,
            |event| {
                match event {
                    EventMsg::TurnStarted(started) => {
                        assert_eq!(started.turn_id, current_turn);
                        turn_starts += 1;
                    }
                    EventMsg::Error(error) => errors.push(error.message.clone()),
                    EventMsg::TurnAborted(_) => {
                        panic!("transport failure must not replace or abort the turn")
                    }
                    EventMsg::StreamError(_) => {
                        panic!("ambiguous summary must not retry inference")
                    }
                    EventMsg::ContextCompacted(_) => {
                        panic!("incomplete summary cannot install a checkpoint")
                    }
                    EventMsg::RawResponseCompleted(_) => {
                        panic!("no current response completed after the transport close")
                    }
                    _ => {}
                }
                matches!(event, EventMsg::TurnComplete(_))
            },
            Duration::from_secs(10),
        ),
    )
    .await;
    let _ = shutdown.send(());
    let (requests, unexpected_routes) = timeout(Duration::from_secs(5), server_task)
        .await
        .expect("owned TCP listener must shut down")
        .expect("owned TCP listener task")?;
    let terminal = terminal.expect("the original turn must settle within the fixed deadline");
    let EventMsg::TurnComplete(completed) = terminal else {
        unreachable!()
    };
    assert_eq!(completed.turn_id, current_turn);
    assert!(
        completed.error.is_some(),
        "network failure must remain visible"
    );
    assert_eq!(turn_starts, 1, "no new or replacement user turn");
    assert_eq!(
        errors.len(),
        1,
        "one terminal failure without duplicate errors"
    );
    assert_eq!(starts.snapshot(), vec![SEED_CALL.to_owned()]);
    assert!(unexpected_routes.is_empty(), "{unexpected_routes:?}");
    assert_eq!(
        requests.len(),
        4,
        "two seed POSTs, one owned cap, exactly one summary send; no replay",
    );
    let bodies = requests
        .iter()
        .map(|body| serde_json::from_slice::<Value>(body))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    assert_eq!(
        bodies
            .iter()
            .map(|body| body["tools"] == json!([]))
            .collect::<Vec<_>>(),
        vec![false, false, false, true],
    );
    let first_limit = requests[2].len() - 1;
    assert!(
        requests[3].len() <= first_limit,
        "summary was sent below the measured cap"
    );
    let current = input(&bodies[2])
        .iter()
        .find(|item| item_contains_text(item, CURRENT_MARKER))
        .expect("original current input");
    let blocks = current["content"]
        .as_array()
        .expect("native JSON array")
        .iter()
        .filter_map(|block| block["text"].as_str())
        .collect::<Vec<_>>();
    assert!(blocks.contains(&current_text.as_str()));
    assert!(blocks.contains(&""));
    assert!(blocks.contains(&CURRENT_TAIL));
    let summary = &bodies[3];
    assert!(
        !input(summary)
            .iter()
            .any(|item| item_contains_text(item, CURRENT_MARKER))
    );
    assert!(
        input(summary)
            .iter()
            .any(|item| item_contains_text(item, SEED_MARKER))
    );
    for kind in ["function_call", "function_call_output"] {
        assert_eq!(
            input(summary)
                .iter()
                .filter(|item| item["type"] == kind && item["call_id"] == SEED_CALL)
                .count(),
            1
        );
    }

    let history = test.codex.conversation_history_snapshot().await;
    let history = serde_json::to_value(history.items().collect::<Vec<_>>())?;
    let history = history.as_array().expect("native JSON array");
    assert!(
        history.starts_with(&seed_history),
        "historical records remain unchanged"
    );
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
                .filter(|item| item["type"] == kind && item["call_id"] == SEED_CALL)
                .count(),
            1
        );
    }

    test.codex.flush_rollout().await?;
    let rollout = fs::read(rollout_path)?;
    assert!(
        rollout.starts_with(&original_rollout),
        "original canonical bytes changed"
    );
    let mut raw_current = Vec::new();
    for line in std::str::from_utf8(&rollout)?.lines() {
        match codex_rollout::parse_rollout_line(line)?.item {
            RolloutItem::Compacted(_) => {
                panic!("ambiguous summary installed a canonical checkpoint")
            }
            RolloutItem::ResponseItem(envelope) => {
                let item = serde_json::to_value(&envelope.item)?;
                if item_contains_text(&item, CURRENT_MARKER) {
                    raw_current.push(item["content"].clone());
                }
            }
            _ => {}
        }
    }
    assert_eq!(raw_current, vec![current["content"].clone()]);
    Ok(())
}
