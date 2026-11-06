//! Structured failure diagnostics and low-cardinality metrics for Guardian V2 scoring and decisions.

use std::time::Duration;

use codex_api::ApiError;
use codex_api::TransportError;
use codex_core::context::GuardianContextMode;
use codex_extension_api::ExtensionMetrics;

use super::decisions::DecisionsError;
use super::sampler::LunaSamplerError;

pub(super) const CLASSIFICATION_METRIC: &str = "codex.guardian_v2.classification";
pub(super) const CLASSIFICATION_DURATION_METRIC: &str =
    "codex.guardian_v2.classification.duration_ms";
pub(super) const CLASSIFICATION_RISK_METRIC: &str = "codex.guardian_v2.classification.risk";
pub(super) const FAST_DECISION_METRIC: &str = "codex.guardian_v2.fast_decision";
pub(super) const REVIEW_FALLBACK_METRIC: &str = "codex.guardian_v2.review_fallback";
pub(super) const TOOL_CALL_LAG_METRIC: &str = "codex.guardian_v2.tool_call_lag";

// Never use error messages as metric tags: they may contain server responses or credentials.
pub(super) fn sampler_failure_reason(error: &LunaSamplerError) -> &'static str {
    match error {
        LunaSamplerError::Provider(_) => "provider_error",
        LunaSamplerError::ConnectionTimeout => "connection_timeout",
        LunaSamplerError::MissingOutput => "missing_output",
        LunaSamplerError::OutputTooLarge => "output_too_large",
        LunaSamplerError::Superseded => "superseded",
        LunaSamplerError::InputTooLarge => "input_too_large",
        LunaSamplerError::QueueFull => "queue_full",
        LunaSamplerError::Api(error) => match error {
            ApiError::Transport(error) => transport_failure_reason(error),
            ApiError::BodyRejected { source, .. } => transport_failure_reason(source),
            ApiError::Api { status, .. } => http_failure_reason(status.as_u16()),
            ApiError::Stream(_) | ApiError::ContentFilter => "stream_error",
            ApiError::ResponseIncomplete(_) => "incomplete_response",
            ApiError::ContextWindowExceeded => "context_window_exceeded",
            ApiError::QuotaExceeded => "quota_exceeded",
            ApiError::UsageNotIncluded => "usage_not_included",
            ApiError::SessionBudgetExceeded => "session_budget_exceeded",
            ApiError::Retryable { .. } => "retryable_api_error",
            ApiError::RateLimitExceeded { .. } | ApiError::RateLimit(_) => "rate_limit",
            ApiError::ModelNotFound { .. } => "model_not_found",
            ApiError::InvalidRequest { .. }
            | ApiError::InvalidAudioInput { .. }
            | ApiError::InvalidPrompt { .. } => "invalid_request",
            ApiError::CyberPolicy { .. }
            | ApiError::BioPolicy { .. }
            | ApiError::MisalignmentPolicyViolation { .. } => "policy_error",
            ApiError::ServerOverloaded { .. } => "server_overloaded",
            ApiError::PolicyCapacityUnavailable { .. } => "policy_capacity_unavailable",
            ApiError::FlexUnavailable => "flex_unavailable",
        },
    }
}

fn http_failure_reason(status: u16) -> &'static str {
    match status {
        401 => "http_401",
        403 => "http_403",
        429 => "http_429",
        400..=499 => "http_4xx",
        500..=599 => "http_5xx",
        _ => "http_other",
    }
}

fn transport_failure_reason(error: &TransportError) -> &'static str {
    match error {
        TransportError::Http { status, .. } => http_failure_reason(status.as_u16()),
        TransportError::Timeout => "transport_timeout",
        TransportError::Connection(_) => "connection_error",
        TransportError::Network(_) => "network_error",
        TransportError::RetryLimit => "retry_limit",
        TransportError::Build(_) => "request_build_error",
        TransportError::ResponseTooLarge { .. } => "response_too_large",
        TransportError::Policy(_) => "network_policy_denied",
    }
}

pub(super) fn record_classification(
    metrics: Option<&dyn ExtensionMetrics>,
    context_mode: GuardianContextMode,
    duration: Duration,
    outcome: &str,
    failure_reason: Option<&str>,
) {
    if outcome == "failure" {
        tracing::warn!(
            failure_reason,
            context_mode = context_mode.as_str(),
            duration_ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
            "Guardian V2 background scoring failed"
        );
    }
    let Some(metrics) = metrics else {
        return;
    };
    let mut tags = vec![
        ("outcome", outcome),
        ("context_mode", context_mode.as_str()),
    ];
    if let Some(reason) = failure_reason {
        tags.push(("failure_reason", reason));
    }
    metrics.counter(CLASSIFICATION_METRIC, /*inc*/ 1, &tags);
    metrics.histogram(
        CLASSIFICATION_DURATION_METRIC,
        i64::try_from(duration.as_millis()).unwrap_or(i64::MAX),
        &tags,
    );
}

pub(super) fn record_classification_risk(metrics: Option<&dyn ExtensionMetrics>, risk_level: &str) {
    let Some(metrics) = metrics else {
        return;
    };
    metrics.counter(
        CLASSIFICATION_RISK_METRIC,
        /*inc*/ 1,
        &[("risk_level", risk_level)],
    );
}

pub(super) fn record_fast_decision(
    metrics: Option<&dyn ExtensionMetrics>,
    decision: &str,
    reason: &str,
) {
    let Some(metrics) = metrics else {
        return;
    };
    metrics.counter(
        FAST_DECISION_METRIC,
        /*inc*/ 1,
        &[("decision", decision), ("reason", reason)],
    );
}

pub(super) fn record_section_costs(
    metrics: Option<&dyn ExtensionMetrics>,
    costs: impl IntoIterator<Item = (&'static str, codex_guardian_context::SectionCost)>,
) {
    let Some(metrics) = metrics else {
        return;
    };
    for (section, cost) in costs {
        for (measurement, value) in cost.measurements() {
            metrics.histogram_with_boundaries(
                codex_guardian_context::SECTION_COST_METRIC,
                i64::try_from(value).unwrap_or(i64::MAX),
                codex_guardian_context::SECTION_COST_BOUNDARIES,
                &[
                    ("target", "async"),
                    ("section", section),
                    ("measurement", measurement),
                ],
            );
        }
    }
}

pub(super) fn record_request_tokens(
    metrics: Option<&dyn ExtensionMetrics>,
    existing: usize,
    total: usize,
) {
    let Some(metrics) = metrics else {
        return;
    };
    for (component, tokens) in [
        ("existing_context", existing),
        ("new_input", total.saturating_sub(existing)),
        ("total", total),
    ] {
        let mut tags = vec![("target", "async"), ("component", component)];
        // Histogram Count grouped by token_bucket counts only admitted complete inputs.
        if component == "total" {
            tags.push((
                "token_bucket",
                codex_guardian_context::request_token_bucket(total),
            ));
        }
        metrics.histogram_with_boundaries(
            codex_guardian_context::REQUEST_TOKENS_METRIC,
            i64::try_from(tokens).unwrap_or(i64::MAX),
            codex_guardian_context::REQUEST_TOKENS_BOUNDARIES,
            &tags,
        );
    }
}

pub(super) fn record_decisions_comparison_outcome(
    metrics: Option<&dyn ExtensionMetrics>,
    outcome: &str,
    reason: &str,
) {
    if let Some(metrics) = metrics {
        metrics.counter(
            "codex.guardian_v2.decisions_comparison",
            /*inc*/ 1,
            &[("outcome", outcome), ("reason", reason)],
        );
    }
}

// Keep all backend diagnostics bounded and free of raw transport data.
pub(super) fn decisions_failure_reason(error: DecisionsError) -> &'static str {
    match error {
        DecisionsError::Credentials => "credentials_unavailable",
        DecisionsError::ClientSetup => "request_build_error",
        DecisionsError::UnexpectedRequestSetup => "unexpected_request_setup",
        DecisionsError::UnsupportedInputItem => "unsupported_input_item",
        DecisionsError::UnsupportedImage => "image",
        DecisionsError::UnsupportedAudio => "audio",
        DecisionsError::InputTooLarge => "input_too_large",
        DecisionsError::Timeout => "transport_timeout",
        // The adapter erases transport details; do not claim a specific network failure.
        DecisionsError::Transport => "decisions_transport",
        DecisionsError::Http(status) => match status {
            401 => "http_401",
            403 => "http_403",
            429 => "http_429",
            400..=499 => "http_4xx",
            500..=599 => "http_5xx",
            _ => "http_other",
        },
        DecisionsError::ResponseTooLarge => "response_too_large",
        DecisionsError::InvalidResponse => "invalid_output",
    }
}

// Called after the authoritative baseline has been published.
pub(super) fn record_decisions_comparison(
    completed: Result<(Result<&'static str, DecisionsError>, Duration), tokio::task::JoinError>,
    responses_sample: Option<(&str, Duration)>,
    metrics: Option<&dyn ExtensionMetrics>,
) {
    let decisions_sample = match completed {
        Ok((result, duration)) => {
            let outcome = match &result {
                Ok(_) => "success",
                Err(
                    DecisionsError::UnsupportedInputItem
                    | DecisionsError::UnsupportedImage
                    | DecisionsError::UnsupportedAudio
                    | DecisionsError::InputTooLarge,
                ) => "skipped",
                Err(_) => "failure",
            };
            record_decisions_comparison_outcome(
                metrics,
                outcome,
                result
                    .as_ref()
                    .err()
                    .map_or("none", |error| decisions_failure_reason(*error)),
            );
            result.ok().map(|risk| (risk, duration))
        }
        Err(error) => {
            let (outcome, reason) = if error.is_cancelled() {
                ("skipped", "superseded")
            } else {
                ("failure", "task_error")
            };
            record_decisions_comparison_outcome(metrics, outcome, reason);
            None
        }
    };
    if let Some(metrics) = metrics {
        if let (Some((_, responses_duration)), Some((_, decisions_duration))) =
            (responses_sample, decisions_sample)
        {
            // Compare latency only for the same successfully classified requests.
            for (backend, duration) in [
                ("responses", responses_duration),
                ("decisions", decisions_duration),
            ] {
                metrics.histogram(
                    "codex.guardian_v2.decisions_comparison.duration_ms",
                    i64::try_from(duration.as_millis()).unwrap_or(i64::MAX),
                    &[("backend", backend), ("outcome", "success")],
                );
            }
        }
        let responses_risk = responses_sample.map(|(risk, _)| risk);
        let decisions_risk = decisions_sample.map(|(risk, _)| risk);
        let comparison = match (responses_risk, decisions_risk) {
            (Some(responses_risk), Some(decisions_risk)) if responses_risk == decisions_risk => {
                "agree"
            }
            (Some(_), Some(_)) => "disagree",
            _ => "unavailable",
        };
        metrics.counter(
            "codex.guardian_v2.decisions_comparison.comparison",
            /*inc*/ 1,
            &[
                ("comparison", comparison),
                ("responses", responses_risk.unwrap_or("unavailable")),
                ("decisions", decisions_risk.unwrap_or("unavailable")),
            ],
        );
    }
}

#[cfg(test)]
mod body_rejection_tests {
    use super::*;

    #[test]
    fn body_rejection_keeps_original_http_failure_classification() {
        for status in [
            http::StatusCode::BAD_REQUEST,
            http::StatusCode::PAYLOAD_TOO_LARGE,
        ] {
            let body = codex_api::EncodedJsonBody::encode(
                &serde_json::json!({"input":"synthetic private request"}),
            )
            .unwrap();
            let size = body.as_bytes().len() as u64;
            let rejection = codex_api::PreStreamBodyRejection::from_http_open(
                status,
                codex_api::BodyRejectionRequestKind::Generation,
                size,
                size - 1,
                body,
            )
            .unwrap();
            let error = ApiError::BodyRejected {
                source: Box::new(TransportError::Http {
                    status,
                    url: None,
                    headers: None,
                    body: Some("synthetic diagnostic".to_owned()),
                    retry_after: None,
                }),
                rejection: Box::new(rejection),
            };
            assert_eq!(
                sampler_failure_reason(&LunaSamplerError::Api(error)),
                "http_4xx"
            );
        }
    }
}
