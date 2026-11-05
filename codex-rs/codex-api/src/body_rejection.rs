//! Parses declared bytes; only the owning HTTP stream-open path can authorize recovery.

use codex_protocol::error::BodyRejectionRequestKind;
use serde::Deserialize;

/// Caller-owned attempt provenance; parsing an error message never creates this value.
#[derive(Clone, Debug)]
pub struct BodyRecoveryOrigin {
    pub(crate) kind: BodyRejectionRequestKind,
    pub(crate) pristine_attempt_chain: bool,
    pub(crate) redirect_observed: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub(crate) require_exact_encoded_request: bool,
}

impl BodyRecoveryOrigin {
    pub fn new(kind: BodyRejectionRequestKind, pristine_attempt_chain: bool) -> Self {
        Self {
            kind,
            pristine_attempt_chain,
            redirect_observed: None,
            require_exact_encoded_request: false,
        }
    }

    pub fn with_redirect_tracking(
        mut self,
        observed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        self.redirect_observed = Some(observed);
        self
    }

    /// Recovery sends must preserve the already validated URL and uncompressed body after auth.
    pub fn with_exact_encoded_request(mut self) -> Self {
        self.require_exact_encoded_request = true;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeclaredBodyBytes {
    pub(crate) request: u64,
    pub(crate) limit: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FlatBodyError {
    error_code: String,
    message: String,
}

pub(crate) fn parse_declared_body_bytes(raw: &str) -> Option<DeclaredBodyBytes> {
    let error: FlatBodyError = serde_json::from_str(raw).ok()?;
    if error.error_code != "BAD_REQUEST" {
        return None;
    }
    let mut remaining = error.message.as_str();
    for word in [
        "Server", "received", "a", "request", "which", "exceeds", "maximum", "allowed", "content",
        "length.",
    ] {
        remaining = required_ascii_space(remaining.strip_prefix(word)?)?;
    }
    remaining = remaining.strip_prefix("RequestSize(bytes):")?;
    remaining = skip_ascii_space(remaining);
    let (request, rest) = decimal(remaining)?;
    remaining = required_ascii_space(rest.strip_prefix(',')?)?;
    remaining = remaining.strip_prefix("Limit(bytes):")?;
    let (limit, rest) = decimal(skip_ascii_space(remaining))?;
    if !rest.is_empty() || limit == 0 || request <= limit {
        return None;
    }
    Some(DeclaredBodyBytes { request, limit })
}

fn skip_ascii_space(value: &str) -> &str {
    let count = value
        .bytes()
        .take_while(|byte| matches!(byte, b'\t'..=b'\r' | b' '))
        .count();
    &value[count..]
}

fn required_ascii_space(value: &str) -> Option<&str> {
    let rest = skip_ascii_space(value);
    (rest.len() < value.len()).then_some(rest)
}

fn decimal(value: &str) -> Option<(u64, &str)> {
    let count = value.bytes().take_while(u8::is_ascii_digit).count();
    if count == 0 {
        return None;
    }
    Some((value[..count].parse().ok()?, &value[count..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    const MESSAGE: &str = "Server received a request which exceeds maximum allowed content length. RequestSize(bytes): 65, Limit(bytes): 64";

    #[test]
    fn body_bytes_exact_flat_contract_and_ascii_whitespace() {
        for message in [
            MESSAGE.to_owned(),
            MESSAGE.replace(": ", ":"),
            MESSAGE.replace("65", "067").replace("64", "066"),
        ] {
            let raw = json!({"error_code":"BAD_REQUEST", "message":message}).to_string();
            assert!(parse_declared_body_bytes(&raw).is_some());
        }
        for separator in ["\t", "\n", "\u{b}", "\u{c}", "\r", " ", "\r\n\t "] {
            let raw =
                json!({"error_code":"BAD_REQUEST", "message":MESSAGE.replace(' ', separator)})
                    .to_string();
            assert_eq!(
                parse_declared_body_bytes(&raw),
                Some(DeclaredBodyBytes {
                    request: 65,
                    limit: 64
                })
            );
        }
        let quoted = serde_json::to_string(MESSAGE).unwrap();
        let reordered = format!(r#"{{"message":{quoted},"error_code":"BAD_REQUEST"}}"#);
        assert_eq!(
            parse_declared_body_bytes(&reordered),
            Some(DeclaredBodyBytes {
                request: 65,
                limit: 64
            })
        );
    }

    #[test]
    fn body_bytes_rejects_other_envelopes_and_duplicate_or_extra_keys() {
        let quoted = serde_json::to_string(MESSAGE).unwrap();
        for raw in [
            String::new(),
            "not json".to_owned(),
            "[]".to_owned(),
            quoted.clone(),
            format!(r#"{{"error":{{"code":null,"message":{quoted}}}}}"#),
            format!(r#"{{"error_code":"BAD_REQUEST","message":{quoted},"param":null}}"#),
            format!(r#"{{"error_code":"BAD_REQUEST","message":{quoted},"message":{quoted}}}"#),
            format!(
                r#"{{"error_code":"BAD_REQUEST","error_code":"BAD_REQUEST","message":{quoted}}}"#
            ),
            format!(r#"{{"error_code":"bad_request","message":{quoted}}}"#),
            format!(r#"{{"error_code":400,"message":{quoted}}}"#),
            format!(r#"{{"error_code":null,"message":{quoted}}}"#),
            r#"{"error_code":"BAD_REQUEST","message":null}"#.to_owned(),
            r#"{"error_code":"BAD_REQUEST","message":[]}"#.to_owned(),
            format!(r#"{{"error_code":"BAD_REQUEST","input":{quoted}}}"#),
            format!(
                r#"{{"error_code":"BAD_REQUEST","message":{quoted},"input[3].arguments":"large"}}"#
            ),
        ] {
            assert_eq!(parse_declared_body_bytes(&raw), None, "{raw}");
        }
    }

    #[test]
    fn body_bytes_rejects_partial_messages_fields_and_ambiguous_units() {
        for message in [
            format!(" {MESSAGE}"), format!("{MESSAGE} "), format!("{MESSAGE}."),
            format!("prefix {MESSAGE}"), format!("{MESSAGE} suffix"), format!("\"{MESSAGE}\""),
            format!("{MESSAGE}\n{MESSAGE}"),
            MESSAGE.replace("65,", "65"), MESSAGE.replace("65,", "65 ,"), MESSAGE.replace("65,", "65,,"),
            MESSAGE.replace("Server", "server"), MESSAGE.replace(' ', "\u{a0}"), MESSAGE.replace(' ', "\u{2003}"),
            MESSAGE.replace("bytes", "tokens"), MESSAGE.replace("bytes", "chars"),
            MESSAGE.replace("bytes", "bits"), MESSAGE.replace("bytes", "MB"), MESSAGE.replace("bytes", "MiB"),
            MESSAGE.replace("Limit(bytes)", "Limit(tokens)"),
            "Request exceeds maximum allowed content length. RequestSize(bytes): 65 Limit(bytes): 64".to_owned(),
            "String above maximum length: input[3].arguments".to_owned(),
            "The image exceeds the maximum size of 64 bytes".to_owned(),
            "The audio exceeds the maximum size of 64 bytes".to_owned(),
        ] {
            let raw = json!({"error_code":"BAD_REQUEST", "message":message}).to_string();
            assert_eq!(parse_declared_body_bytes(&raw), None, "{message}");
        }
    }

    #[test]
    fn body_bytes_checked_unsigned_decimal_and_distinct_limits() {
        for (request, limit, expected) in [
            (
                "65",
                "64",
                Some(DeclaredBodyBytes {
                    request: 65,
                    limit: 64,
                }),
            ),
            (
                "67",
                "66",
                Some(DeclaredBodyBytes {
                    request: 67,
                    limit: 66,
                }),
            ),
            (
                "18446744073709551615",
                "64",
                Some(DeclaredBodyBytes {
                    request: u64::MAX,
                    limit: 64,
                }),
            ),
            ("65", "0", None),
            ("64", "64", None),
            ("63", "64", None),
            ("", "64", None),
            ("65", "", None),
            ("-65", "64", None),
            ("+65", "64", None),
            ("65.0", "64", None),
            ("6.5e1", "64", None),
            ("0x41", "64", None),
            ("6_5", "64", None),
            ("6,5", "64", None),
            ("６５", "64", None),
            ("18446744073709551616", "64", None),
            ("65", "18446744073709551616", None),
            ("65", "-64", None),
            ("65", "+64", None),
            ("65", "64.0", None),
        ] {
            let message = format!(
                "Server received a request which exceeds maximum allowed content length. RequestSize(bytes): {request}, Limit(bytes): {limit}"
            );
            let raw = json!({"error_code":"BAD_REQUEST", "message":message}).to_string();
            assert_eq!(
                parse_declared_body_bytes(&raw),
                expected,
                "{request}/{limit}"
            );
        }
    }
}
