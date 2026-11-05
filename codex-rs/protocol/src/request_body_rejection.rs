//! Opaque request ownership captured only by the Responses HTTP stream-open boundary.

use codex_http_client::EncodedJsonBody;
use http::StatusCode;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyRejectionRequestKind {
    Generation,
    PrefixCompaction,
}

/// The encoded body is deliberately absent from Debug, Display and protocol serialization.
#[derive(Clone)]
pub struct PreStreamBodyRejection {
    status: StatusCode,
    kind: BodyRejectionRequestKind,
    body: EncodedJsonBody,
    limit_bytes: u64,
}

impl PreStreamBodyRejection {
    /// Only the owning HTTP-open path may attach this after validating the entire attempt chain.
    /// The byte contract applies to uncompressed encoded JSON; other byte domains must refuse.
    pub fn from_http_open(
        status: StatusCode,
        kind: BodyRejectionRequestKind,
        declared_request_bytes: u64,
        declared_limit_bytes: u64,
        uncompressed_body: EncodedJsonBody,
    ) -> Option<Self> {
        let encoded_bytes = u64::try_from(uncompressed_body.as_bytes().len()).ok()?;
        if !matches!(
            status,
            StatusCode::BAD_REQUEST | StatusCode::PAYLOAD_TOO_LARGE
        ) || declared_limit_bytes == 0
            || declared_request_bytes <= declared_limit_bytes
            || declared_request_bytes != encoded_bytes
        {
            return None;
        }
        Some(Self {
            status,
            kind,
            body: uncompressed_body,
            limit_bytes: declared_limit_bytes,
        })
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }
    pub fn request_kind(&self) -> BodyRejectionRequestKind {
        self.kind
    }
    pub fn encoded_body(&self) -> &EncodedJsonBody {
        &self.body
    }
    pub fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }
}

impl fmt::Debug for PreStreamBodyRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreStreamBodyRejection")
            .field("status", &self.status)
            .field("kind", &self.kind)
            .field("encoded_bytes", &self.body.as_bytes().len())
            .field("limit_bytes", &self.limit_bytes)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CodexErr;
    use crate::error::CodexErrorDetails;
    use pretty_assertions::assert_eq;

    #[test]
    fn body_rejection_binds_encoded_bytes_and_refuses_invalid_budget() {
        let body =
            EncodedJsonBody::encode(&serde_json::json!({"input":"synthetic private sentinel"}))
                .unwrap();
        let actual = body.as_bytes().len() as u64;
        for status in [StatusCode::BAD_REQUEST, StatusCode::PAYLOAD_TOO_LARGE] {
            let rejection = PreStreamBodyRejection::from_http_open(
                status,
                BodyRejectionRequestKind::Generation,
                actual,
                actual - 1,
                body.clone(),
            )
            .unwrap();
            assert_eq!(rejection.status(), status);
            assert_eq!(
                rejection.request_kind(),
                BodyRejectionRequestKind::Generation
            );
            assert_eq!(rejection.encoded_body().as_bytes(), body.as_bytes());
            assert_eq!(rejection.limit_bytes(), actual - 1);
        }
        for (status, request, limit) in [
            (StatusCode::OK, actual, actual - 1),
            (StatusCode::TOO_MANY_REQUESTS, actual, actual - 1),
            (StatusCode::BAD_REQUEST, actual, 0),
            (StatusCode::BAD_REQUEST, actual, actual),
            (StatusCode::BAD_REQUEST, actual, actual + 1),
            (StatusCode::BAD_REQUEST, actual + 1, actual - 1),
            (StatusCode::BAD_REQUEST, u64::MAX, actual - 1),
        ] {
            assert!(
                PreStreamBodyRejection::from_http_open(
                    status,
                    BodyRejectionRequestKind::Generation,
                    request,
                    limit,
                    body.clone()
                )
                .is_none()
            );
        }
    }

    #[test]
    fn body_rejection_is_redacted_and_preserves_error_category_without_unchanged_retry() {
        let body =
            EncodedJsonBody::encode(&serde_json::json!({"input":"never-log-this-synthetic-input"}))
                .unwrap();
        let size = body.as_bytes().len() as u64;
        let rejection = PreStreamBodyRejection::from_http_open(
            StatusCode::BAD_REQUEST,
            BodyRejectionRequestKind::Generation,
            size,
            size - 1,
            body,
        )
        .unwrap();
        assert!(!format!("{rejection:?}").contains("never-log"));
        let error = CodexErr::InvalidRequest("public body rejection".to_owned())
            .with_pre_stream_body_rejection(rejection);
        assert!(matches!(
            error.details(),
            CodexErrorDetails::InvalidRequest(_)
        ));
        assert!(error.pre_stream_body_rejection().is_some());
        assert!(error.retry_delay(1).is_none());
        assert!(!format!("{error:?} {error}").contains("never-log"));
        assert_eq!(
            serde_json::to_string(&error.to_codex_protocol_error())
                .unwrap()
                .contains("never-log"),
            false
        );
    }
}
