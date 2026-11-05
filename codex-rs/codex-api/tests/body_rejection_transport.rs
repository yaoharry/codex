//! A real redirect inside one transport call cannot prove a pristine HTTP-open rejection.

use codex_api::AuthError;
use codex_api::AuthProvider;
use codex_api::BodyRecoveryOrigin;
use codex_api::BodyRejectionRequestKind;
use codex_api::Compression;
use codex_api::EncodedJsonBody;
use codex_api::Provider;
use codex_api::ReqwestTransport;
use codex_api::ResponsesClient;
use codex_http_client::ClientRouteClass;
use codex_http_client::HttpClientBuilder;
use codex_http_client::HttpClientFactory;
use codex_http_client::OutboundProxyPolicy;
use http::HeaderMap;
use http::StatusCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

struct NoAuth;
impl AuthProvider for NoAuth {
    fn add_auth_headers(&self, _: &mut HeaderMap) -> Result<(), AuthError> {
        Ok(())
    }
}

struct BodyLimit;
impl Respond for BodyLimit {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let size = request.body.len();
        ResponseTemplate::new(413).set_body_json(serde_json::json!({
            "error_code":"BAD_REQUEST", "message":format!(
                "Server received a request which exceeds maximum allowed content length. RequestSize(bytes): {size}, Limit(bytes): {}", size - 1
            )
        }))
    }
}

#[tokio::test]
async fn redirected_http_body_rejection_does_not_authorize_recovery() -> anyhow::Result<()> {
    for status in [307, 308] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(status).insert_header("location", "/redirected"))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/redirected"))
            .respond_with(BodyLimit)
            .expect(1)
            .mount(&server)
            .await;
        let observed = Arc::new(AtomicBool::new(false));
        let factory = HttpClientFactory::new(OutboundProxyPolicy::ReqwestDefault);
        let http = HttpClientBuilder::new()
            .with_redirect_tracking(Arc::clone(&observed))
            .build_respecting_outbound_proxy_policy(
                &factory,
                &format!("{}/responses", server.uri()),
                ClientRouteClass::Api,
            )?;
        let provider = Provider {
            name: "synthetic".to_owned(),
            base_url: server.uri(),
            query_params: None,
            headers: HeaderMap::new(),
            retry: codex_api::RetryConfig {
                max_attempts: 0,
                base_delay: Duration::from_millis(1),
                retry_429: false,
                retry_5xx: false,
                retry_transport: false,
            },
            stream_idle_timeout: Duration::from_secs(2),
        };
        let client = ResponsesClient::new(
            ReqwestTransport::from_http_client(http),
            provider,
            Arc::new(NoAuth),
        );
        let body = EncodedJsonBody::encode(
            &serde_json::json!({"model":"synthetic", "input":"complete synthetic input"}),
        )?;
        let error = client
            .stream_encoded(
                body,
                HeaderMap::new(),
                Compression::None,
                None,
                Some(
                    BodyRecoveryOrigin::new(BodyRejectionRequestKind::Generation, true)
                        .with_redirect_tracking(Arc::clone(&observed)),
                ),
            )
            .await
            .err()
            .expect("redirected rejection");
        let error = codex_api::map_api_error(error);
        assert!(observed.load(Ordering::SeqCst));
        assert!(error.pre_stream_body_rejection().is_none());
        assert!(
            matches!(error.details(), codex_protocol::error::CodexErrorDetails::UnexpectedStatus(error) if error.status == StatusCode::PAYLOAD_TOO_LARGE)
        );
        server.verify().await;
    }
    Ok(())
}
