use crate::auth::SharedAuthProvider;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::telemetry::run_with_request_telemetry;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::Request;
use codex_client::RequestBody;
use codex_client::RequestTelemetry;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use http::HeaderMap;
use http::Method;
use serde_json::Value;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use tracing::instrument;

pub(crate) struct EndpointSession<T: HttpTransport> {
    transport: T,
    provider: Provider,
    auth: SharedAuthProvider,
    request_telemetry: Option<Arc<dyn RequestTelemetry>>,
}

impl<T: HttpTransport> EndpointSession<T> {
    pub(crate) fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            transport,
            provider,
            auth,
            request_telemetry: None,
        }
    }

    pub(crate) fn with_request_telemetry(
        mut self,
        request: Option<Arc<dyn RequestTelemetry>>,
    ) -> Self {
        self.request_telemetry = request;
        self
    }

    pub(crate) fn provider(&self) -> &Provider {
        &self.provider
    }

    fn make_request(
        &self,
        method: &Method,
        path: &str,
        extra_headers: &HeaderMap,
        body: Option<&RequestBody>,
    ) -> Request {
        let mut req = self.provider.build_request(method.clone(), path);
        req.headers.extend(extra_headers.clone());
        if let Some(body) = body {
            req.body = Some(body.clone());
        }
        req
    }

    pub(crate) async fn execute(
        &self,
        method: Method,
        path: &str,
        extra_headers: HeaderMap,
        body: Option<Value>,
    ) -> Result<Response, ApiError> {
        self.execute_with(method, path, extra_headers, body, |_| {})
            .await
    }

    #[instrument(
        name = "endpoint_session.execute_with",
        level = "info",
        skip_all,
        fields(http.method = %method, api.path = path)
    )]
    pub(crate) async fn execute_with<C>(
        &self,
        method: Method,
        path: &str,
        extra_headers: HeaderMap,
        body: Option<Value>,
        configure: C,
    ) -> Result<Response, ApiError>
    where
        C: Fn(&mut Request),
    {
        let body = body.map(RequestBody::Json);
        let make_request = || {
            let mut req = self.make_request(&method, path, &extra_headers, body.as_ref());
            configure(&mut req);
            req
        };

        let response = run_with_request_telemetry(
            self.provider.retry.to_policy(),
            self.request_telemetry.clone(),
            make_request,
            |req| {
                let auth = self.auth.clone();
                let transport = &self.transport;
                async move {
                    let req = auth.apply_auth(req).await.map_err(TransportError::from)?;
                    transport.execute(req).await
                }
            },
        )
        .await?;

        Ok(response)
    }

    #[instrument(
        name = "endpoint_session.stream_encoded_json_with",
        level = "info",
        skip_all,
        fields(http.method = %method, api.path = path)
    )]
    pub(crate) async fn stream_encoded_json_with<C>(
        &self,
        method: Method,
        path: &str,
        extra_headers: HeaderMap,
        body: Option<EncodedJsonBody>,
        body_recovery_origin: Option<crate::BodyRecoveryOrigin>,
        configure: C,
    ) -> Result<StreamResponse, ApiError>
    where
        C: Fn(&mut Request),
    {
        let body = body.map(RequestBody::EncodedJson);
        let mut request = self.make_request(&method, path, &extra_headers, body.as_ref());
        configure(&mut request);
        let request = request.into_prepared().map_err(TransportError::Build)?;
        let exact_request = if body_recovery_origin
            .as_ref()
            .is_some_and(|origin| origin.require_exact_encoded_request)
        {
            let Some(RequestBody::EncodedJson(body)) = &request.body else {
                return Err(TransportError::Build(
                    "owned recovery request is not encoded JSON".to_owned(),
                )
                .into());
            };
            if request.headers.contains_key(http::header::CONTENT_ENCODING) {
                return Err(TransportError::Build(
                    "owned recovery request is compressed".to_owned(),
                )
                .into());
            }
            Some((request.method.clone(), request.url.clone(), body.clone()))
        } else {
            None
        };
        let attempts = AtomicU64::new(0);
        let observed = Mutex::new(None);
        let capture_body = body_recovery_origin.is_some();
        let make_request = || {
            attempts.fetch_add(1, Ordering::Relaxed);
            request.clone()
        };

        let stream = run_with_request_telemetry(
            self.provider.retry.to_policy(),
            self.request_telemetry.clone(),
            make_request,
            |req| {
                let auth = self.auth.clone();
                let transport = &self.transport;
                let observed = &observed;
                let exact_request = &exact_request;
                async move {
                    let req = auth.apply_auth(req).await.map_err(TransportError::from)?;
                    let req = if capture_body {
                        if exact_request.is_some()
                            && (req.compression != codex_client::RequestCompression::None
                                || req.headers.contains_key(http::header::CONTENT_ENCODING)
                                || !matches!(&req.body, Some(RequestBody::EncodedJson(_))))
                        {
                            return Err(TransportError::Build("owned recovery request changed during authentication".to_owned()));
                        }
                        let req = req.into_prepared().map_err(TransportError::Build)?;
                        if let Some((method, url, body)) = exact_request {
                            let same_body = matches!(&req.body, Some(RequestBody::EncodedJson(prepared)) if prepared.as_bytes() == body.as_bytes());
                            if req.method != *method || req.url != *url || !same_body
                                || req.headers.contains_key(http::header::CONTENT_ENCODING)
                            {
                                return Err(TransportError::Build("owned recovery request changed during authentication".to_owned()));
                            }
                        }
                        let capture = if req.headers.contains_key(http::header::CONTENT_ENCODING) {
                            None
                        } else if let Some(RequestBody::EncodedJson(body)) = &req.body {
                            Some((req.url.clone(), body.clone()))
                        } else {
                            None
                        };
                        if let Ok(mut observed) = observed.lock() {
                            *observed = capture;
                        }
                        req
                    } else {
                        req
                    };
                    transport.stream(req).await
                }
            },
        )
        .await;

        match stream {
            Ok(stream) => Ok(stream),
            Err(source) => {
                let rejection = body_recovery_origin
                    .filter(|origin| {
                        origin.pristine_attempt_chain
                            && !origin
                                .redirect_observed
                                .as_ref()
                                .is_some_and(|flag| flag.load(Ordering::SeqCst))
                            && attempts.load(Ordering::Relaxed) == 1
                            && method == Method::POST
                            && path == "/responses"
                    })
                    .and_then(|origin| {
                        let TransportError::Http {
                            status,
                            url,
                            body: Some(error_body),
                            ..
                        } = &source
                        else {
                            return None;
                        };
                        let bytes = crate::body_rejection::parse_declared_body_bytes(error_body)?;
                        let observed = observed.lock().ok()?;
                        let (request_url, body) = observed.as_ref()?;
                        if url.as_ref().is_some_and(|url| url != request_url) {
                            return None;
                        }
                        codex_protocol::error::PreStreamBodyRejection::from_http_open(
                            *status,
                            origin.kind,
                            bytes.request,
                            bytes.limit,
                            body.clone(),
                        )
                    });
                match rejection {
                    Some(rejection) => Err(ApiError::BodyRejected {
                        source: Box::new(source),
                        rejection: Box::new(rejection),
                    }),
                    None => Err(ApiError::Transport(source)),
                }
            }
        }
    }
}
