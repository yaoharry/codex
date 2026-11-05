//! Model-step attempt provenance survives transport resets and auth retry loops.

use super::*;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

pub(super) struct BodyAttemptChain {
    clean_start: bool,
    http_calls: AtomicU64,
    non_http: AtomicBool,
    stream_opened: AtomicBool,
    completed: AtomicBool,
    recovery_attempted: AtomicBool,
    owner: Mutex<Option<Option<u64>>>,
    pub(super) redirected: Arc<AtomicBool>,
}

impl BodyAttemptChain {
    fn new(clean_start: bool, non_http: bool) -> Self {
        Self {
            clean_start,
            http_calls: AtomicU64::new(0),
            non_http: AtomicBool::new(non_http),
            stream_opened: AtomicBool::new(false),
            completed: AtomicBool::new(false),
            recovery_attempted: AtomicBool::new(false),
            owner: Mutex::new(None),
            redirected: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(super) fn enter_http(&self, owner: Option<u64>) -> bool {
        let first = self.http_calls.fetch_add(1, Ordering::SeqCst) == 0;
        let Ok(mut recorded) = self.owner.lock() else {
            return false;
        };
        let matching_owner = match *recorded {
            Some(recorded) => recorded == owner,
            None => {
                *recorded = Some(owner);
                true
            }
        };
        first
            && matching_owner
            && self.clean_start
            && !self.non_http.load(Ordering::SeqCst)
            && !self.stream_opened.load(Ordering::SeqCst)
            && !self.redirected.load(Ordering::SeqCst)
    }

    pub(super) fn opened(&self) {
        self.stream_opened.store(true, Ordering::SeqCst);
    }
}

#[derive(Default)]
pub(super) struct BodyRecoveryState {
    pub(super) chain: Option<Arc<BodyAttemptChain>>,
    unsettled_step: bool,
    before_step_non_http: bool,
}

impl BodyRecoveryState {
    fn begin_step(&mut self) {
        // Transport/auth resets never call this. Only the normal model-step lifecycle does.
        self.unsettled_step = self
            .chain
            .as_ref()
            .is_some_and(|chain| !chain.completed.load(Ordering::SeqCst));
        self.chain = Some(Arc::new(BodyAttemptChain::new(
            !self.unsettled_step,
            std::mem::take(&mut self.before_step_non_http),
        )));
    }

    pub(super) fn websocket_attempt(&mut self) {
        if let Some(chain) = &self.chain {
            chain.non_http.store(true, Ordering::SeqCst);
        } else {
            self.before_step_non_http = true;
        }
    }
}

impl ModelClientSession {
    pub(crate) fn supports_body_recovery(&self) -> bool {
        self.client.request_contributors.is_empty()
    }
    pub(crate) fn begin_body_recovery_step(&mut self) {
        self.body_recovery.begin_step();
    }

    pub(crate) fn complete_body_recovery_step(&self) {
        if let Some(chain) = &self.body_recovery.chain {
            chain.completed.store(true, Ordering::SeqCst);
        }
    }

    pub(crate) fn take_body_recovery_attempt(&self) -> Option<Option<u64>> {
        let chain = self.body_recovery.chain.as_ref()?;
        if !chain.clean_start
            || chain.non_http.load(Ordering::SeqCst)
            || chain.stream_opened.load(Ordering::SeqCst)
            || chain.http_calls.load(Ordering::SeqCst) != 1
            || chain.recovery_attempted.swap(true, Ordering::SeqCst)
        {
            return None;
        }
        *chain.owner.lock().ok()?
    }

    pub(crate) fn project_body_recovery_item(
        &self,
        item: &ResponseItem,
        model: &ModelInfo,
        metadata: &CodexResponsesMetadata,
    ) -> Result<Option<ResponseItem>> {
        let prompt = Prompt {
            input: vec![item.clone()],
            tools: Arc::default(),
            parallel_tool_calls: false,
            base_instructions: codex_protocol::models::BaseInstructions {
                text: String::new(),
                provenance: None,
            },
            output_schema: None,
            output_schema_strict: true,
            prompt_cache_ttl: None,
            cyber_access_program: None,
        };
        let mut request = self.client.build_responses_request(
            &prompt,
            model,
            None,
            model.default_reasoning_summary,
            None,
            metadata,
            false,
        )?;
        self.client
            .prepare_response_items_for_request(&mut request.input);
        Ok((request.input.len() == 1).then(|| request.input.remove(0)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn stream_body_recovery_request(
        &mut self,
        body: codex_api::EncodedJsonBody,
        model: &ModelInfo,
        telemetry: &SessionTelemetry,
        metadata: &CodexResponsesMetadata,
        owner: Option<u64>,
        kind: codex_protocol::error::BodyRejectionRequestKind,
        trace: &InferenceTraceContext,
    ) -> Result<ResponseStream> {
        if self.client.auth_owner_generation() != owner {
            return Err(CodexErr::TurnAborted);
        }
        let setup = self
            .client
            .current_client_setup(ClientRouting::Workspace)
            .await?;
        if setup.auth_owner_generation != owner || self.client.auth_owner_generation() != owner {
            return Err(CodexErr::TurnAborted);
        }
        let redirected = Arc::new(AtomicBool::new(false));
        let transport = setup.http_transport_with_redirect_tracking(
            "/responses",
            Arc::clone(&redirected),
            true,
        )?;
        let auth_context = AuthRequestTelemetryContext::new(
            setup.auth.as_ref().map(CodexAuth::auth_mode),
            setup.api_auth.as_ref(),
            setup.agent_identity_telemetry.clone(),
            PendingUnauthorizedRetry::default(),
        )
        .map_err(|error| self.client.state.provider.map_api_error(error))?;
        let (request_telemetry, sse_telemetry) = Self::build_streaming_telemetry(
            telemetry,
            auth_context,
            RequestRouteTelemetry::for_endpoint("/responses"),
            self.client.state.auth_env_telemetry.clone(),
        );
        let executor = ResponsesHttpExecutor::new(&self.client);
        let mut options = executor
            .build_responses_options(
                &self.turn_state,
                metadata,
                Compression::None,
                model.use_responses_lite,
            )
            .await;
        options.extra_headers.extend(
            self.client
                .responses_headers(setup.auth.as_ref(), &model.slug),
        );
        options.body_recovery_origin = Some(
            codex_api::BodyRecoveryOrigin::new(kind, true)
                .with_exact_encoded_request()
                .with_redirect_tracking(Arc::clone(&redirected)),
        );
        let attempt = trace.start_attempt();
        attempt.add_request_headers(&mut options.extra_headers);
        if let Ok(raw) = serde_json::from_slice::<&serde_json::value::RawValue>(body.as_bytes()) {
            attempt.record_started(&raw);
        }
        let mut provider = setup.api_provider;
        provider.retry.max_attempts = 0;
        let client = ApiResponsesClient::new(transport, provider, setup.api_auth)
            .with_telemetry(Some(request_telemetry), Some(sse_telemetry));
        let stream = match client.stream_encoded_request(body, options).await {
            Ok(stream) => stream,
            Err(error) => {
                let debug = extract_response_debug_context_from_api_error(&error);
                let error = self.client.state.provider.map_api_error(error);
                attempt.record_failed(&error, debug.request_id.as_deref(), &[]);
                return Err(error);
            }
        };
        if redirected.load(Ordering::SeqCst) || self.client.auth_owner_generation() != owner {
            return Err(CodexErr::TurnAborted);
        }
        Ok(map_response_stream(
            stream,
            telemetry.clone(),
            attempt,
            &self.client.state,
            Vec::new(),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_recovery_chain_retains_ws_auth_and_retry_ambiguity() {
        let mut state = BodyRecoveryState::default();
        state.begin_step();
        let chain = state.chain.as_ref().unwrap();
        assert!(chain.enter_http(Some(1)));
        assert!(
            !chain.enter_http(Some(1)),
            "auth loop or outer retry is not first HTTP"
        );
        state.websocket_attempt();
        assert!(!state.chain.as_ref().unwrap().enter_http(Some(1)));
        state.begin_step();
        assert!(
            !state.chain.as_ref().unwrap().enter_http(Some(1)),
            "unsettled prior step survives reset"
        );
    }

    #[test]
    fn body_recovery_chain_new_budget_only_after_completed_step() {
        let mut state = BodyRecoveryState::default();
        state.begin_step();
        assert!(state.chain.as_ref().unwrap().enter_http(Some(1)));
        state
            .chain
            .as_ref()
            .unwrap()
            .completed
            .store(true, Ordering::SeqCst);
        state.begin_step();
        assert!(
            state.chain.as_ref().unwrap().enter_http(Some(1)),
            "known completed prior steps are distinct"
        );
        state.chain.as_ref().unwrap().opened();
        assert!(!state.chain.as_ref().unwrap().enter_http(Some(2)));
    }

    #[test]
    fn body_recovery_chain_cold_ws_failure_cannot_be_hidden_by_http_fallback() {
        let mut state = BodyRecoveryState::default();
        state.websocket_attempt();
        state.begin_step();
        assert!(!state.chain.as_ref().unwrap().enter_http(None));
    }

    #[test]
    fn body_recovery_chain_ambiguity_stays_with_failed_step_until_known_completion() {
        let mut state = BodyRecoveryState::default();
        state.begin_step();
        state.websocket_attempt();
        let chain = state.chain.as_ref().unwrap();
        assert!(!chain.enter_http(Some(1)));
        assert!(
            !chain.enter_http(Some(1)),
            "same-step fallback/auth counters cannot reset authority"
        );
        chain.completed.store(true, Ordering::SeqCst);
        state.begin_step();
        assert!(
            state.chain.as_ref().unwrap().enter_http(Some(1)),
            "a known completed step permits a fresh independent step"
        );
    }
}
