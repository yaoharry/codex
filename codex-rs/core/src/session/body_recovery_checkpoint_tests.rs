//! New strict recovery requirements, separate from stock in-memory adoption semantics.

use super::*;
use codex_thread_store::*;
use pretty_assertions::assert_eq;
use std::any::Any;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

#[test]
fn body_recovery_current_context_requires_host_source_and_exact_retention() {
    use super::super::preserving_compaction::HistoricalUserPolicy;
    use super::super::preserving_compaction::is_retained_current_context;
    use super::super::preserving_compaction::retain_for_policy;
    use codex_protocol::models::ContentItemKind;
    use codex_protocol::models::ContentItemMetadata;
    use codex_protocol::models::ContentItemProvenance;
    use codex_protocol::models::InternalChatMessageMetadataPassthrough;
    let classified = |role: &str, kind: &str, source| {
        let mut item = user_message("Markers in text never establish ownership.");
        let ResponseItem::Message {
            role: actual_role,
            internal_chat_message_metadata_passthrough,
            ..
        } = &mut item
        else {
            unreachable!()
        };
        *actual_role = role.to_owned();
        *internal_chat_message_metadata_passthrough =
            Some(InternalChatMessageMetadataPassthrough {
                turn_id: Some("current-turn".to_owned()),
                content_item_kinds: Some(vec![ContentItemKind(kind.to_owned())]),
                content_item_metadata: Some(vec![ContentItemMetadata { provenance: source }]),
                ..Default::default()
            });
        ResponseItemEnvelope::new(item)
    };
    for (role, kind, source, allowed) in [
        ("developer", "unknown", ContentItemProvenance::Unknown, true),
        ("system", "unknown", ContentItemProvenance::Unknown, true),
        (
            "user",
            "environments.environment_context",
            ContentItemProvenance::Harness { hook: false },
            true,
        ),
        (
            "user",
            "agents_md.instructions",
            ContentItemProvenance::AgentsMd,
            true,
        ),
        (
            "user",
            "user.text",
            ContentItemProvenance::Harness { hook: false },
            false,
        ),
        (
            "user",
            "user.goal",
            ContentItemProvenance::Harness { hook: false },
            false,
        ),
        (
            "user",
            "images.unsupported",
            ContentItemProvenance::Harness { hook: false },
            false,
        ),
        (
            "user",
            "audio.unsupported",
            ContentItemProvenance::Harness { hook: false },
            false,
        ),
        (
            "user",
            "unknown",
            ContentItemProvenance::Harness { hook: false },
            false,
        ),
        (
            "user",
            "unrecognized.context",
            ContentItemProvenance::Harness { hook: false },
            false,
        ),
        (
            "user",
            "environments.environment_context",
            ContentItemProvenance::AgentsMd,
            false,
        ),
        (
            "user",
            "agents_md.instructions",
            ContentItemProvenance::Harness { hook: false },
            false,
        ),
        (
            "user",
            "",
            ContentItemProvenance::Harness { hook: false },
            false,
        ),
        (
            "user",
            "environments.environment_context",
            ContentItemProvenance::Unknown,
            false,
        ),
        (
            "user",
            "environments.environment_context",
            ContentItemProvenance::User,
            false,
        ),
        (
            "user",
            "environments.environment_context",
            ContentItemProvenance::Tool { namespace: None },
            false,
        ),
        (
            "user",
            "environments.environment_context",
            ContentItemProvenance::Harness { hook: true },
            false,
        ),
        (
            "assistant",
            "environments.environment_context",
            ContentItemProvenance::Harness { hook: false },
            false,
        ),
    ] {
        let item = classified(role, kind, source);
        let original = item.clone();
        assert_eq!(is_retained_current_context(&item), allowed, "{role}:{kind}");
        if allowed {
            for policy in [
                HistoricalUserPolicy::Retain,
                HistoricalUserPolicy::SummarizeKnownSampled,
            ] {
                assert!(retain_for_policy(&item, policy, u64::MAX));
            }
        }
        assert_eq!(item, original);
    }
    for change in [
        "missing_kinds",
        "missing_source",
        "partial_source",
        "partial_kinds",
        "media",
        "empty_content",
        "encrypted",
        "mixed_user",
        "mixed_source",
    ] {
        let mut item = classified(
            "user",
            "environments.environment_context",
            ContentItemProvenance::Harness { hook: false },
        );
        let ResponseItem::Message {
            content,
            encrypted_content,
            internal_chat_message_metadata_passthrough: Some(metadata),
            ..
        } = &mut item.item
        else {
            unreachable!()
        };
        match change {
            "missing_kinds" => metadata.content_item_kinds = None,
            "missing_source" => metadata.content_item_metadata = None,
            "partial_source" => metadata.content_item_metadata = Some(Vec::new()),
            "partial_kinds" => metadata.content_item_kinds = Some(Vec::new()),
            "media" => content[0] = serde_json::from_value(serde_json::json!({"type":"input_image", "image_url":"data:image/png;base64,synthetic"})).expect("synthetic image"),
            "empty_content" => content.clear(),
            "encrypted" => *encrypted_content = Some("synthetic opaque input".to_owned()),
            "mixed_user" | "mixed_source" => {
                content.push(ContentItem::InputText { text: "Fresh authorization stays protected.".to_owned() });
                metadata.content_item_kinds.as_mut().expect("kinds").push(ContentItemKind(if change == "mixed_user" { "user.text" } else { "environments.environment_context" }.to_owned()));
                metadata.content_item_metadata.as_mut().expect("source").push(ContentItemMetadata { provenance: if change == "mixed_user" { ContentItemProvenance::User } else { ContentItemProvenance::Unknown } });
            }
            _ => unreachable!(),
        }
        assert!(!is_retained_current_context(&item), "{change}");
    }
}

enum CheckpointFault {
    Allow,
    Reject,
    RejectAfterCanonical,
    BeforeQueue(mpsc::UnboundedSender<PendingAppend>),
    AfterCanonical(mpsc::UnboundedSender<PendingAppend>),
}

struct PendingAppend {
    release: oneshot::Sender<()>,
    completed: oneshot::Receiver<ThreadStoreResult<()>>,
}

struct CheckpointStore {
    inner: Arc<dyn ThreadStore>,
    fault: CheckpointFault,
    attempts: AtomicUsize,
}

macro_rules! delegate_store_methods {
    ($(fn $name:ident($param:ident: $params:ty) -> $result:ty;)*) => {
        $(fn $name(&self, $param: $params) -> ThreadStoreFuture<'_, $result> {
            self.inner.$name($param)
        })*
    };
}

impl ThreadStore for CheckpointStore {
    fn as_any(&self) -> &dyn Any {
        self
    }

    delegate_store_methods! {
        fn create_thread(params: CreateThreadParams) -> ();
        fn resume_thread(params: ResumeThreadParams) -> Arc<Vec<RolloutItem>>;
        fn discard_thread(thread_id: ThreadId) -> ();
        fn load_history(params: LoadThreadHistoryParams) -> StoredThreadHistory;
        fn load_latest_model_context(params: LoadThreadHistoryParams) -> StoredModelContext;
        fn read_thread(params: ReadThreadParams) -> StoredThread;
        fn read_thread_by_rollout_path(params: ReadThreadByRolloutPathParams) -> StoredThread;
        fn list_threads(params: ListThreadsParams) -> ThreadPage;
        fn update_thread_metadata(params: UpdateThreadMetadataParams) -> Option<StoredThread>;
        fn archive_thread(params: ArchiveThreadParams) -> ();
        fn unarchive_thread(params: ArchiveThreadParams) -> StoredThread;
        fn delete_thread(params: DeleteThreadParams) -> ();
        fn flush_thread(thread_id: ThreadId) -> ();
        fn shutdown_thread(thread_id: ThreadId) -> ();
    }

    fn persist_thread(
        &self,
        thread_id: ThreadId,
        context: PersistContext,
    ) -> ThreadStoreFuture<'_, ()> {
        self.inner.persist_thread(thread_id, context)
    }

    fn append_items(&self, params: AppendThreadItemsParams) -> ThreadStoreFuture<'_, ()> {
        Box::pin(async move {
            if !params
                .items
                .iter()
                .any(|item| matches!(item, RolloutItem::Compacted(_)))
            {
                return self.inner.append_items(params).await;
            }
            self.attempts.fetch_add(1, Ordering::SeqCst);
            match &self.fault {
                CheckpointFault::Allow => self.inner.append_items(params).await,
                CheckpointFault::Reject => Err(ThreadStoreError::Internal {
                    message: "synthetic checkpoint write failure".to_owned(),
                }),
                CheckpointFault::RejectAfterCanonical => {
                    self.inner.append_items(params).await?;
                    Err(ThreadStoreError::OutcomeUnknown {
                        message: "synthetic post-canonical acknowledgement failure".to_owned(),
                    })
                }
                CheckpointFault::BeforeQueue(pending)
                | CheckpointFault::AfterCanonical(pending) => {
                    let after_canonical = matches!(&self.fault, CheckpointFault::AfterCanonical(_));
                    let result = if after_canonical {
                        Some(self.inner.append_items(params.clone()).await)
                    } else {
                        None
                    };
                    let (release, released) = oneshot::channel();
                    let (complete, completed) = oneshot::channel();
                    pending
                        .send(PendingAppend { release, completed })
                        .expect("append observer");
                    released.await.expect("release append");
                    let result = match result {
                        Some(result) => result,
                        None => self.inner.append_items(params).await,
                    };
                    let _ = complete.send(result.clone());
                    result
                }
            }
        })
    }
}

async fn session_with_checkpoint_fault(fault: CheckpointFault) -> (Arc<Session>, Arc<TurnContext>) {
    session_with_checkpoint_history(
        fault,
        &[
            user_message("completed user input"),
            assistant_message("completed answer"),
        ],
    )
    .await
}

async fn session_with_checkpoint_history(
    fault: CheckpointFault,
    historical: &[ResponseItem],
) -> (Arc<Session>, Arc<TurnContext>) {
    let (mut session, turn, _events) = make_session_and_context_with_rx().await;
    let unique = Arc::get_mut(&mut session).expect("unique session");
    unique.services.thread_store = Arc::new(CheckpointStore {
        inner: Arc::clone(&unique.services.thread_store),
        fault,
        attempts: AtomicUsize::new(0),
    });
    initialize_thread_persistence(unique).await;
    session
        .ensure_rollout_materialized(PersistContext::Standard)
        .await;
    session
        .flush_rollout()
        .await
        .expect("flush initialized persistence");
    let older = session.new_default_turn().await;
    session
        .record_started_turn(&older.sub_id, Some(older.attribution()))
        .await;
    session
        .record_conversation_items(&older, older.model_info(), historical)
        .await;
    session
        .record_started_turn(&turn.sub_id, Some(turn.attribution()))
        .await;
    session
        .record_conversation_items(
            &turn,
            turn.model_info(),
            &[user_message("protected current input")],
        )
        .await;
    (session, turn)
}

async fn install_prefix(
    session: &Arc<Session>,
    turn: &Arc<TurnContext>,
    cancellation: &CancellationToken,
) -> bool {
    install_selected_prefix(session, turn, 2, 2, cancellation).await
}

async fn install_selected_prefix(
    session: &Arc<Session>,
    turn: &Arc<TurnContext>,
    selected: usize,
    protected_start: usize,
    cancellation: &CancellationToken,
) -> bool {
    let history = session.clone_history().await;
    let step = session
        .capture_step_context(Arc::clone(turn), cancellation)
        .await
        .expect("capture step");
    let mut replacement = history.annotated_items()[..selected]
        .iter()
        .filter(|item| super::super::preserving_compaction::retain_for_body_recovery(item))
        .cloned()
        .collect::<Vec<_>>();
    replacement.push(ResponseItemEnvelope::new(ResponseItem::Compaction {
        id: None,
        encrypted_content: "synthetic-completed-prefix-summary".to_owned(),
        internal_chat_message_metadata_passthrough: None,
    }));
    session
        .install_preserving_compaction_prefix(
            &step,
            &history,
            protected_start,
            CompactionPrefix {
                generation: history.history_version(),
                len: selected,
                auth_owner_generation: session.services.model_client.auth_owner_generation(),
            },
            replacement,
            "synthetic-compaction".to_owned(),
            cancellation,
        )
        .await
}

#[test_case::test_case(false; "before_canonical")]
#[test_case::test_case(true; "after_canonical_unknown_ack")]
#[tokio::test]
async fn body_recovery_checkpoint_failure_preserves_active_history(after_canonical: bool) {
    let fault = if after_canonical {
        CheckpointFault::RejectAfterCanonical
    } else {
        CheckpointFault::Reject
    };
    let (session, turn) = session_with_checkpoint_fault(fault).await;
    let original = session.clone_history().await;
    let window = session.current_window_id().await;
    let installed = install_prefix(&session, &turn, &CancellationToken::new()).await;
    assert_eq!(
        checkpoint_attempts(&session),
        1,
        "exercise the failed acknowledgement"
    );
    let current = session.clone_history().await;
    assert!(
        !installed && current.annotated_items() == original.annotated_items(),
        "failed append reported installed={installed}, active history preserved={}",
        current.annotated_items() == original.annotated_items(),
    );
    assert_eq!(session.current_window_id().await, window);
    assert_original_records_kept(&session, &original).await;
    check_newer_input_on_resume(&session, &turn).await;
}

#[test_case::test_case(true; "cancelled_before_enqueue")]
#[test_case::test_case(false; "completed_append_precedes_new_input")]
#[tokio::test]
async fn body_recovery_checkpoint_preserves_newer_input_on_resume(cancel_before_append: bool) {
    let (pending, mut appends) = mpsc::unbounded_channel();
    let (session, turn) =
        session_with_checkpoint_fault(CheckpointFault::BeforeQueue(pending)).await;
    let cancellation = CancellationToken::new();
    let mut installation = Box::pin(install_prefix(&session, &turn, &cancellation));
    let append = tokio::select! {
        append = appends.recv() => append.expect("checkpoint append"),
        result = &mut installation => panic!("append unexpectedly completed: {result}"),
    };
    let PendingAppend { release, completed } = append;
    if cancel_before_append {
        drop(installation);
    } else {
        release.send(()).expect("release append");
        assert!(installation.await);
        completed
            .await
            .expect("append completion")
            .expect("durable append");
        return check_newer_input_on_resume(&session, &turn).await;
    }
    let newer = user_message("newer accepted input after cancellation");
    session
        .record_conversation_items(&turn, turn.model_info(), &[newer])
        .await;
    assert!(release.send(()).is_err(), "cancelled before native enqueue");
    assert!(completed.await.is_err(), "cancelled append did not commit");
    assert_newer_input_on_resume(&session, &turn).await;
}

async fn check_newer_input_on_resume(session: &Session, turn: &TurnContext) {
    session
        .record_conversation_items(
            turn,
            turn.model_info(),
            &[user_message("newer accepted input after cancellation")],
        )
        .await;
    assert_newer_input_on_resume(session, turn).await;
}

async fn assert_newer_input_on_resume(session: &Session, turn: &TurnContext) {
    let expected = session.clone_history().await;
    let newer = expected.annotated_items().last().expect("newer item");
    let stored = session
        .live_thread()
        .expect("live thread")
        .load_history(true)
        .await
        .expect("stored history");
    assert!(
        stored.items.iter().any(|item| {
            matches!(item, RolloutItem::ResponseItem(item) if item.item == newer.item)
        }),
        "newer accepted input must already be canonical"
    );
    let resumed = session
        .reconstruct_history_from_rollout(turn, &stored.items)
        .await;
    let protected = expected.annotated_items().iter().find(|item| matches!(&item.item,
        ResponseItem::Message { role, content, .. } if role == "user" && matches!(content.as_slice(),
            [ContentItem::InputText { text }] if text == "protected current input"
        )
    )).expect("protected current input");
    assert_eq!(
        resumed
            .history
            .iter()
            .filter(|item| item.item == protected.item)
            .count(),
        1,
        "the captured protected input remains exact and once"
    );
    assert_eq!(
        resumed
            .history
            .iter()
            .filter(|item| item.item == newer.item)
            .count(),
        1,
        "a cancelled old checkpoint must not hide a newer accepted input on resume",
    );
    let protected_position = resumed
        .history
        .iter()
        .position(|item| item.item == protected.item)
        .expect("protected input position");
    let newer_position = resumed
        .history
        .iter()
        .position(|item| item.item == newer.item)
        .expect("newer input position");
    assert!(
        protected_position < newer_position,
        "canonical replay preserves accepted input order"
    );
}

fn checkpoint_attempts(session: &Session) -> usize {
    session
        .services
        .thread_store
        .as_any()
        .downcast_ref::<CheckpointStore>()
        .expect("fault store")
        .attempts
        .load(Ordering::SeqCst)
}

async fn assert_original_records_kept(session: &Session, original: &ContextManager) {
    let stored = session
        .live_thread()
        .expect("live thread")
        .load_history(true)
        .await
        .expect("stored history");
    for original in original.annotated_items() {
        assert_eq!(
            stored
                .items
                .iter()
                .filter(|item| matches!(item,
                    RolloutItem::ResponseItem(item) if item.item == original.item
                ))
                .count(),
            1,
            "original canonical item remains exact and once"
        );
    }
}

#[tokio::test]
async fn body_recovery_checkpoint_stop_before_enqueue_does_not_wait_for_store() {
    let (pending, mut appends) = mpsc::unbounded_channel();
    let (session, turn) =
        session_with_checkpoint_fault(CheckpointFault::BeforeQueue(pending)).await;
    let original = session.clone_history().await;
    let cancellation = CancellationToken::new();
    let mut installation = Box::pin(install_prefix(&session, &turn, &cancellation));
    let append = tokio::select! {
        append = appends.recv() => append.expect("blocked append"),
        result = &mut installation => panic!("expected blocked store, got {result}"),
    };
    cancellation.cancel();
    assert!(!installation.await);
    assert!(
        append.release.send(()).is_err(),
        "Stop did not wait for store release"
    );
    assert!(append.completed.await.is_err());
    assert_eq!(
        session.clone_history().await.annotated_items(),
        original.annotated_items()
    );
    assert_original_records_kept(&session, &original).await;
    check_newer_input_on_resume(&session, &turn).await;
}

#[test_case::test_case(false; "before_queue")]
#[test_case::test_case(true; "after_canonical_unknown_ack")]
#[tokio::test]
async fn body_recovery_checkpoint_bounds_queued_steer_and_actual_interrupt(after_canonical: bool) {
    tokio::time::timeout(Duration::from_secs(15), async {
        let (pending, mut appends) = mpsc::unbounded_channel();
        let fault = if after_canonical {
            CheckpointFault::AfterCanonical(pending)
        } else {
            CheckpointFault::BeforeQueue(pending)
        };
        let (session, turn) = session_with_checkpoint_fault(fault).await;
        let submission = |id: &str, op| Submission {
            id: id.to_owned(),
            op,
            turn_extension_init: None,
            trace: None,
            parent_turn_id: None,
            root_turn_id: None,
            residency_guard: None,
            message_delivery: None,
        };
        let (tx_sub, rx_sub) = async_channel::unbounded();
        let mut submissions = Box::pin(tokio::task::unconstrained(submission_loop(
            Arc::clone(&session),
            Arc::clone(&turn.config),
            rx_sub,
        )));
        let (ready, ready_rx) = oneshot::channel();
        tx_sub
            .send(submission(
                "bootstrap",
                Op::ThreadSettings {
                    thread_settings: Default::default(),
                    reply: Some(ready),
                },
            ))
            .await
            .expect("queue bootstrap barrier");
        tokio::select! {
            result = ready_rx => { result.expect("bootstrap reply").expect("bootstrap settings"); },
            () = submissions.as_mut() => panic!("dispatcher exited before bootstrap"),
        }
        assert!(futures::poll!(submissions.as_mut()).is_pending());
        session
            .spawn_task(
                Arc::clone(&turn),
                Vec::new(),
                NeverEndingTask {
                    kind: TaskKind::Regular,
                    listen_to_cancellation_token: true,
                },
            )
            .await;
        let cancellation = {
            let active = session.active_turn.lock().await;
            active
                .as_ref()
                .and_then(|turn| turn.task.as_ref())
                .expect("registered native task")
                .cancellation_token
                .clone()
        };
        let original = session.clone_history().await;
        let mut installation = Box::pin(install_prefix(&session, &turn, &cancellation));
        let append = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::select! {
                append = appends.recv() => append.expect("blocked checkpoint ACK"),
                result = installation.as_mut() => panic!("checkpoint never reached the store: {result}"),
                () = submissions.as_mut() => panic!("dispatcher exited before checkpoint entered the store"),
            }
        }).await.expect("checkpoint enters the ACK barrier");

        tokio::time::pause();
        let (steer_reply, mut accepted) = oneshot::channel();
        tx_sub
            .send(submission(
                "queued-steer",
                Op::TurnInput {
                    request: Box::new(codex_protocol::turn_input::TurnInputRequest::user_input(
                        vec![UserInput::Text {
                            text: "queued steer remains owned by native admission".to_owned(),
                            text_elements: Vec::new(),
                        }],
                    )),
                    mode: codex_protocol::turn_input::TurnInputMode::Steer {
                        expected_turn_id: turn.sub_id.clone(),
                    },
                    reply: steer_reply,
                },
            ))
            .await
            .expect("queue steer before Stop");
        tx_sub
            .send(submission("queued-stop", Op::Interrupt))
            .await
            .expect("queue actual Interrupt");
        let (barrier, barrier_rx) = oneshot::channel();
        tx_sub
            .send(submission(
                "after-stop",
                Op::ThreadSettings {
                    thread_settings: Default::default(),
                    reply: Some(barrier),
                },
            ))
            .await
            .expect("queue native dispatch barrier");
        assert!(futures::poll!(submissions.as_mut()).is_pending());
        assert!(
            session.state.try_lock().is_err(),
            "checkpoint owns the source state"
        );
        assert!(
            session.active_turn.try_lock().is_err(),
            "steer waits for state while owning active turn"
        );
        assert!(
            !cancellation.is_cancelled(),
            "queued Interrupt has not overtaken the steer"
        );
        assert!(matches!(
            accepted.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        tokio::time::advance(Duration::from_secs(6)).await;
        tokio::time::resume();
        let controls = async {
            tokio::select! {
                result = barrier_rx => { result.expect("post-Stop reply").expect("post-Stop settings"); },
                () = submissions.as_mut() => panic!("dispatcher exited before Interrupt completed"),
            }
        };
        let (installed, ()) = tokio::time::timeout(Duration::from_secs(2), async {
            tokio::join!(installation, controls)
        })
        .await
        .expect("checkpoint I/O budget must release queued Interrupt without releasing the store");
        assert!(!installed, "an expired ACK grants no retry receipt");
        assert!(
            cancellation.is_cancelled(),
            "actual Interrupt cancelled the registered task"
        );
        assert!(matches!(
            accepted
                .await
                .expect("steer reply")
                .expect("steer admitted"),
            codex_protocol::turn_input::TurnInputSubmission::Steered { .. }
        ));
        let validation = async {
            assert!(
                append.release.send(()).is_err(),
                "no store release solved the control convoy"
            );
            assert!(append.completed.await.is_err());
            assert!(
                session
                    .clone_history()
                    .await
                    .annotated_items()
                    .starts_with(original.annotated_items())
            );
            assert_original_records_kept(&session, &original).await;
            let stored = session
                .live_thread()
                .expect("live thread")
                .load_history(true)
                .await
                .expect("canonical history");
            assert_eq!(
                stored
                    .items
                    .iter()
                    .filter(|item| matches!(item, RolloutItem::Compacted(_)))
                    .count(),
                usize::from(after_canonical)
            );
            check_newer_input_on_resume(&session, &turn).await;
            tx_sub
                .send(submission("finish", Op::Shutdown))
                .await
                .expect("stop dispatcher");
        };
        tokio::time::timeout(Duration::from_secs(2), async {
            tokio::join!(submissions, validation)
        }).await.expect("post-control persistence checks and dispatcher shutdown must finish together");
    }).await.expect("whole Stop control test watchdog");
}

#[test_case::test_case(false; "function_output_cut")]
#[test_case::test_case(true; "custom_output_cut")]
#[tokio::test]
async fn body_recovery_checkpoint_accepts_closed_output_without_final_assistant(custom: bool) {
    let call = if custom {
        json!({"type":"custom_tool_call", "call_id":"finished", "name":"owned_tool", "namespace":"owned", "input":"synthetic"})
    } else {
        json!({"type":"function_call", "call_id":"finished", "name":"owned_tool", "namespace":"owned", "arguments":"{\"synthetic\":true}"})
    };
    let output = if custom {
        json!({"type":"custom_tool_call_output", "call_id":"finished", "name":"owned_tool", "output":"complete synthetic result"})
    } else {
        json!({"type":"function_call_output", "call_id":"finished", "name":"owned_tool", "namespace":"owned", "output":"complete synthetic result"})
    };
    let historical = vec![
        user_message("completed user input"),
        serde_json::from_value(call).unwrap(),
        serde_json::from_value(output).unwrap(),
    ];
    let (session, turn) =
        session_with_checkpoint_history(CheckpointFault::Allow, &historical).await;
    let original = session.clone_history().await;
    let original_window = session.current_window_id().await;
    assert!(install_selected_prefix(&session, &turn, 3, 3, &CancellationToken::new()).await);
    assert_eq!(checkpoint_attempts(&session), 1);
    assert_ne!(session.current_window_id().await, original_window);
    let current = session.clone_history().await;
    assert_eq!(
        &current.annotated_items()[2..],
        &original.annotated_items()[3..],
        "entire protected suffix is unchanged"
    );
    assert_eq!(
        current.annotated_items()[0],
        original.annotated_items()[0],
        "selected user input is retained exactly"
    );
    assert_original_records_kept(&session, &original).await;
    check_newer_input_on_resume(&session, &turn).await;
}

#[tokio::test]
async fn body_recovery_checkpoint_preserves_complete_current_media_instructions_and_open_call() {
    let (session, turn) = session_with_checkpoint_fault(CheckpointFault::Allow).await;
    let current_items = [
        serde_json::from_value(json!({"type":"message", "role":"developer", "content":[
            {"type":"input_text", "text":"preserve complete current instructions"}
        ]})).unwrap(),
        serde_json::from_value(json!({"type":"message", "role":"user", "content":[
            {"type":"input_text", "text":"雪🧪\n".repeat(12000)},
            {"type":"input_image", "image_url":"https://example.invalid/synthetic.png"}
        ]})).unwrap(),
        serde_json::from_value(json!({"type":"function_call", "call_id":"live", "name":"owned_tool", "arguments":"{\"nested\":{\"value\":\"unchanged\"}}"})).unwrap(),
    ];
    session
        .record_conversation_items(&turn, turn.model_info(), &current_items)
        .await;
    let original = session.clone_history().await;
    assert!(install_prefix(&session, &turn, &CancellationToken::new()).await);
    let current = session.clone_history().await;
    assert_eq!(
        &current.annotated_items()[2..],
        &original.annotated_items()[2..]
    );
    assert_original_records_kept(&session, &original).await;
}

#[test_case::test_case(false; "local_durable_install")]
#[test_case::test_case(true; "local_failed_ack_after_canonical")]
#[tokio::test]
async fn body_recovery_checkpoint_installs_only_completed_local_summary(failed_ack: bool) {
    use super::super::body_recovery_summary::LocalSummaryAttempt;
    use super::super::body_recovery_summary::build_local_summary_candidate;
    use super::super::preserving_compaction::HistoricalUserPolicy;
    use super::super::preserving_compaction::PreservingSummary;
    let fault = if failed_ack {
        CheckpointFault::RejectAfterCanonical
    } else {
        CheckpointFault::Allow
    };
    let (session, turn) = session_with_checkpoint_fault(fault).await;
    let original = session.clone_history().await;
    let window_before = session.current_window_id().await;
    let cancellation = CancellationToken::new();
    let step = session
        .capture_step_context(Arc::clone(&turn), &cancellation)
        .await
        .expect("step");
    let output = [assistant_message("fresh completed local summary")];
    let local = build_local_summary_candidate(
        &turn.sub_id,
        LocalSummaryAttempt::Completed {
            response_id: "local-compaction-response",
            output: &output,
        },
    )
    .expect("completed summary");
    let expected_message = local.message().to_owned();
    let mut summary = PreservingSummary::local(local);
    summary.assign_id(&session);
    let expected_summary = summary.item().clone();
    let window = session
        .prepare_preserving_window(&original, &cancellation)
        .await
        .expect("prepared window");
    let retained = original.annotated_items()[..2]
        .iter()
        .filter(|item| super::super::preserving_compaction::retain_for_body_recovery(item))
        .cloned()
        .collect();
    let receipt = session
        .install_body_recovery_prefix(
            &step,
            &original,
            2,
            CompactionPrefix {
                generation: original.history_version(),
                len: 2,
                auth_owner_generation: session.services.model_client.auth_owner_generation(),
            },
            retained,
            &summary,
            Some(&window),
            HistoricalUserPolicy::Retain,
            &cancellation,
        )
        .await;
    assert_eq!(receipt.is_none(), failed_ack);
    assert_eq!(checkpoint_attempts(&session), 1);
    assert_original_records_kept(&session, &original).await;
    let stored = session
        .live_thread()
        .expect("thread")
        .load_history(true)
        .await
        .expect("history");
    let checkpoint = stored
        .items
        .iter()
        .find_map(|item| match item {
            RolloutItem::Compacted(item) => Some(item),
            _ => None,
        })
        .expect("canonical checkpoint");
    assert_eq!(checkpoint.message, expected_message);
    assert_eq!(
        checkpoint.compaction_response_id.as_deref(),
        Some("local-compaction-response")
    );
    assert!(!crate::context::is_user_authorization_message(
        &expected_summary.item
    ));
    if let Some(receipt) = receipt {
        assert_eq!(receipt.protected_start, 2);
        assert_eq!(
            &receipt.history.annotated_items()[2..],
            &original.annotated_items()[2..]
        );
        assert_eq!(
            receipt.history.annotated_items()[1].item,
            expected_summary.item
        );
        assert_ne!(session.current_window_id().await, window_before);
    } else {
        assert_eq!(
            session.clone_history().await.annotated_items(),
            original.annotated_items()
        );
        assert_eq!(session.current_window_id().await, window_before);
    }
    check_newer_input_on_resume(&session, &turn).await;
}

#[test_case::test_case(false; "drop_after_canonical")]
#[test_case::test_case(true; "cancel_after_canonical")]
#[tokio::test]
async fn body_recovery_checkpoint_cancel_after_canonical_preserves_input(cancel_token: bool) {
    let (pending, mut appends) = mpsc::unbounded_channel();
    let (session, turn) =
        session_with_checkpoint_fault(CheckpointFault::AfterCanonical(pending)).await;
    let original = session.clone_history().await;
    let cancellation = CancellationToken::new();
    let mut installation = Box::pin(install_prefix(&session, &turn, &cancellation));
    let append = tokio::select! {
        append = appends.recv() => append.expect("canonical checkpoint"),
        result = &mut installation => panic!("expected acknowledgement gate, got {result}"),
    };
    if cancel_token {
        cancellation.cancel();
        assert!(!installation.await);
        assert!(append.release.send(()).is_err());
        assert!(append.completed.await.is_err());
    } else {
        drop(installation);
        assert!(append.release.send(()).is_err());
        assert!(append.completed.await.is_err());
    }
    assert_eq!(
        session.clone_history().await.annotated_items(),
        original.annotated_items()
    );
    assert_original_records_kept(&session, &original).await;
    check_newer_input_on_resume(&session, &turn).await;
}

#[test_case::test_case("cancel")]
#[test_case::test_case("owner")]
#[test_case::test_case("history")]
#[test_case::test_case("auth")]
#[test_case::test_case("protected")]
#[tokio::test]
async fn body_recovery_checkpoint_refuses_stale_or_protected_prefix(change: &str) {
    let (session, turn) = session_with_checkpoint_fault(CheckpointFault::Allow).await;
    let original = session.clone_history().await;
    let step = session
        .capture_step_context(Arc::clone(&turn), &CancellationToken::new())
        .await
        .expect("step");
    let cancellation = CancellationToken::new();
    let mut prefix = CompactionPrefix {
        generation: original.history_version(),
        len: 2,
        auth_owner_generation: session.services.model_client.auth_owner_generation(),
    };
    let mut protected_start = 2;
    match change {
        "cancel" => cancellation.cancel(),
        "owner" => session.record_started_turn("replacement", None).await,
        "history" => prefix.generation = prefix.generation.wrapping_add(1),
        "auth" => {
            prefix.auth_owner_generation = Some(
                prefix
                    .auth_owner_generation
                    .unwrap_or_default()
                    .wrapping_add(1),
            )
        }
        "protected" => protected_start = 1,
        _ => unreachable!(),
    }
    let mut replacement = original.annotated_items()[..2]
        .iter()
        .filter(|item| super::super::preserving_compaction::retain_for_body_recovery(item))
        .cloned()
        .collect::<Vec<_>>();
    replacement.push(ResponseItemEnvelope::new(ResponseItem::Compaction {
        id: None,
        encrypted_content: "summary".to_owned(),
        internal_chat_message_metadata_passthrough: None,
    }));
    assert!(
        !session
            .install_preserving_compaction_prefix(
                &step,
                &original,
                protected_start,
                prefix,
                replacement,
                "summary-response".to_owned(),
                &cancellation,
            )
            .await
    );
    assert_eq!(checkpoint_attempts(&session), 0);
    assert_eq!(
        session.clone_history().await.annotated_items(),
        original.annotated_items()
    );
}

#[test]
fn body_recovery_checkpoint_selects_only_closed_groups() {
    use super::super::preserving_compaction::completed_prefix_ends;
    for (name, output, expected) in [
        (
            "matching",
            json!({"type":"function_call_output", "call_id":"call", "output":"complete"}),
            vec![3, 4],
        ),
        (
            "wrong_attempt",
            json!({"type":"function_call_output", "call_id":"other", "output":"complete"}),
            vec![],
        ),
        (
            "wrong_name",
            json!({"type":"function_call_output", "call_id":"call", "name":"other", "output":"complete"}),
            vec![],
        ),
        (
            "wrong_namespace",
            json!({"type":"function_call_output", "call_id":"call", "namespace":"other", "output":"complete"}),
            vec![],
        ),
        (
            "wrong_kind",
            json!({"type":"custom_tool_call_output", "call_id":"call", "output":"complete"}),
            vec![],
        ),
    ] {
        let items = vec![user_message("historical input").into(), ResponseItemEnvelope::new(
            serde_json::from_value(json!({"type":"function_call", "call_id":"call", "name":"owned_tool", "namespace":"owned", "arguments":"{}"})).unwrap()
        ), ResponseItemEnvelope::new(serde_json::from_value(output).unwrap()), assistant_message("done").into()];
        assert_eq!(
            completed_prefix_ends(&items, items.len()),
            expected,
            "{name}"
        );
        assert!(
            completed_prefix_ends(&items, 2).is_empty(),
            "no boundary inside a group: {name}"
        );
    }
    let mut items = vec![
        user_message("older").into(),
        assistant_message("done").into(),
    ];
    items.push(ResponseItemEnvelope::new(serde_json::from_value(json!({"type":"function_call", "call_id":"open", "name":"owned_tool", "arguments":"{}"})).unwrap()));
    items.push(assistant_message("live").into());
    assert_eq!(completed_prefix_ends(&items, items.len()), vec![2]);
    assert!(completed_prefix_ends(&items, items.len() + 1).is_empty());
}
