//! New strict recovery requirements, separate from stock in-memory adoption semantics.

use super::*;
use codex_thread_store::*;
use pretty_assertions::assert_eq;
use std::any::Any;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

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
