//! Canonical item comparison keeps persisted identity and ignores runtime output success only.
#![cfg(test)]

use crate::session::preserving_compaction::canonical_response_items_match;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

fn persisted_items(items: &[ResponseItemEnvelope]) -> Value {
    serde_json::to_value(items.iter().map(|item| &item.item).collect::<Vec<_>>())
        .expect("actual persisted response items")
}

fn restore_items(value: Value) -> Vec<ResponseItemEnvelope> {
    serde_json::from_value::<Vec<ResponseItem>>(value)
        .expect("deserialize response items")
        .into_iter()
        .map(ResponseItemEnvelope::new)
        .collect()
}

fn history(
    function_success: bool,
    custom_success: bool,
    structured_outputs: bool,
) -> Vec<ResponseItemEnvelope> {
    let function_output = if structured_outputs {
        json!([{"type": "input_text", "text": "Function result: café 雪."}])
    } else {
        json!("Function result: café 雪.")
    };
    let custom_output = if structured_outputs {
        json!([{"type": "input_text", "text": "Custom result: résumé 🦀."}])
    } else {
        json!("Custom result: résumé 🦀.")
    };
    let mut items = restore_items(json!([
        {
            "type": "function_call",
            "id": "fc_saved_function",
            "call_id": "saved-function-call",
            "name": "owned_function",
            "namespace": "functions",
            "arguments": "{\"count\":1}",
            "status": "completed"
        },
        {
            "type": "function_call_output",
            "id": "fco_saved_function",
            "call_id": "saved-function-call",
            "name": "owned_function",
            "namespace": "functions",
            "output": function_output
        },
        {
            "type": "custom_tool_call",
            "id": "ctc_saved_custom",
            "call_id": "saved-custom-call",
            "name": "owned_custom",
            "namespace": "custom_tools",
            "input": "Original custom input.",
            "status": "completed"
        },
        {
            "type": "custom_tool_call_output",
            "id": "ctco_saved_custom",
            "call_id": "saved-custom-call",
            "name": "owned_custom",
            "output": custom_output
        },
        {
            "type": "message",
            "id": "msg_current_input",
            "role": "user",
            "content": [{"type": "input_text", "text": "Preserve current input exactly."}]
        }
    ]));

    for (index, success) in [(1, function_success), (3, custom_success)] {
        match &mut items[index].item {
            ResponseItem::FunctionCallOutput { output, .. }
            | ResponseItem::CustomToolCallOutput { output, .. } => {
                output.success = Some(success);
            }
            _ => unreachable!("synthetic output position"),
        }
    }
    items
}

#[test]
fn canonical_comparison_accepts_real_roundtrip_loss_of_runtime_output_success() {
    for structured_outputs in [false, true] {
        for function_success in [false, true] {
            for custom_success in [false, true] {
                let expected = history(function_success, custom_success, structured_outputs);
                let encoded = serde_json::to_vec(&persisted_items(&expected))
                    .expect("serialize canonical history");
                let canonical =
                    restore_items(serde_json::from_slice(&encoded).expect("canonical JSON"));
                let expected_before = expected.clone();
                let canonical_before = canonical.clone();

                assert_ne!(canonical, expected, "runtime metadata really differs");
                for index in [1, 3] {
                    match &canonical[index].item {
                        ResponseItem::FunctionCallOutput { output, .. }
                        | ResponseItem::CustomToolCallOutput { output, .. } => {
                            assert_eq!(output.success, None);
                        }
                        _ => unreachable!("synthetic output position"),
                    }
                }
                assert_eq!(persisted_items(&canonical), persisted_items(&expected),);
                assert!(canonical_response_items_match(&canonical, &expected));
                assert!(canonical_response_items_match(&expected, &canonical));
                assert!(canonical_response_items_match(&expected, &expected));
                assert_eq!(expected, expected_before, "live history remains intact");
                assert_eq!(
                    canonical, canonical_before,
                    "canonical history remains intact"
                );
            }
        }
    }
}

#[test]
fn canonical_comparison_rejects_tampered_persisted_fields() {
    let expected = history(true, false, false);
    let expected_before = expected.clone();
    let encoded =
        serde_json::to_vec(&persisted_items(&expected)).expect("serialize canonical history");
    let canonical_json: Value = serde_json::from_slice(&encoded).expect("canonical JSON");
    let mutations = [
        ("function call id", "/0/id", json!("fc_changed")),
        ("function call identity", "/0/call_id", json!("other-call")),
        ("function name", "/0/name", json!("other_function")),
        (
            "function namespace",
            "/0/namespace",
            json!("other_namespace"),
        ),
        ("function arguments", "/0/arguments", json!("{\"count\":2}")),
        ("function status", "/0/status", json!("in_progress")),
        ("function output id", "/1/id", json!("fco_changed")),
        (
            "function output call identity",
            "/1/call_id",
            json!("other-call"),
        ),
        ("function output name", "/1/name", json!("other_function")),
        (
            "function output namespace",
            "/1/namespace",
            json!("other_namespace"),
        ),
        (
            "function output body",
            "/1/output",
            json!("Changed function result."),
        ),
        (
            "function output body shape",
            "/1/output",
            json!([{"type": "input_text", "text": "Function result: café 雪."}]),
        ),
        ("custom call id", "/2/id", json!("ctc_changed")),
        ("custom call identity", "/2/call_id", json!("other-call")),
        ("custom name", "/2/name", json!("other_custom")),
        ("custom namespace", "/2/namespace", json!("other_namespace")),
        ("custom input", "/2/input", json!("Changed custom input.")),
        ("custom status", "/2/status", json!("in_progress")),
        ("custom output id", "/3/id", json!("ctco_changed")),
        (
            "custom output call identity",
            "/3/call_id",
            json!("other-call"),
        ),
        ("custom output name", "/3/name", json!("other_custom")),
        (
            "custom output body",
            "/3/output",
            json!("Changed custom result."),
        ),
        (
            "custom output body shape",
            "/3/output",
            json!([{"type": "input_text", "text": "Custom result: résumé 🦀."}]),
        ),
        ("current message id", "/4/id", json!("msg_changed")),
        ("current role", "/4/role", json!("developer")),
        (
            "current text",
            "/4/content/0/text",
            json!("Changed current input."),
        ),
    ];

    for (label, pointer, replacement) in mutations {
        let mut tampered_json = canonical_json.clone();
        *tampered_json
            .pointer_mut(pointer)
            .expect("existing persisted field") = replacement;
        let tampered = restore_items(tampered_json);
        let tampered_before = tampered.clone();

        assert!(
            !canonical_response_items_match(&tampered, &expected),
            "{label}"
        );
        assert!(
            !canonical_response_items_match(&expected, &tampered),
            "{label}"
        );
        assert_eq!(
            tampered, tampered_before,
            "{label}: canonical source unchanged"
        );
        assert_eq!(expected, expected_before, "{label}: live source unchanged");
    }
}

#[test]
fn canonical_comparison_rejects_changed_item_order_and_multiplicity() {
    let expected = history(true, false, true);
    let expected_before = expected.clone();
    let encoded =
        serde_json::to_vec(&persisted_items(&expected)).expect("serialize canonical history");
    let canonical = restore_items(serde_json::from_slice(&encoded).expect("canonical JSON"));

    for alteration in ["reordered", "missing", "inserted", "duplicated"] {
        let mut tampered = canonical.clone();
        match alteration {
            "reordered" => tampered.swap(1, 3),
            "missing" => {
                tampered.remove(1);
            }
            "inserted" => tampered.insert(
                2,
                ResponseItemEnvelope::new(
                    serde_json::from_value(json!({
                        "type": "message",
                        "id": "msg_inserted",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Unexpected persisted output."}]
                    }))
                    .expect("synthetic inserted item"),
                ),
            ),
            "duplicated" => tampered.insert(2, tampered[1].clone()),
            _ => unreachable!("explicit alteration matrix"),
        }
        let tampered_before = tampered.clone();

        assert!(
            !canonical_response_items_match(&tampered, &expected),
            "{alteration}"
        );
        assert!(
            !canonical_response_items_match(&expected, &tampered),
            "{alteration}"
        );
        assert_eq!(
            tampered, tampered_before,
            "{alteration}: canonical source unchanged"
        );
        assert_eq!(
            expected, expected_before,
            "{alteration}: live source unchanged"
        );
    }
}
