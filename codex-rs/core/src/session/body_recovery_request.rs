//! Working copies of the actual rejected JSON preserve all untouched wire items.

use crate::client::ModelClientSession;
use crate::context_manager::ContextManager;
use crate::responses_metadata::CodexResponsesMetadata;
use codex_api::EncodedJsonBody;
use codex_protocol::error::Result;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use serde::Serialize;
use serde::ser::SerializeMap;
use serde_json::value::RawValue;
use std::collections::BTreeMap;

struct RoutingFirst<'a>(&'a BTreeMap<String, Box<RawValue>>);

impl Serialize for RoutingFirst<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for name in ["model", "stream", "service_tier", "input"] {
            if let Some(value) = self.0.get(name) {
                map.serialize_entry(name, value)?;
            }
        }
        for (name, value) in self.0 {
            if !matches!(name.as_str(), "model" | "stream" | "service_tier" | "input") {
                map.serialize_entry(name, value)?;
            }
        }
        map.end()
    }
}

pub(super) struct FrozenBody {
    fields: BTreeMap<String, Box<RawValue>>,
    input: Vec<Box<RawValue>>,
    positions: Vec<Option<usize>>,
    leading: usize,
    pub(super) aligned: usize,
}

impl FrozenBody {
    pub(super) fn capture(
        body: &EncodedJsonBody,
        history: &ContextManager,
        protected: usize,
        client: &ModelClientSession,
        model: &ModelInfo,
        metadata: &CodexResponsesMetadata,
    ) -> Result<Option<Self>> {
        let mut fields: BTreeMap<String, Box<RawValue>> = serde_json::from_slice(body.as_bytes())?;
        if fields
            .get("model")
            .and_then(|raw| serde_json::from_str::<String>(raw.get()).ok())
            .as_deref()
            != Some(model.slug.as_str())
        {
            return Ok(None);
        }
        let Some(raw_input) = fields.remove("input") else {
            return Ok(None);
        };
        let input: Vec<Box<RawValue>> = serde_json::from_str(raw_input.get())?;
        let values = input
            .iter()
            .map(|raw| serde_json::from_str::<serde_json::Value>(raw.get()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut positions = Vec::new();
        let mut cursor = 0;
        let mut leading = 0;
        let mut started = false;
        for envelope in history.annotated_items().iter().take(protected) {
            let Some(projected) =
                client.project_body_recovery_item(&envelope.item, model, metadata)?
            else {
                if matches!(envelope.item, ResponseItem::ConfigurationUpdate { .. }) {
                    positions.push(None);
                    continue;
                }
                break;
            };
            let value = serde_json::to_value(projected)?;
            while !started && cursor < input.len() && values[cursor] != value {
                let extra: ResponseItem = serde_json::from_str(input[cursor].get())?;
                if !matches!(&extra, ResponseItem::Message { role, .. } if role == "developer" || role == "system")
                    && !matches!(
                        extra,
                        ResponseItem::AdditionalTools { .. }
                            | ResponseItem::ConfigurationUpdate { .. }
                    )
                {
                    return Ok(None);
                }
                cursor += 1;
                leading = cursor;
            }
            if cursor >= input.len() || values[cursor] != value {
                break;
            }
            positions.push(Some(cursor));
            cursor += 1;
            started = true;
        }
        let aligned = positions.len();
        Ok(Some(Self {
            fields,
            input,
            positions,
            leading,
            aligned,
        }))
    }

    fn encode(
        &self,
        input: Vec<Box<RawValue>>,
        metadata: &CodexResponsesMetadata,
        compaction: bool,
    ) -> Result<EncodedJsonBody> {
        let mut fields = self.fields.clone();
        fields.insert("input".to_owned(), serde_json::value::to_raw_value(&input)?);
        let mut client_metadata: BTreeMap<String, String> = match fields.get("client_metadata") {
            Some(raw) => serde_json::from_str(raw.get())?,
            None => BTreeMap::new(),
        };
        client_metadata.extend(metadata.client_metadata(false));
        fields.insert(
            "client_metadata".to_owned(),
            serde_json::value::to_raw_value(&client_metadata)?,
        );
        if compaction {
            fields.insert("tools".to_owned(), RawValue::from_string("[]".to_owned())?);
            fields.insert(
                "tool_choice".to_owned(),
                RawValue::from_string("\"none\"".to_owned())?,
            );
            fields.insert(
                "parallel_tool_calls".to_owned(),
                RawValue::from_string("false".to_owned())?,
            );
        }
        Ok(EncodedJsonBody::encode(&RoutingFirst(&fields))?)
    }

    pub(super) fn compaction(
        &self,
        selected: usize,
        prompt: &ResponseItem,
        metadata: &CodexResponsesMetadata,
    ) -> Result<EncodedJsonBody> {
        let mut input = self.input[..self.leading].to_vec();
        for position in self.positions[..selected].iter().flatten() {
            input.push(self.input[*position].clone());
        }
        input.push(serde_json::value::to_raw_value(prompt)?);
        self.encode(input, metadata, true)
    }

    pub(super) fn retry(
        &self,
        selected: usize,
        retained: &[bool],
        summary: &ResponseItem,
        metadata: &CodexResponsesMetadata,
    ) -> Result<EncodedJsonBody> {
        let end = self.positions[..selected]
            .iter()
            .flatten()
            .max()
            .map(|position| position + 1)
            .unwrap_or(self.leading);
        let mut input = self.input[..self.leading].to_vec();
        for (index, position) in self.positions[..selected].iter().enumerate() {
            if retained[index]
                && let Some(position) = position
            {
                input.push(self.input[*position].clone());
            }
        }
        input.push(serde_json::value::to_raw_value(summary)?);
        input.extend_from_slice(&self.input[end..]);
        self.encode(input, metadata, false)
    }
}
