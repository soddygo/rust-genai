use crate::ModelIden;
use crate::adapter::openai::OpenAIAdapter;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
	ChatMessage, ChatOptionsSet, ChatRequest, ChatResponse, ChatRole, ChatStreamResponse,
	ContentPart,
};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::WebResponse;
use crate::{Result, ServiceTarget};
use reqwest::RequestBuilder;
use serde_json;
use std::collections::HashMap;

pub struct MiniMaxAdapter;

impl MiniMaxAdapter {
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "MINIMAX_API_KEY";
}

impl Adapter for MiniMaxAdapter {
	const DEFAULT_API_KEY_ENV_NAME: Option<&'static str> = Some(Self::API_KEY_DEFAULT_ENV_NAME);

	fn default_endpoint() -> Endpoint {
		const BASE_URL: &str = "https://api.minimax.chat/v1/";
		Endpoint::from_static(BASE_URL)
	}

	fn default_auth() -> AuthData {
		match Self::DEFAULT_API_KEY_ENV_NAME {
			Some(env_name) => AuthData::from_env(env_name),
			None => AuthData::None,
		}
	}

	async fn all_model_names(kind: AdapterKind, endpoint: Endpoint, auth: AuthData) -> Result<Vec<String>> {
		OpenAIAdapter::list_model_names_for_end_target(kind, endpoint, auth).await
	}

	fn get_service_url(model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> Result<String> {
		OpenAIAdapter::util_get_service_url(model, service_type, endpoint)
	}

	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		mut chat_req: ChatRequest,
		chat_options: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		merge_system_messages(&mut chat_req);
		reorder_tool_messages(&mut chat_req);
		let mut data =
			OpenAIAdapter::util_to_web_request_data(target, service_type, chat_req, chat_options, None)?;

		// Post-process the payload for MiniMax API compatibility.
		// MiniMax differs from OpenAI in several ways:
		// - requires max_completion_tokens instead of max_tokens
		// - does not support stream_options, strict in tools, or tool_choice
		// - requires string content (not content arrays)
		// - needs reasoning_split to separate reasoning from content
		if let Some(obj) = data.payload.as_object_mut() {
			// 1. Rename max_tokens → max_completion_tokens
			if let Some(max_tokens) = obj.remove("max_tokens") {
				obj.insert("max_completion_tokens".to_string(), max_tokens);
			}

			// 2. Remove fields MiniMax does not support
			obj.remove("stream_options");
			obj.remove("tool_choice");
			obj.remove("parallel_tool_calls");

			// 3. Remove strict from each tool definition
			if let Some(tools) = obj.get_mut("tools").and_then(|v| v.as_array_mut()) {
				for tool in tools {
					if let Some(func) = tool
						.as_object_mut()
						.and_then(|t| t.get_mut("function"))
						.and_then(|f| f.as_object_mut())
					{
						func.remove("strict");
					}
				}
			}

			// 4. Flatten array content to string in messages
			if let Some(messages) = obj.get_mut("messages").and_then(|v| v.as_array_mut()) {
				for msg in messages {
					if let Some(msg_obj) = msg.as_object_mut() {
						if let Some(content) = msg_obj.get("content") {
							if content.is_array() {
								let flattened: String = content
									.as_array()
									.into_iter()
									.flatten()
									.filter_map(|part| {
										part.get("text").and_then(|t| t.as_str())
									})
									.collect::<Vec<_>>()
									.join("");
								msg_obj.insert(
									"content".to_string(),
									serde_json::Value::String(flattened),
								);
							}
						}
					}
				}
			}

			// 5. Inject reasoning_split so MiniMax returns reasoning in a
			//    separate reasoning_content delta instead of embedding
			//    <think> tags in content.
			obj.insert("reasoning_split".to_string(), serde_json::json!(true));
		}
		Ok(data)
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		OpenAIAdapter::to_chat_response(model_iden, web_response, options_set)
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		OpenAIAdapter::to_chat_stream(model_iden, reqwest_builder, options_set)
	}

	fn to_embed_request_data(
		service_target: crate::ServiceTarget,
		embed_req: crate::embed::EmbedRequest,
		options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::adapter::WebRequestData> {
		OpenAIAdapter::to_embed_request_data(service_target, embed_req, options_set)
	}

	fn to_embed_response(
		model_iden: crate::ModelIden,
		web_response: crate::webc::WebResponse,
		options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::embed::EmbedResponse> {
		OpenAIAdapter::to_embed_response(model_iden, web_response, options_set)
	}
}

/// Merge ChatRequest.system + all System-role ChatMessage entries into
/// a single system prompt, joined by "\n\n".
fn merge_system_messages(chat_req: &mut ChatRequest) {
	let mut systems: Vec<String> = Vec::new();
	if let Some(ref sys) = chat_req.system {
		if !sys.is_empty() {
			systems.push(sys.clone());
		}
	}
	chat_req.messages.retain(|msg| {
		if msg.role == ChatRole::System {
			if let Some(text) = msg.content.joined_texts() {
				if !text.is_empty() {
					systems.push(text);
				}
			}
			false
		} else {
			true
		}
	});
	if systems.is_empty() {
		chat_req.system = None;
	} else {
		tracing::debug!(
			count = systems.len(),
			"MiniMax: merged {} system messages into single system prompt",
			systems.len()
		);
		chat_req.system = Some(systems.join("\n\n"));
	}
}

/// MiniMax requires each tool result to immediately follow its corresponding
/// tool call (error 2013: "tool call result does not follow tool call").
///
/// Codex emits all FunctionCall items first, then all FunctionCallOutput
/// items. This function interleaves them so the message order satisfies
/// MiniMax's constraint.
fn reorder_tool_messages(chat_req: &mut ChatRequest) {
	let msgs = std::mem::take(&mut chat_req.messages);
	let mut result: Vec<ChatMessage> = Vec::with_capacity(msgs.len());
	let mut i = 0;

	while i < msgs.len() {
		// Check if this message is an assistant message with tool calls
		let has_tool_calls = msgs[i].role == ChatRole::Assistant
			&& msgs[i]
				.content
				.parts()
				.iter()
				.any(|p| matches!(p, ContentPart::ToolCall(_)));

		if !has_tool_calls {
			result.push(msgs[i].clone());
			i += 1;
			continue;
		}

		// Collect consecutive tool call messages and their call_ids
		let mut tc_msgs: Vec<(ChatMessage, String)> = Vec::new();
		while i < msgs.len() {
			let is_tc = msgs[i].role == ChatRole::Assistant
				&& msgs[i]
					.content
					.parts()
					.iter()
					.any(|p| matches!(p, ContentPart::ToolCall(_)));
			if !is_tc {
				break;
			}
			for part in msgs[i].content.parts() {
				if let ContentPart::ToolCall(tc) = part {
					tc_msgs.push((msgs[i].clone(), tc.call_id.clone()));
				}
			}
			i += 1;
		}

		// Collect tool result messages that immediately follow
		let mut tr_map: HashMap<String, ChatMessage> = HashMap::new();
		while i < msgs.len() && msgs[i].role == ChatRole::Tool {
			for part in msgs[i].content.parts() {
				if let ContentPart::ToolResponse(tr) = part {
					tr_map.insert(tr.call_id.clone(), msgs[i].clone());
				}
			}
			i += 1;
		}

		// Interleave: for each tool call, emit it followed by its result
		let reordered_count = tc_msgs.len();
		for (tc_msg, call_id) in &tc_msgs {
			result.push(tc_msg.clone());
			if let Some(tr_msg) = tr_map.remove(call_id) {
				result.push(tr_msg);
			}
		}

		// Emit any remaining tool results that didn't match a tool call
		for (_, tr_msg) in tr_map {
			result.push(tr_msg);
		}

		if reordered_count > 1 {
			tracing::debug!(
				count = reordered_count,
				"MiniMax: interleaved {} tool calls with their results",
				reordered_count
			);
		}
	}

	chat_req.messages = result;
}
