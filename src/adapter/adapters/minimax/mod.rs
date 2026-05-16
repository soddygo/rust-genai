//! MiniMax API adapter — OpenAI-compatible with system message merging.
//!
//! MiniMax rejects multiple `role: "system"` messages (error 2013).
//! This adapter merges `ChatRequest.system` + System-role `ChatMessage` entries
//! into a single system prompt before serialization.
//! API Documentation: <https://platform.minimaxi.com/>
//! Endpoint: <https://api.minimax.chat/v1/>

mod adapter_impl;

pub use adapter_impl::*;
