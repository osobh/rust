// LSP type definitions for rust-analyzer-g
// Extracted from main binary to reduce file size

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// LSP JSON-RPC message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcMessage {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: Option<String>,
    pub params: Option<Value>,
    pub result: Option<Value>,
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

/// LSP Initialize request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    #[serde(rename = "processId")]
    pub process_id: Option<i32>,
    #[serde(rename = "clientInfo")]
    pub client_info: Option<ClientInfo>,
    #[serde(rename = "rootUri")]
    pub root_uri: Option<String>,
    pub capabilities: ClientCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCapabilities {
    #[serde(rename = "textDocument")]
    pub text_document: Option<TextDocumentClientCapabilities>,
    pub workspace: Option<WorkspaceClientCapabilities>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentClientCapabilities {
    pub hover: Option<HoverCapability>,
    pub completion: Option<CompletionCapability>,
    #[serde(rename = "publishDiagnostics")]
    pub publish_diagnostics: Option<PublishDiagnosticsCapability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverCapability {
    #[serde(rename = "dynamicRegistration")]
    pub dynamic_registration: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionCapability {
    #[serde(rename = "dynamicRegistration")]
    pub dynamic_registration: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishDiagnosticsCapability {
    #[serde(rename = "relatedInformation")]
    pub related_information: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceClientCapabilities {
    #[serde(rename = "didChangeConfiguration")]
    pub did_change_configuration: Option<DidChangeConfigurationCapability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidChangeConfigurationCapability {
    #[serde(rename = "dynamicRegistration")]
    pub dynamic_registration: Option<bool>,
}

/// LSP Initialize result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    pub capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: Option<ServerInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    #[serde(rename = "textDocumentSync")]
    pub text_document_sync: Option<i32>,
    #[serde(rename = "hoverProvider")]
    pub hover_provider: Option<bool>,
    #[serde(rename = "completionProvider")]
    pub completion_provider: Option<CompletionOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionOptions {
    #[serde(rename = "triggerCharacters")]
    pub trigger_characters: Option<Vec<String>>,
}

/// Text document synchronization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidOpenTextDocumentParams {
    #[serde(rename = "textDocument")]
    pub text_document: TextDocumentItem,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentItem {
    pub uri: String,
    #[serde(rename = "languageId")]
    pub language_id: String,
    pub version: i32,
    pub text: String,
}

/// Hover request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverParams {
    #[serde(rename = "textDocument")]
    pub text_document: TextDocumentIdentifier,
    pub position: Position,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentIdentifier {
    pub uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

/// Hover result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hover {
    pub contents: HoverContents,
    pub range: Option<Range>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HoverContents {
    Scalar(String),
    Array(Vec<String>),
    Markup(MarkupContent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkupContent {
    pub kind: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

/// Completion request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionParams {
    #[serde(rename = "textDocument")]
    pub text_document: TextDocumentIdentifier,
    pub position: Position,
    pub context: Option<CompletionContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionContext {
    #[serde(rename = "triggerKind")]
    pub trigger_kind: i32,
    #[serde(rename = "triggerCharacter")]
    pub trigger_character: Option<String>,
}

/// Completion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionList {
    #[serde(rename = "isIncomplete")]
    pub is_incomplete: bool,
    pub items: Vec<CompletionItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    pub label: String,
    pub kind: Option<i32>,
    pub detail: Option<String>,
    pub documentation: Option<String>,
    #[serde(rename = "insertText")]
    pub insert_text: Option<String>,
}

/// Diagnostic types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishDiagnosticsParams {
    pub uri: String,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub range: Range,
    pub severity: Option<i32>,
    pub code: Option<String>,
    pub source: Option<String>,
    pub message: String,
    #[serde(rename = "relatedInformation")]
    pub related_information: Option<Vec<DiagnosticRelatedInformation>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticRelatedInformation {
    pub location: Location,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub uri: String,
    pub range: Range,
}