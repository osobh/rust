//! GPU-accelerated lexical analysis

use crate::error::Result;
use std::path::Path;

pub mod tokenizer;

/// Token type (matches gpu_types.h)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TokenType {
    Invalid = 0,
    EOF,
    
    // Literals
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    CharLiteral,
    BoolLiteral,
    
    // Keywords
    KeywordFn,
    KeywordLet,
    KeywordMut,
    KeywordConst,
    KeywordIf,
    KeywordElse,
    KeywordWhile,
    KeywordFor,
    KeywordReturn,
    KeywordStruct,
    KeywordEnum,
    KeywordImpl,
    KeywordTrait,
    KeywordPub,
    KeywordUse,
    KeywordMod,
    KeywordSelf,
    KeywordSuper,
    KeywordCrate,
    
    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Ampersand,
    Pipe,
    Caret,
    Tilde,
    Bang,
    Equal,
    Less,
    Greater,
    
    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Semicolon,
    Comma,
    Dot,
    Colon,
    DoubleColon,
    Arrow,
    FatArrow,
    
    // Comments and whitespace
    LineComment,
    BlockComment,
    Whitespace,
    Newline,
}

/// Token structure
#[derive(Debug, Clone)]
pub struct Token {
    pub ty: TokenType,
    pub start_pos: u32,
    pub length: u32,
    pub line: u32,
    pub column: u32,
}

/// Tokenize a source file on GPU
pub fn tokenize_gpu(source: &str) -> Result<Vec<Token>> {
    tokenizer::gpu::tokenize(source)
}

/// Tokenize a source file on CPU (reference implementation)
pub fn tokenize_cpu(source: &str) -> Result<Vec<Token>> {
    tokenizer::cpu::tokenize(source)
}