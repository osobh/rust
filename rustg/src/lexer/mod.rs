//! GPU-accelerated lexical analysis

use crate::error::Result;
use std::path::Path;

pub mod tokenizer;

/// Token type (matches gpu_types.h)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TokenType {
    /// Invalid or unrecognized token
    Invalid = 0,
    /// End of file marker
    EOF,
    
    // Literals
    /// Variable or function identifier
    Identifier,
    /// Integer literal (e.g., 42, 0x1F)
    IntegerLiteral,
    /// Floating point literal (e.g., 3.14, 1e-5)
    FloatLiteral,
    /// String literal (e.g., "hello")
    StringLiteral,
    /// Character literal (e.g., 'a')
    CharLiteral,
    /// Boolean literal (true or false)
    BoolLiteral,
    
    // Keywords
    /// 'fn' keyword for function declarations
    KeywordFn,
    /// 'let' keyword for variable binding
    KeywordLet,
    /// 'mut' keyword for mutable variables
    KeywordMut,
    /// 'const' keyword for constants
    KeywordConst,
    /// 'if' keyword for conditionals
    KeywordIf,
    /// 'else' keyword for alternative branches
    KeywordElse,
    /// 'while' keyword for loops
    KeywordWhile,
    /// 'for' keyword for iteration
    KeywordFor,
    /// 'return' keyword for function returns
    KeywordReturn,
    /// 'struct' keyword for structure definitions
    KeywordStruct,
    /// 'enum' keyword for enumeration definitions
    KeywordEnum,
    /// 'impl' keyword for implementation blocks
    KeywordImpl,
    /// 'trait' keyword for trait definitions
    KeywordTrait,
    /// 'pub' keyword for public visibility
    KeywordPub,
    /// 'use' keyword for imports
    KeywordUse,
    /// 'mod' keyword for module declarations
    KeywordMod,
    /// 'self' keyword for self reference
    KeywordSelf,
    /// 'super' keyword for parent module reference
    KeywordSuper,
    /// 'crate' keyword for crate root reference
    KeywordCrate,
    
    // Operators
    /// '+' addition operator
    Plus,
    /// '-' subtraction operator
    Minus,
    /// '*' multiplication operator
    Star,
    /// '/' division operator
    Slash,
    /// '%' modulo operator
    Percent,
    /// '&' bitwise AND operator
    Ampersand,
    /// '|' bitwise OR operator
    Pipe,
    /// '^' bitwise XOR operator
    Caret,
    /// '~' bitwise NOT operator
    Tilde,
    /// '!' logical NOT operator
    Bang,
    /// '=' equality operator
    Equal,
    /// '<' less than operator
    Less,
    /// '>' greater than operator
    Greater,
    
    // Delimiters
    /// '(' left parenthesis
    LeftParen,
    /// ')' right parenthesis
    RightParen,
    /// '{' left brace
    LeftBrace,
    /// '}' right brace
    RightBrace,
    /// '[' left bracket
    LeftBracket,
    /// ']' right bracket
    RightBracket,
    /// ';' semicolon
    Semicolon,
    /// ',' comma
    Comma,
    /// '.' dot operator
    Dot,
    /// ':' colon
    Colon,
    /// '::' double colon for namespacing
    DoubleColon,
    /// '->' arrow for function returns
    Arrow,
    /// '=>' fat arrow for match arms
    FatArrow,
    
    // Comments and whitespace
    /// Line comment starting with '//'
    LineComment,
    /// Block comment /* ... */
    BlockComment,
    /// Whitespace characters (spaces, tabs)
    Whitespace,
    /// Newline character
    Newline,
}

/// Token structure representing a lexical unit
#[derive(Debug, Clone)]
pub struct Token {
    /// Type of the token
    pub ty: TokenType,
    /// Starting position in source code (byte offset)
    pub start_pos: u32,
    /// Length of the token in bytes
    pub length: u32,
    /// Line number (1-based)
    pub line: u32,
    /// Column number (1-based)
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