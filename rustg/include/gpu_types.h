#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace rustg {

// Basic type aliases for GPU code
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

// Token types for lexical analysis
enum class TokenType : u32 {
    // Special
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
};

// Token structure optimized for GPU (Structure-of-Arrays will be used in practice)
struct Token {
    TokenType type;
    u32 start_pos;
    u32 length;
    u32 line;
    u32 column;
};

// Character classification for fast lookup
enum class CharClass : u8 {
    Invalid = 0,
    Whitespace,
    Letter,
    Digit,
    Operator,
    Delimiter,
    Quote,
    Newline,
    Other,
};

// AST Node types
enum class ASTNodeType : u32 {
    Invalid = 0,
    Program,
    Function,
    Parameter,
    Block,
    Statement,
    Expression,
    BinaryOp,
    UnaryOp,
    Literal,
    Identifier,
    Call,
    StructDef,
    EnumDef,
    ImplBlock,
    TraitDef,
};

// Memory pool configuration
struct MemoryPoolConfig {
    size_t block_size;
    u32 num_blocks;
    void* base_address;
    u32* allocation_bitmap;
};

// Compilation context for GPU kernels
struct CompilationContext {
    const char* source_code;
    size_t source_length;
    Token* token_buffer;
    u32* token_count;
    u32 max_tokens;
    void* ast_memory;
    size_t ast_memory_size;
    u32* error_buffer;
    u32* error_count;
    u32 max_errors;
};

// Thread configuration constants
constexpr u32 WARP_SIZE = 32;
constexpr u32 BLOCK_SIZE = 256;  // 8 warps
constexpr u32 MAX_SHARED_MEMORY = 48 * 1024;  // 48KB
constexpr u32 CHARS_PER_THREAD = 64;  // Each thread processes 64 characters

// Error codes
enum class ErrorCode : u32 {
    Success = 0,
    OutOfMemory,
    InvalidSyntax,
    UnexpectedToken,
    UnterminatedString,
    InvalidCharacter,
    TooManyTokens,
    TooManyErrors,
};

} // namespace rustg