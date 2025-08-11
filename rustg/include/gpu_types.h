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
    Unknown = 0,    // Unknown/undefined token
    Invalid,
    EndOfFile,  // Avoid EOF macro conflict
    
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
    
    // Advanced Rust tokens
    Lifetime,        // 'a, 'static, etc.
    Generic,         // Generic type parameters
    Underscore,      // _ wildcard
    DotDot,          // ..
    DotDotEq,        // ..=
    RefMut,          // &mut
    TuplePattern,    // Pattern in tuples
    SlicePattern,    // Pattern in slices
    MacroBang,       // macro! invocation
    
    // Operators
    Plus,
    Minus,
    Star,
    Dollar,          // $ for macro variables
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
    ColonColon = DoubleColon,  // Alias for compatibility
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

// Macro fragment types for macro_rules!
enum class FragmentType : u8 {
    Expr,         // $x:expr - Expression
    Ident,        // $x:ident - Identifier
    Path,         // $x:path - Path
    Type,         // $x:ty - Type
    Pattern,      // $x:pat - Pattern
    Stmt,         // $x:stmt - Statement
    Block,        // $x:block - Block
    Item,         // $x:item - Item
    Meta,         // $x:meta - Meta item
    Literal,      // $x:literal - Literal
    Lifetime,     // $x:lifetime - Lifetime
    Vis,          // $x:vis - Visibility
    Tt,           // $x:tt - Token tree
    TokenTree = Tt,  // Alias for compatibility
};

// ASTNode structure for GPU processing
struct ASTNode {
    ASTNodeType type;
    u32 token_index;      // Index in token array
    u32 parent_index;     // Index of parent node
    u32 first_child;      // Index of first child
    u32 next_sibling;     // Index of next sibling
    u32 data;             // Type-specific data (operator precedence, etc.)
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
constexpr u32 CHARS_PER_THREAD_OPT = 64;  // Optimized version constant

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