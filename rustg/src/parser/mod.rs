//! GPU-accelerated parsing

use crate::error::Result;
use crate::lexer::Token;

/// AST Node type (matches gpu_types.h)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ASTNodeType {
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
}

/// AST Node
#[derive(Debug, Clone)]
pub struct ASTNode {
    pub ty: ASTNodeType,
    pub token_index: Option<usize>,
    pub children: Vec<ASTNode>,
}

/// Parse tokens into an AST on GPU
pub fn parse_gpu(tokens: &[Token]) -> Result<ASTNode> {
    todo!("GPU parser not yet implemented")
}

/// Parse tokens into an AST on CPU (reference implementation)
pub fn parse_cpu(tokens: &[Token]) -> Result<ASTNode> {
    todo!("CPU parser not yet implemented")
}