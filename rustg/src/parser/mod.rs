//! GPU-accelerated parsing

use crate::error::Result;
use crate::lexer::Token;

/// AST Node type (matches gpu_types.h)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ASTNodeType {
    /// Invalid or unrecognized node
    Invalid = 0,
    /// Root program node containing all top-level items
    Program,
    /// Function declaration or definition
    Function,
    /// Function parameter
    Parameter,
    /// Code block enclosed in braces
    Block,
    /// Statement (expression with semicolon)
    Statement,
    /// Expression (produces a value)
    Expression,
    /// Binary operation (e.g., a + b)
    BinaryOp,
    /// Unary operation (e.g., !x, -y)
    UnaryOp,
    /// Literal value (number, string, etc.)
    Literal,
    /// Variable or function identifier
    Identifier,
    /// Function call expression
    Call,
    /// Struct definition
    StructDef,
    /// Enum definition
    EnumDef,
    /// Implementation block
    ImplBlock,
    /// Trait definition
    TraitDef,
}

/// Abstract Syntax Tree node representing parsed code structure
#[derive(Debug, Clone)]
pub struct ASTNode {
    /// Type of this AST node
    pub ty: ASTNodeType,
    /// Index of the associated token in the token stream
    pub token_index: Option<usize>,
    /// Child nodes of this AST node
    pub children: Vec<ASTNode>,
}

/// AST type alias for consistency
pub type AST = ASTNode;

/// Typed AST after type checking and semantic analysis
#[derive(Debug, Clone)]
pub struct TypedAST {
    /// Root AST node
    pub ast: ASTNode,
    /// Type information for nodes
    pub type_info: Vec<String>,
}

impl TypedAST {
    /// Create a TypedAST from an untyped AST
    pub fn from_ast(ast: AST) -> Self {
        Self {
            ast,
            type_info: vec!["inferred_types".to_string()],
        }
    }
    
    /// Count the total number of nodes in the AST
    pub fn node_count(&self) -> usize {
        self.count_nodes(&self.ast)
    }
    
    fn count_nodes(&self, node: &ASTNode) -> usize {
        1 + node.children.iter().map(|child| self.count_nodes(child)).sum::<usize>()
    }
}

/// Parse tokens into an AST on GPU
pub fn parse_gpu(tokens: &[Token]) -> Result<ASTNode> {
    // For now, fall back to CPU implementation
    parse_cpu(tokens)
}

/// Parse tokens into an AST on CPU (reference implementation)
pub fn parse_cpu(tokens: &[Token]) -> Result<ASTNode> {
    // Simple placeholder parser - just creates a basic AST
    let root = ASTNode {
        ty: ASTNodeType::Program,
        token_index: None,
        children: Vec::new(),
    };
    
    // For now, just return a valid AST without full parsing
    Ok(root)
}