//! Tests for the tokenizer (CPU and GPU implementations)

use rustg::lexer::{Token, TokenType};

#[test]
fn test_cpu_tokenizer_basic() {
    let source = "fn main() { println!(\"Hello, world!\"); }";
    let tokens = rustg::lexer::tokenize_cpu(source).expect("Failed to tokenize");
    
    // Verify we got tokens
    assert!(!tokens.is_empty());
    
    // Verify EOF token at the end
    assert_eq!(tokens.last().unwrap().ty, TokenType::EOF);
}

#[test]
fn test_cpu_tokenizer_empty_source() {
    let source = "";
    let tokens = rustg::lexer::tokenize_cpu(source).expect("Failed to tokenize");
    
    // Should have just EOF token
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].ty, TokenType::EOF);
}

#[test]
fn test_cpu_tokenizer_whitespace() {
    let source = "   \t\n  ";
    let tokens = rustg::lexer::tokenize_cpu(source).expect("Failed to tokenize");
    
    // Should have whitespace tokens and EOF
    assert!(tokens.len() > 1);
    assert_eq!(tokens.last().unwrap().ty, TokenType::EOF);
}

#[test]
fn test_cpu_tokenizer_operators() {
    let source = "+ - * / = < >";
    let tokens = rustg::lexer::tokenize_cpu(source).expect("Failed to tokenize");
    
    // Filter out whitespace
    let ops: Vec<_> = tokens.iter()
        .filter(|t| !matches!(t.ty, TokenType::Whitespace | TokenType::EOF))
        .collect();
    
    assert_eq!(ops.len(), 7);
    assert_eq!(ops[0].ty, TokenType::Plus);
    assert_eq!(ops[1].ty, TokenType::Minus);
    assert_eq!(ops[2].ty, TokenType::Star);
    assert_eq!(ops[3].ty, TokenType::Slash);
    assert_eq!(ops[4].ty, TokenType::Equal);
    assert_eq!(ops[5].ty, TokenType::Less);
    assert_eq!(ops[6].ty, TokenType::Greater);
}

#[test]
fn test_cpu_tokenizer_delimiters() {
    let source = "()[]{};,.";
    let tokens = rustg::lexer::tokenize_cpu(source).expect("Failed to tokenize");
    
    // Filter out EOF
    let delims: Vec<_> = tokens.iter()
        .filter(|t| t.ty != TokenType::EOF)
        .collect();
    
    assert_eq!(delims.len(), 9);
    assert_eq!(delims[0].ty, TokenType::LeftParen);
    assert_eq!(delims[1].ty, TokenType::RightParen);
    assert_eq!(delims[2].ty, TokenType::LeftBracket);
    assert_eq!(delims[3].ty, TokenType::RightBracket);
    assert_eq!(delims[4].ty, TokenType::LeftBrace);
    assert_eq!(delims[5].ty, TokenType::RightBrace);
    assert_eq!(delims[6].ty, TokenType::Semicolon);
    assert_eq!(delims[7].ty, TokenType::Comma);
    assert_eq!(delims[8].ty, TokenType::Dot);
}

#[test]
fn test_cpu_tokenizer_line_column_tracking() {
    let source = "a\nb\nc";
    let tokens = rustg::lexer::tokenize_cpu(source).expect("Failed to tokenize");
    
    // Filter out newlines and EOF
    let ids: Vec<_> = tokens.iter()
        .filter(|t| t.ty == TokenType::Identifier)
        .collect();
    
    assert_eq!(ids.len(), 3);
    assert_eq!(ids[0].line, 1);
    assert_eq!(ids[0].column, 1);
    assert_eq!(ids[1].line, 2);
    assert_eq!(ids[1].column, 1);
    assert_eq!(ids[2].line, 3);
    assert_eq!(ids[2].column, 1);
}

#[test]
fn test_cpu_tokenizer_complex_expression() {
    let source = "let x = 42 + 3 * (10 - 5);";
    let tokens = rustg::lexer::tokenize_cpu(source).expect("Failed to tokenize");
    
    // Verify we have reasonable number of tokens
    assert!(tokens.len() > 10);
    
    // Verify specific tokens are present (not checking exact positions)
    let has_identifier = tokens.iter().any(|t| t.ty == TokenType::Identifier);
    let has_number = tokens.iter().any(|t| t.ty == TokenType::IntegerLiteral);
    let has_plus = tokens.iter().any(|t| t.ty == TokenType::Plus);
    let has_star = tokens.iter().any(|t| t.ty == TokenType::Star);
    let has_lparen = tokens.iter().any(|t| t.ty == TokenType::LeftParen);
    let has_rparen = tokens.iter().any(|t| t.ty == TokenType::RightParen);
    let has_semicolon = tokens.iter().any(|t| t.ty == TokenType::Semicolon);
    
    assert!(has_identifier);
    assert!(has_number);
    assert!(has_plus);
    assert!(has_star);
    assert!(has_lparen);
    assert!(has_rparen);
    assert!(has_semicolon);
}

// GPU tokenizer tests (will fail until implemented)
#[test]
#[ignore] // Remove ignore when GPU tokenizer is implemented
fn test_gpu_tokenizer_basic() {
    let source = "fn main() { }";
    let gpu_tokens = rustg::lexer::tokenize_gpu(source).expect("Failed to tokenize on GPU");
    let cpu_tokens = rustg::lexer::tokenize_cpu(source).expect("Failed to tokenize on CPU");
    
    // GPU and CPU should produce identical results
    assert_eq!(gpu_tokens.len(), cpu_tokens.len());
    for (gpu_token, cpu_token) in gpu_tokens.iter().zip(cpu_tokens.iter()) {
        assert_eq!(gpu_token.ty, cpu_token.ty);
        assert_eq!(gpu_token.start_pos, cpu_token.start_pos);
        assert_eq!(gpu_token.length, cpu_token.length);
    }
}

#[test]
#[ignore] // Remove ignore when GPU tokenizer is implemented
fn test_gpu_tokenizer_performance() {
    // Generate a large source file
    let mut source = String::new();
    for i in 0..10000 {
        source.push_str(&format!("let var_{} = {};\n", i, i));
    }
    
    // Time CPU version
    let cpu_start = std::time::Instant::now();
    let cpu_tokens = rustg::lexer::tokenize_cpu(&source).expect("CPU tokenization failed");
    let cpu_time = cpu_start.elapsed();
    
    // Time GPU version
    let gpu_start = std::time::Instant::now();
    let gpu_tokens = rustg::lexer::tokenize_gpu(&source).expect("GPU tokenization failed");
    let gpu_time = gpu_start.elapsed();
    
    // Verify correctness
    assert_eq!(gpu_tokens.len(), cpu_tokens.len());
    
    // Check speedup (should be >10x for large files)
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("GPU speedup: {:.2}x", speedup);
    assert!(speedup > 10.0, "GPU should be at least 10x faster");
}