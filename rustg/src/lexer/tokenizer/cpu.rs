//! CPU reference implementation of the tokenizer

use crate::error::Result;
use crate::lexer::{Token, TokenType};

/// CPU tokenizer - reference implementation
pub fn tokenize(source: &str) -> Result<Vec<Token>> {
    // TODO: Implement full CPU tokenizer
    // This is a placeholder implementation
    let mut tokens = Vec::new();
    let mut pos = 0u32;
    let mut line = 1u32;
    let mut column = 1u32;
    
    for ch in source.chars() {
        let token_type = match ch {
            ' ' | '\t' => TokenType::Whitespace,
            '\n' => {
                line += 1;
                column = 0;
                TokenType::Newline
            }
            '(' => TokenType::LeftParen,
            ')' => TokenType::RightParen,
            '{' => TokenType::LeftBrace,
            '}' => TokenType::RightBrace,
            '[' => TokenType::LeftBracket,
            ']' => TokenType::RightBracket,
            ';' => TokenType::Semicolon,
            ',' => TokenType::Comma,
            '.' => TokenType::Dot,
            ':' => TokenType::Colon,
            '+' => TokenType::Plus,
            '-' => TokenType::Minus,
            '*' => TokenType::Star,
            '/' => TokenType::Slash,
            '=' => TokenType::Equal,
            '<' => TokenType::Less,
            '>' => TokenType::Greater,
            _ if ch.is_alphabetic() || ch == '_' => TokenType::Identifier,
            _ if ch.is_numeric() => TokenType::IntegerLiteral,
            _ => TokenType::Invalid,
        };
        
        tokens.push(Token {
            ty: token_type,
            start_pos: pos,
            length: ch.len_utf8() as u32,
            line,
            column,
        });
        
        pos += ch.len_utf8() as u32;
        column += 1;
    }
    
    tokens.push(Token {
        ty: TokenType::EOF,
        start_pos: pos,
        length: 0,
        line,
        column,
    });
    
    Ok(tokens)
}