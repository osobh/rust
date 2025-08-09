#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include "../include/gpu_types.h"

namespace rustg {

// Macro pattern types for testing
enum class MacroPatternType : uint8_t {
    Literal,        // println!("hello")
    Variable,       // println!("{}", x)
    Repetition,     // vec![1, 2, 3]
    Fragment,       // $x:expr, $y:ident
    Nested,         // macro inside macro
    Custom         // User-defined patterns
};

// Macro expansion result
struct MacroExpansionResult {
    Token* expanded_tokens;
    uint32_t token_count;
    uint32_t* hygiene_contexts;
    bool success;
    char error_message[256];
};

// Test structure for macro patterns
struct MacroTestCase {
    const char* name;
    const char* input;
    const char* expected_output;
    MacroPatternType pattern_type;
    uint32_t expected_token_count;
};

// External kernel functions (to be implemented)
extern "C" void launch_macro_pattern_matcher(
    const Token* tokens, uint32_t token_count,
    uint8_t* pattern_matches, uint32_t* match_positions,
    uint32_t* match_count, uint32_t max_matches);

extern "C" void launch_macro_expander(
    const Token* tokens, uint32_t token_count,
    const uint8_t* pattern_matches, const uint32_t* match_positions,
    uint32_t match_count, Token* expanded_tokens,
    uint32_t* expanded_count, uint32_t* hygiene_contexts,
    uint32_t max_expanded);

extern "C" void launch_hygiene_tracker(
    const Token* tokens, uint32_t token_count,
    uint32_t* hygiene_contexts, uint32_t current_context);

class MacroExpansionTests {
private:
    // Test cases for different macro patterns
    std::vector<MacroTestCase> test_cases = {
        // Simple literal macro
        {
            "println_literal",
            "println!(\"Hello, World!\")",
            "std::io::_print(format_args!(\"Hello, World!\\n\"))",
            MacroPatternType::Literal,
            15
        },
        
        // Variable substitution
        {
            "println_variable",
            "println!(\"{}\", x)",
            "std::io::_print(format_args!(\"{}\\n\", x))",
            MacroPatternType::Variable,
            12
        },
        
        // Vec macro with repetition
        {
            "vec_macro",
            "vec![1, 2, 3]",
            "<[_]>::into_vec(box [1, 2, 3])",
            MacroPatternType::Repetition,
            11
        },
        
        // Custom macro with fragments
        {
            "custom_macro",
            "my_macro!(foo: bar, baz: qux)",
            "expanded_custom_macro_content",
            MacroPatternType::Fragment,
            8
        },
        
        // Nested macro
        {
            "nested_macro",
            "outer!(inner!(x))",
            "outer_expanded(inner_expanded(x))",
            MacroPatternType::Nested,
            7
        },
        
        // Assert macro
        {
            "assert_macro",
            "assert!(x > 0)",
            "if !(x > 0) { panic!(\"assertion failed\") }",
            MacroPatternType::Custom,
            14
        },
        
        // Debug macro
        {
            "debug_macro",
            "dbg!(x + y)",
            "{ eprintln!(\"[{}:{}] {} = {:?}\", file!(), line!(), stringify!(x + y), x + y); x + y }",
            MacroPatternType::Custom,
            25
        }
    };
    
    // Helper to create test tokens
    void create_test_tokens(const char* input, Token* tokens, uint32_t& count) {
        count = 0;
        size_t len = strlen(input);
        size_t pos = 0;
        
        while (pos < len && count < 1000) {
            // Skip whitespace
            while (pos < len && isspace(input[pos])) pos++;
            if (pos >= len) break;
            
            Token& token = tokens[count++];
            token.start_pos = pos;
            token.line = 1;
            token.column = pos + 1;
            
            // Identify token type (simplified)
            if (input[pos] == '!') {
                token.type = TokenType::MacroBang;
                token.length = 1;
                pos++;
            } else if (input[pos] == '(' || input[pos] == ')' ||
                      input[pos] == '[' || input[pos] == ']' ||
                      input[pos] == '{' || input[pos] == '}') {
                token.type = TokenType::LeftParen; // Simplified
                token.length = 1;
                pos++;
            } else if (input[pos] == '"') {
                token.type = TokenType::StringLiteral;
                size_t start = pos;
                pos++;
                while (pos < len && input[pos] != '"') {
                    if (input[pos] == '\\') pos++;
                    pos++;
                }
                if (pos < len) pos++;
                token.length = pos - start;
            } else if (isalpha(input[pos]) || input[pos] == '_') {
                token.type = TokenType::Identifier;
                size_t start = pos;
                while (pos < len && (isalnum(input[pos]) || input[pos] == '_')) {
                    pos++;
                }
                token.length = pos - start;
            } else {
                token.type = TokenType::Plus; // Default
                token.length = 1;
                pos++;
            }
        }
    }
    
public:
    // Test pattern matching kernel
    bool test_pattern_matching() {
        std::cout << "\n=== Testing Macro Pattern Matching ===\n";
        
        // Allocate GPU memory
        Token* d_tokens;
        uint8_t* d_pattern_matches;
        uint32_t* d_match_positions;
        uint32_t* d_match_count;
        
        cudaMalloc(&d_tokens, 1000 * sizeof(Token));
        cudaMalloc(&d_pattern_matches, 100);
        cudaMalloc(&d_match_positions, 100 * sizeof(uint32_t));
        cudaMalloc(&d_match_count, sizeof(uint32_t));
        
        bool all_passed = true;
        
        for (const auto& test : test_cases) {
            // Create tokens
            Token host_tokens[1000];
            uint32_t token_count;
            create_test_tokens(test.input, host_tokens, token_count);
            
            // Copy to GPU
            cudaMemcpy(d_tokens, host_tokens, token_count * sizeof(Token),
                      cudaMemcpyHostToDevice);
            cudaMemset(d_match_count, 0, sizeof(uint32_t));
            
            // Launch pattern matcher
            launch_macro_pattern_matcher(
                d_tokens, token_count,
                d_pattern_matches, d_match_positions,
                d_match_count, 100
            );
            
            // Check results
            uint32_t match_count;
            cudaMemcpy(&match_count, d_match_count, sizeof(uint32_t),
                      cudaMemcpyDeviceToHost);
            
            bool passed = match_count > 0;
            std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] "
                     << test.name << " - Found " << match_count << " matches\n";
            
            if (!passed) all_passed = false;
        }
        
        // Cleanup
        cudaFree(d_tokens);
        cudaFree(d_pattern_matches);
        cudaFree(d_match_positions);
        cudaFree(d_match_count);
        
        return all_passed;
    }
    
    // Test macro expansion kernel
    bool test_macro_expansion() {
        std::cout << "\n=== Testing Macro Expansion ===\n";
        
        // Allocate GPU memory
        Token* d_tokens;
        Token* d_expanded;
        uint8_t* d_pattern_matches;
        uint32_t* d_match_positions;
        uint32_t* d_expanded_count;
        uint32_t* d_hygiene_contexts;
        
        cudaMalloc(&d_tokens, 1000 * sizeof(Token));
        cudaMalloc(&d_expanded, 5000 * sizeof(Token));
        cudaMalloc(&d_pattern_matches, 100);
        cudaMalloc(&d_match_positions, 100 * sizeof(uint32_t));
        cudaMalloc(&d_expanded_count, sizeof(uint32_t));
        cudaMalloc(&d_hygiene_contexts, 5000 * sizeof(uint32_t));
        
        bool all_passed = true;
        
        for (const auto& test : test_cases) {
            // Create tokens
            Token host_tokens[1000];
            uint32_t token_count;
            create_test_tokens(test.input, host_tokens, token_count);
            
            // Copy to GPU
            cudaMemcpy(d_tokens, host_tokens, token_count * sizeof(Token),
                      cudaMemcpyHostToDevice);
            cudaMemset(d_expanded_count, 0, sizeof(uint32_t));
            
            // Simulate pattern matches
            uint8_t pattern_matches[100] = {1}; // First token is macro
            uint32_t match_positions[100] = {0};
            uint32_t match_count = 1;
            
            cudaMemcpy(d_pattern_matches, pattern_matches, 100,
                      cudaMemcpyHostToDevice);
            cudaMemcpy(d_match_positions, match_positions, 100 * sizeof(uint32_t),
                      cudaMemcpyHostToDevice);
            
            // Launch expander
            launch_macro_expander(
                d_tokens, token_count,
                d_pattern_matches, d_match_positions, match_count,
                d_expanded, d_expanded_count, d_hygiene_contexts, 5000
            );
            
            // Check results
            uint32_t expanded_count;
            cudaMemcpy(&expanded_count, d_expanded_count, sizeof(uint32_t),
                      cudaMemcpyDeviceToHost);
            
            bool passed = expanded_count >= test.expected_token_count;
            std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] "
                     << test.name << " - Expanded to " << expanded_count << " tokens"
                     << " (expected " << test.expected_token_count << ")\n";
            
            if (!passed) all_passed = false;
        }
        
        // Cleanup
        cudaFree(d_tokens);
        cudaFree(d_expanded);
        cudaFree(d_pattern_matches);
        cudaFree(d_match_positions);
        cudaFree(d_expanded_count);
        cudaFree(d_hygiene_contexts);
        
        return all_passed;
    }
    
    // Test hygiene tracking
    bool test_hygiene_tracking() {
        std::cout << "\n=== Testing Hygiene Context Tracking ===\n";
        
        // Test that expanded tokens get unique hygiene contexts
        Token* d_tokens;
        uint32_t* d_hygiene_contexts;
        
        cudaMalloc(&d_tokens, 100 * sizeof(Token));
        cudaMalloc(&d_hygiene_contexts, 100 * sizeof(uint32_t));
        
        // Create test tokens
        Token host_tokens[10];
        for (int i = 0; i < 10; ++i) {
            host_tokens[i].type = TokenType::Identifier;
            host_tokens[i].start_pos = i * 5;
            host_tokens[i].length = 3;
        }
        
        cudaMemcpy(d_tokens, host_tokens, 10 * sizeof(Token),
                  cudaMemcpyHostToDevice);
        cudaMemset(d_hygiene_contexts, 0, 100 * sizeof(uint32_t));
        
        // Launch hygiene tracker
        launch_hygiene_tracker(d_tokens, 10, d_hygiene_contexts, 1);
        
        // Check results
        uint32_t host_contexts[10];
        cudaMemcpy(host_contexts, d_hygiene_contexts, 10 * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost);
        
        bool passed = true;
        for (int i = 0; i < 10; ++i) {
            if (host_contexts[i] == 0) {
                passed = false;
                break;
            }
        }
        
        std::cout << "  [" << (passed ? "PASS" : "FAIL") 
                 << "] Hygiene contexts assigned\n";
        
        // Cleanup
        cudaFree(d_tokens);
        cudaFree(d_hygiene_contexts);
        
        return passed;
    }
    
    // Run all tests
    bool run_all_tests() {
        std::cout << "\n🧪 Testing GPU Macro Expansion System\n";
        std::cout << "=====================================\n";
        
        bool pattern_pass = test_pattern_matching();
        bool expansion_pass = test_macro_expansion();
        bool hygiene_pass = test_hygiene_tracking();
        
        std::cout << "\n=== Test Summary ===\n";
        std::cout << "Pattern Matching: " << (pattern_pass ? "✅ PASS" : "❌ FAIL") << "\n";
        std::cout << "Macro Expansion:  " << (expansion_pass ? "✅ PASS" : "❌ FAIL") << "\n";
        std::cout << "Hygiene Tracking: " << (hygiene_pass ? "✅ PASS" : "❌ FAIL") << "\n";
        
        bool all_passed = pattern_pass && expansion_pass && hygiene_pass;
        std::cout << "\nOverall: " << (all_passed ? "✅ ALL TESTS PASSED" : "❌ SOME TESTS FAILED") << "\n";
        
        return all_passed;
    }
};

} // namespace rustg

// Main test runner
int main() {
    rustg::MacroExpansionTests tests;
    return tests.run_all_tests() ? 0 : 1;
}