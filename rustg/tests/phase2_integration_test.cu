#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <cassert>
#include "../include/gpu_types.h"

using namespace std::chrono;

namespace rustg {

// External kernel functions
extern "C" void launch_tokenizer_optimized(
    const char* source, size_t source_len,
    Token* tokens, uint32_t* token_count, uint32_t max_tokens);

extern "C" void launch_macro_pattern_matcher(
    const Token* tokens, uint32_t token_count,
    const char* source, uint32_t source_len,
    uint8_t* pattern_matches, uint32_t* match_positions,
    uint32_t* match_count, uint32_t max_matches);

extern "C" void launch_macro_expander(
    const Token* tokens, uint32_t token_count,
    const uint8_t* pattern_matches, const uint32_t* match_positions,
    uint32_t match_count, Token* expanded_tokens,
    uint32_t* expanded_count, uint32_t* hygiene_contexts,
    uint32_t max_expanded);

extern "C" void launch_macro_pipeline(
    const Token* input_tokens, uint32_t input_count,
    Token* output_tokens, uint32_t* output_count,
    uint32_t* hygiene_contexts, uint32_t max_output);

class Phase2IntegrationTests {
private:
    struct TestCase {
        std::string name;
        std::string input;
        std::string expected_pattern;
        bool should_expand;
        uint32_t min_tokens;
    };
    
    std::vector<TestCase> test_cases = {
        // Basic macro tests
        {
            "simple_println",
            "fn main() { println!(\"Hello, World!\"); }",
            "std::io::_print",
            true,
            20
        },
        
        // Vec macro with multiple elements
        {
            "vec_multiple",
            "let v = vec![1, 2, 3, 4, 5];",
            "into_vec",
            true,
            15
        },
        
        // Nested macros
        {
            "nested_macros",
            "println!(\"Result: {}\", vec![1, 2].len());",
            "format_args",
            true,
            25
        },
        
        // Assert with expression
        {
            "assert_expression",
            "assert!(x > 0 && y < 10);",
            "panic",
            true,
            15
        },
        
        // Format macro
        {
            "format_macro",
            "let s = format!(\"x = {}, y = {}\", x, y);",
            "format_args",
            true,
            20
        },
        
        // Custom macro (should be detected but not expanded)
        {
            "custom_macro",
            "my_custom_macro!(foo, bar, baz);",
            "",
            false,
            8
        },
        
        // Multiple macros in sequence
        {
            "multiple_macros",
            "println!(\"Start\"); dbg!(x); println!(\"End\");",
            "_print",
            true,
            30
        },
        
        // Macro in expression position
        {
            "expr_position",
            "let result = vec![1, 2, 3].iter().map(|x| x * 2).collect();",
            "into_vec",
            true,
            25
        }
    };
    
public:
    // Test Phase 1 -> Phase 2 integration
    bool test_tokenization_to_macro_detection() {
        std::cout << "\n=== Testing Tokenization -> Macro Detection ===\n";
        
        bool all_passed = true;
        
        for (const auto& test : test_cases) {
            // Allocate GPU memory
            char* d_source;
            Token* d_tokens;
            uint32_t* d_token_count;
            uint8_t* d_pattern_matches;
            uint32_t* d_match_positions;
            uint32_t* d_match_count;
            
            size_t source_len = test.input.length();
            uint32_t max_tokens = source_len * 2;
            
            cudaMalloc(&d_source, source_len + 1);
            cudaMalloc(&d_tokens, max_tokens * sizeof(Token));
            cudaMalloc(&d_token_count, sizeof(uint32_t));
            cudaMalloc(&d_pattern_matches, max_tokens);
            cudaMalloc(&d_match_positions, 100 * sizeof(uint32_t));
            cudaMalloc(&d_match_count, sizeof(uint32_t));
            
            // Copy source to GPU
            cudaMemcpy(d_source, test.input.c_str(), source_len + 1, cudaMemcpyHostToDevice);
            cudaMemset(d_token_count, 0, sizeof(uint32_t));
            cudaMemset(d_match_count, 0, sizeof(uint32_t));
            
            // Phase 1: Tokenization
            launch_tokenizer_optimized(d_source, source_len, 
                                     d_tokens, d_token_count, max_tokens);
            
            // Get token count
            uint32_t token_count;
            cudaMemcpy(&token_count, d_token_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            // Phase 2: Macro detection
            launch_macro_pattern_matcher(d_tokens, token_count, d_source, source_len,
                                        d_pattern_matches, d_match_positions,
                                        d_match_count, 100);
            
            // Check results
            uint32_t match_count;
            cudaMemcpy(&match_count, d_match_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            bool passed = (test.should_expand && match_count > 0) || 
                         (!test.should_expand && match_count == 0);
            
            std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] "
                     << test.name << " - Tokens: " << token_count 
                     << ", Macros: " << match_count << "\n";
            
            if (!passed) all_passed = false;
            
            // Cleanup
            cudaFree(d_source);
            cudaFree(d_tokens);
            cudaFree(d_token_count);
            cudaFree(d_pattern_matches);
            cudaFree(d_match_positions);
            cudaFree(d_match_count);
        }
        
        return all_passed;
    }
    
    // Test complete pipeline
    bool test_full_macro_pipeline() {
        std::cout << "\n=== Testing Full Macro Pipeline ===\n";
        
        bool all_passed = true;
        
        for (const auto& test : test_cases) {
            if (!test.should_expand) continue;
            
            // Allocate GPU memory
            char* d_source;
            Token* d_tokens;
            Token* d_expanded;
            uint32_t* d_token_count;
            uint32_t* d_expanded_count;
            uint32_t* d_hygiene;
            
            size_t source_len = test.input.length();
            uint32_t max_tokens = source_len * 2;
            uint32_t max_expanded = max_tokens * 3;
            
            cudaMalloc(&d_source, source_len + 1);
            cudaMalloc(&d_tokens, max_tokens * sizeof(Token));
            cudaMalloc(&d_expanded, max_expanded * sizeof(Token));
            cudaMalloc(&d_token_count, sizeof(uint32_t));
            cudaMalloc(&d_expanded_count, sizeof(uint32_t));
            cudaMalloc(&d_hygiene, max_expanded * sizeof(uint32_t));
            
            // Copy source and tokenize
            cudaMemcpy(d_source, test.input.c_str(), source_len + 1, cudaMemcpyHostToDevice);
            cudaMemset(d_token_count, 0, sizeof(uint32_t));
            cudaMemset(d_expanded_count, 0, sizeof(uint32_t));
            
            // Tokenize
            launch_tokenizer_optimized(d_source, source_len,
                                     d_tokens, d_token_count, max_tokens);
            
            uint32_t token_count;
            cudaMemcpy(&token_count, d_token_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            // Run full pipeline
            launch_macro_pipeline(d_tokens, token_count,
                                d_expanded, d_expanded_count,
                                d_hygiene, max_expanded);
            
            // Check results
            uint32_t expanded_count;
            cudaMemcpy(&expanded_count, d_expanded_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            bool passed = expanded_count >= test.min_tokens;
            
            std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] "
                     << test.name << " - Original: " << token_count
                     << ", Expanded: " << expanded_count << "\n";
            
            if (!passed) all_passed = false;
            
            // Cleanup
            cudaFree(d_source);
            cudaFree(d_tokens);
            cudaFree(d_expanded);
            cudaFree(d_token_count);
            cudaFree(d_expanded_count);
            cudaFree(d_hygiene);
        }
        
        return all_passed;
    }
    
    // Test hygiene preservation
    bool test_hygiene_contexts() {
        std::cout << "\n=== Testing Hygiene Context Preservation ===\n";
        
        // Test that expanded macros get unique hygiene contexts
        const char* source = "let x = 1; println!(\"{}\", x); let x = 2;";
        size_t source_len = strlen(source);
        
        // GPU allocations
        char* d_source;
        Token* d_tokens;
        Token* d_expanded;
        uint32_t* d_token_count;
        uint32_t* d_expanded_count;
        uint32_t* d_hygiene;
        
        uint32_t max_tokens = 100;
        uint32_t max_expanded = 300;
        
        cudaMalloc(&d_source, source_len + 1);
        cudaMalloc(&d_tokens, max_tokens * sizeof(Token));
        cudaMalloc(&d_expanded, max_expanded * sizeof(Token));
        cudaMalloc(&d_token_count, sizeof(uint32_t));
        cudaMalloc(&d_expanded_count, sizeof(uint32_t));
        cudaMalloc(&d_hygiene, max_expanded * sizeof(uint32_t));
        
        cudaMemcpy(d_source, source, source_len + 1, cudaMemcpyHostToDevice);
        cudaMemset(d_token_count, 0, sizeof(uint32_t));
        cudaMemset(d_expanded_count, 0, sizeof(uint32_t));
        cudaMemset(d_hygiene, 0, max_expanded * sizeof(uint32_t));
        
        // Process
        launch_tokenizer_optimized(d_source, source_len,
                                 d_tokens, d_token_count, max_tokens);
        
        uint32_t token_count;
        cudaMemcpy(&token_count, d_token_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        launch_macro_pipeline(d_tokens, token_count,
                            d_expanded, d_expanded_count,
                            d_hygiene, max_expanded);
        
        // Check hygiene contexts
        uint32_t expanded_count;
        cudaMemcpy(&expanded_count, d_expanded_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        std::vector<uint32_t> hygiene_contexts(expanded_count);
        cudaMemcpy(hygiene_contexts.data(), d_hygiene, 
                  expanded_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        // Verify unique contexts for macro-expanded tokens
        bool has_unique = false;
        for (uint32_t i = 1; i < expanded_count; ++i) {
            if (hygiene_contexts[i] != hygiene_contexts[0]) {
                has_unique = true;
                break;
            }
        }
        
        std::cout << "  [" << (has_unique ? "PASS" : "FAIL") 
                 << "] Hygiene contexts are unique\n";
        
        // Cleanup
        cudaFree(d_source);
        cudaFree(d_tokens);
        cudaFree(d_expanded);
        cudaFree(d_token_count);
        cudaFree(d_expanded_count);
        cudaFree(d_hygiene);
        
        return has_unique;
    }
    
    // Performance benchmark
    bool test_performance() {
        std::cout << "\n=== Performance Benchmarks ===\n";
        
        // Generate large input with many macros
        std::string large_input;
        for (int i = 0; i < 1000; ++i) {
            large_input += "println!(\"Value: {}\", " + std::to_string(i) + ");\n";
            large_input += "let v" + std::to_string(i) + " = vec![" + std::to_string(i) + "];\n";
        }
        
        size_t source_len = large_input.length();
        uint32_t max_tokens = source_len * 2;
        uint32_t max_expanded = max_tokens * 3;
        
        // Allocate
        char* d_source;
        Token* d_tokens;
        Token* d_expanded;
        uint32_t* d_token_count;
        uint32_t* d_expanded_count;
        uint32_t* d_hygiene;
        
        cudaMalloc(&d_source, source_len + 1);
        cudaMalloc(&d_tokens, max_tokens * sizeof(Token));
        cudaMalloc(&d_expanded, max_expanded * sizeof(Token));
        cudaMalloc(&d_token_count, sizeof(uint32_t));
        cudaMalloc(&d_expanded_count, sizeof(uint32_t));
        cudaMalloc(&d_hygiene, max_expanded * sizeof(uint32_t));
        
        cudaMemcpy(d_source, large_input.c_str(), source_len + 1, cudaMemcpyHostToDevice);
        
        // Warm-up
        for (int i = 0; i < 5; ++i) {
            cudaMemset(d_token_count, 0, sizeof(uint32_t));
            launch_tokenizer_optimized(d_source, source_len,
                                     d_tokens, d_token_count, max_tokens);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        const int num_runs = 100;
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            cudaMemset(d_token_count, 0, sizeof(uint32_t));
            cudaMemset(d_expanded_count, 0, sizeof(uint32_t));
            
            launch_tokenizer_optimized(d_source, source_len,
                                     d_tokens, d_token_count, max_tokens);
            
            uint32_t token_count;
            cudaMemcpy(&token_count, d_token_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            launch_macro_pipeline(d_tokens, token_count,
                                d_expanded, d_expanded_count,
                                d_hygiene, max_expanded);
        }
        cudaDeviceSynchronize();
        
        auto end = high_resolution_clock::now();
        duration<double, std::milli> elapsed = end - start;
        
        double avg_time = elapsed.count() / num_runs;
        double throughput = (source_len / 1024.0) / (avg_time / 1000.0); // KB/s
        
        std::cout << "  Source size: " << source_len / 1024 << " KB\n";
        std::cout << "  Average time: " << avg_time << " ms\n";
        std::cout << "  Throughput: " << throughput << " KB/s\n";
        std::cout << "  Macros/second: " << (2000.0 / avg_time * 1000) << "\n";
        
        bool passed = throughput > 500.0; // Target: >500 KB/s
        std::cout << "  [" << (passed ? "PASS" : "FAIL") 
                 << "] Performance target met\n";
        
        // Cleanup
        cudaFree(d_source);
        cudaFree(d_tokens);
        cudaFree(d_expanded);
        cudaFree(d_token_count);
        cudaFree(d_expanded_count);
        cudaFree(d_hygiene);
        
        return passed;
    }
    
    // Run all tests
    bool run_all_tests() {
        std::cout << "\n🧪 Phase 2 Integration Tests\n";
        std::cout << "================================\n";
        
        bool tokenization_pass = test_tokenization_to_macro_detection();
        bool pipeline_pass = test_full_macro_pipeline();
        bool hygiene_pass = test_hygiene_contexts();
        bool performance_pass = test_performance();
        
        std::cout << "\n=== Test Summary ===\n";
        std::cout << "Tokenization->Detection: " << (tokenization_pass ? "✅" : "❌") << "\n";
        std::cout << "Full Pipeline: " << (pipeline_pass ? "✅" : "❌") << "\n";
        std::cout << "Hygiene Preservation: " << (hygiene_pass ? "✅" : "❌") << "\n";
        std::cout << "Performance: " << (performance_pass ? "✅" : "❌") << "\n";
        
        bool all_passed = tokenization_pass && pipeline_pass && 
                         hygiene_pass && performance_pass;
        
        std::cout << "\nOverall: " << (all_passed ? "✅ ALL TESTS PASSED" : "❌ SOME TESTS FAILED") << "\n";
        
        return all_passed;
    }
};

} // namespace rustg

int main() {
    rustg::Phase2IntegrationTests tests;
    return tests.run_all_tests() ? 0 : 1;
}