#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <cstring>
#include "../include/gpu_types.h"

// Forward declarations for kernels to test
extern "C" {
  void launch_tokenizer_kernel(
      const char* source,
      size_t source_len,
      rustg::Token* tokens,
      uint32_t* token_count,
      uint32_t max_tokens);
  
  void launch_char_classifier_kernel(
      const char* source,
      size_t source_len,
      rustg::CharClass* classes);
}

// Test fixture for GPU tokenizer tests
class GpuTokenizerTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaSetDevice(0);
  }
  
  void TearDown() override {
    cudaDeviceSynchronize();
  }
};

// Test character classification kernel
TEST_F(GpuTokenizerTest, CharacterClassification) {
  const char* source = "abc 123 +-*/";
  size_t len = strlen(source);
  
  // Allocate device memory
  char* d_source;
  rustg::CharClass* d_classes;
  cudaMalloc(&d_source, len);
  cudaMalloc(&d_classes, len * sizeof(rustg::CharClass));
  
  // Copy source to device
  cudaMemcpy(d_source, source, len, cudaMemcpyHostToDevice);
  
  // Launch kernel
  launch_char_classifier_kernel(d_source, len, d_classes);
  
  // Copy results back
  std::vector<rustg::CharClass> classes(len);
  cudaMemcpy(classes.data(), d_classes, len * sizeof(rustg::CharClass), 
             cudaMemcpyDeviceToHost);
  
  // Verify classifications
  EXPECT_EQ(classes[0], rustg::CharClass::Letter);  // 'a'
  EXPECT_EQ(classes[1], rustg::CharClass::Letter);  // 'b'
  EXPECT_EQ(classes[2], rustg::CharClass::Letter);  // 'c'
  EXPECT_EQ(classes[3], rustg::CharClass::Whitespace); // ' '
  EXPECT_EQ(classes[4], rustg::CharClass::Digit);   // '1'
  EXPECT_EQ(classes[5], rustg::CharClass::Digit);   // '2'
  EXPECT_EQ(classes[6], rustg::CharClass::Digit);   // '3'
  EXPECT_EQ(classes[7], rustg::CharClass::Whitespace); // ' '
  EXPECT_EQ(classes[8], rustg::CharClass::Operator);  // '+'
  EXPECT_EQ(classes[9], rustg::CharClass::Operator);  // '-'
  EXPECT_EQ(classes[10], rustg::CharClass::Operator); // '*'
  EXPECT_EQ(classes[11], rustg::CharClass::Operator); // '/'
  
  cudaFree(d_source);
  cudaFree(d_classes);
}

// Test basic tokenization
TEST_F(GpuTokenizerTest, BasicTokenization) {
  const char* source = "fn main() { }";
  size_t len = strlen(source);
  uint32_t max_tokens = 100;
  
  // Allocate device memory
  char* d_source;
  rustg::Token* d_tokens;
  uint32_t* d_token_count;
  
  cudaMalloc(&d_source, len);
  cudaMalloc(&d_tokens, max_tokens * sizeof(rustg::Token));
  cudaMalloc(&d_token_count, sizeof(uint32_t));
  cudaMemset(d_token_count, 0, sizeof(uint32_t));
  
  // Copy source to device
  cudaMemcpy(d_source, source, len, cudaMemcpyHostToDevice);
  
  // Launch tokenizer kernel
  launch_tokenizer_kernel(d_source, len, d_tokens, d_token_count, max_tokens);
  
  // Copy results back
  std::vector<rustg::Token> tokens(max_tokens);
  uint32_t token_count;
  cudaMemcpy(tokens.data(), d_tokens, max_tokens * sizeof(rustg::Token), 
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&token_count, d_token_count, sizeof(uint32_t), 
             cudaMemcpyDeviceToHost);
  
  // Verify we got tokens
  EXPECT_GT(token_count, 0);
  EXPECT_LT(token_count, max_tokens);
  
  // Verify first token is an identifier (fn)
  EXPECT_EQ(tokens[0].type, rustg::TokenType::KeywordFn);
  EXPECT_EQ(tokens[0].start_pos, 0);
  EXPECT_EQ(tokens[0].length, 2);
  
  cudaFree(d_source);
  cudaFree(d_tokens);
  cudaFree(d_token_count);
}

// Test memory coalescing patterns
TEST_F(GpuTokenizerTest, MemoryCoalescingPattern) {
  // Generate aligned source data
  const size_t aligned_size = 1024;  // Multiple of warp size
  std::vector<char> source(aligned_size, 'a');
  
  char* d_source;
  rustg::CharClass* d_classes;
  
  cudaMalloc(&d_source, aligned_size);
  cudaMalloc(&d_classes, aligned_size * sizeof(rustg::CharClass));
  
  cudaMemcpy(d_source, source.data(), aligned_size, cudaMemcpyHostToDevice);
  
  // Launch with coalesced access pattern
  launch_char_classifier_kernel(d_source, aligned_size, d_classes);
  
  // Verify no errors
  cudaError_t err = cudaGetLastError();
  EXPECT_EQ(err, cudaSuccess);
  
  cudaFree(d_source);
  cudaFree(d_classes);
}

// Test warp cooperation for token boundaries
TEST_F(GpuTokenizerTest, WarpCooperationTokenBoundaries) {
  // Create source with tokens at warp boundaries
  const char* source = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
  size_t len = strlen(source);
  uint32_t max_tokens = 10;
  
  char* d_source;
  rustg::Token* d_tokens;
  uint32_t* d_token_count;
  
  cudaMalloc(&d_source, len);
  cudaMalloc(&d_tokens, max_tokens * sizeof(rustg::Token));
  cudaMalloc(&d_token_count, sizeof(uint32_t));
  cudaMemset(d_token_count, 0, sizeof(uint32_t));
  
  cudaMemcpy(d_source, source, len, cudaMemcpyHostToDevice);
  
  launch_tokenizer_kernel(d_source, len, d_tokens, d_token_count, max_tokens);
  
  std::vector<rustg::Token> tokens(max_tokens);
  uint32_t token_count;
  cudaMemcpy(tokens.data(), d_tokens, max_tokens * sizeof(rustg::Token), 
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&token_count, d_token_count, sizeof(uint32_t), 
             cudaMemcpyDeviceToHost);
  
  // Should have 3 tokens: identifier, whitespace, identifier
  EXPECT_GE(token_count, 3);
  
  cudaFree(d_source);
  cudaFree(d_tokens);
  cudaFree(d_token_count);
}

// Performance benchmark test
TEST_F(GpuTokenizerTest, PerformanceBenchmark) {
  // Generate large source file
  std::string source;
  for (int i = 0; i < 10000; ++i) {
    source += "let var_" + std::to_string(i) + " = " + std::to_string(i) + ";\n";
  }
  
  size_t len = source.length();
  uint32_t max_tokens = len / 2;  // Conservative estimate
  
  // Allocate device memory
  char* d_source;
  rustg::Token* d_tokens;
  uint32_t* d_token_count;
  
  cudaMalloc(&d_source, len);
  cudaMalloc(&d_tokens, max_tokens * sizeof(rustg::Token));
  cudaMalloc(&d_token_count, sizeof(uint32_t));
  
  cudaMemcpy(d_source, source.c_str(), len, cudaMemcpyHostToDevice);
  
  // Warm up
  launch_tokenizer_kernel(d_source, len, d_tokens, d_token_count, max_tokens);
  cudaDeviceSynchronize();
  
  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  
  const int iterations = 100;
  for (int i = 0; i < iterations; ++i) {
    launch_tokenizer_kernel(d_source, len, d_tokens, d_token_count, max_tokens);
  }
  cudaDeviceSynchronize();
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  double avg_time_us = duration.count() / (double)iterations;
  double throughput_mbs = (len / avg_time_us);  // MB/s
  
  // Log performance metrics
  std::cout << "Average tokenization time: " << avg_time_us << " us\n";
  std::cout << "Throughput: " << throughput_mbs << " MB/s\n";
  
  // Verify minimum throughput (1 GB/s = 1000 MB/s)
  EXPECT_GT(throughput_mbs, 1000.0);
  
  cudaFree(d_source);
  cudaFree(d_tokens);
  cudaFree(d_token_count);
}

// Test error handling for invalid input
TEST_F(GpuTokenizerTest, ErrorHandling) {
  // Test with null pointer
  uint32_t token_count;
  uint32_t* d_token_count;
  cudaMalloc(&d_token_count, sizeof(uint32_t));
  
  launch_tokenizer_kernel(nullptr, 0, nullptr, d_token_count, 0);
  
  cudaMemcpy(&token_count, d_token_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  EXPECT_EQ(token_count, 0);
  
  cudaFree(d_token_count);
}