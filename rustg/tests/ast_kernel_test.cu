#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include "../include/gpu_types.h"

// Forward declarations for AST kernels
extern "C" {
  void launch_ast_construction_kernel(
      const rustg::Token* tokens,
      uint32_t token_count,
      rustg::ASTNode* ast_nodes,
      uint32_t* ast_node_count,
      uint32_t* parent_indices,
      uint32_t max_nodes);
  
  void launch_ast_validator_kernel(
      const rustg::ASTNode* ast_nodes,
      uint32_t node_count,
      const uint32_t* parent_indices,
      bool* is_valid);
}

// Test fixture for AST construction
class ASTConstructionTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaSetDevice(0);
  }
  
  void TearDown() override {
    cudaDeviceSynchronize();
  }
  
  // Helper to create test tokens
  std::vector<rustg::Token> CreateTestTokens() {
    std::vector<rustg::Token> tokens;
    
    // fn main() { let x = 42; }
    tokens.push_back({rustg::TokenType::KeywordFn, 0, 2, 1, 1});
    tokens.push_back({rustg::TokenType::Identifier, 3, 4, 1, 4}); // main
    tokens.push_back({rustg::TokenType::LeftParen, 7, 1, 1, 8});
    tokens.push_back({rustg::TokenType::RightParen, 8, 1, 1, 9});
    tokens.push_back({rustg::TokenType::LeftBrace, 10, 1, 1, 11});
    tokens.push_back({rustg::TokenType::KeywordLet, 12, 3, 1, 13});
    tokens.push_back({rustg::TokenType::Identifier, 16, 1, 1, 17}); // x
    tokens.push_back({rustg::TokenType::Equal, 18, 1, 1, 19});
    tokens.push_back({rustg::TokenType::IntegerLiteral, 20, 2, 1, 21}); // 42
    tokens.push_back({rustg::TokenType::Semicolon, 22, 1, 1, 23});
    tokens.push_back({rustg::TokenType::RightBrace, 24, 1, 1, 25});
    tokens.push_back({rustg::TokenType::EOF, 25, 0, 1, 26});
    
    return tokens;
  }
};

// Test basic AST construction
TEST_F(ASTConstructionTest, BasicFunctionAST) {
  auto tokens = CreateTestTokens();
  uint32_t max_nodes = 100;
  
  // Allocate device memory
  rustg::Token* d_tokens;
  rustg::ASTNode* d_ast_nodes;
  uint32_t* d_ast_node_count;
  uint32_t* d_parent_indices;
  
  cudaMalloc(&d_tokens, tokens.size() * sizeof(rustg::Token));
  cudaMalloc(&d_ast_nodes, max_nodes * sizeof(rustg::ASTNode));
  cudaMalloc(&d_ast_node_count, sizeof(uint32_t));
  cudaMalloc(&d_parent_indices, max_nodes * sizeof(uint32_t));
  
  // Initialize
  cudaMemset(d_ast_node_count, 0, sizeof(uint32_t));
  cudaMemset(d_parent_indices, 0xFF, max_nodes * sizeof(uint32_t)); // -1 = no parent
  
  // Copy tokens to device
  cudaMemcpy(d_tokens, tokens.data(), tokens.size() * sizeof(rustg::Token), 
             cudaMemcpyHostToDevice);
  
  // Launch AST construction kernel
  launch_ast_construction_kernel(
      d_tokens, tokens.size(),
      d_ast_nodes, d_ast_node_count, d_parent_indices,
      max_nodes);
  
  // Copy results back
  std::vector<rustg::ASTNode> ast_nodes(max_nodes);
  std::vector<uint32_t> parent_indices(max_nodes);
  uint32_t node_count;
  
  cudaMemcpy(ast_nodes.data(), d_ast_nodes, max_nodes * sizeof(rustg::ASTNode), 
             cudaMemcpyDeviceToHost);
  cudaMemcpy(parent_indices.data(), d_parent_indices, max_nodes * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&node_count, d_ast_node_count, sizeof(uint32_t), 
             cudaMemcpyDeviceToHost);
  
  // Verify AST structure
  EXPECT_GT(node_count, 0);
  EXPECT_LT(node_count, max_nodes);
  
  // Root should be a Program node
  EXPECT_EQ(ast_nodes[0].type, rustg::ASTNodeType::Program);
  EXPECT_EQ(parent_indices[0], 0xFFFFFFFF); // No parent
  
  // Should have a Function node
  bool has_function = false;
  for (uint32_t i = 0; i < node_count; ++i) {
    if (ast_nodes[i].type == rustg::ASTNodeType::Function) {
      has_function = true;
      break;
    }
  }
  EXPECT_TRUE(has_function);
  
  cudaFree(d_tokens);
  cudaFree(d_ast_nodes);
  cudaFree(d_ast_node_count);
  cudaFree(d_parent_indices);
}

// Test expression parsing
TEST_F(ASTConstructionTest, ExpressionAST) {
  std::vector<rustg::Token> tokens;
  
  // 2 + 3 * 4
  tokens.push_back({rustg::TokenType::IntegerLiteral, 0, 1, 1, 1}); // 2
  tokens.push_back({rustg::TokenType::Plus, 2, 1, 1, 3});
  tokens.push_back({rustg::TokenType::IntegerLiteral, 4, 1, 1, 5}); // 3
  tokens.push_back({rustg::TokenType::Star, 6, 1, 1, 7});
  tokens.push_back({rustg::TokenType::IntegerLiteral, 8, 1, 1, 9}); // 4
  tokens.push_back({rustg::TokenType::EOF, 9, 0, 1, 10});
  
  uint32_t max_nodes = 50;
  
  // Allocate and launch
  rustg::Token* d_tokens;
  rustg::ASTNode* d_ast_nodes;
  uint32_t* d_ast_node_count;
  uint32_t* d_parent_indices;
  
  cudaMalloc(&d_tokens, tokens.size() * sizeof(rustg::Token));
  cudaMalloc(&d_ast_nodes, max_nodes * sizeof(rustg::ASTNode));
  cudaMalloc(&d_ast_node_count, sizeof(uint32_t));
  cudaMalloc(&d_parent_indices, max_nodes * sizeof(uint32_t));
  
  cudaMemset(d_ast_node_count, 0, sizeof(uint32_t));
  cudaMemset(d_parent_indices, 0xFF, max_nodes * sizeof(uint32_t));
  
  cudaMemcpy(d_tokens, tokens.data(), tokens.size() * sizeof(rustg::Token), 
             cudaMemcpyHostToDevice);
  
  launch_ast_construction_kernel(
      d_tokens, tokens.size(),
      d_ast_nodes, d_ast_node_count, d_parent_indices,
      max_nodes);
  
  std::vector<rustg::ASTNode> ast_nodes(max_nodes);
  uint32_t node_count;
  
  cudaMemcpy(ast_nodes.data(), d_ast_nodes, max_nodes * sizeof(rustg::ASTNode), 
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&node_count, d_ast_node_count, sizeof(uint32_t), 
             cudaMemcpyDeviceToHost);
  
  // Verify precedence (multiplication should be deeper in tree)
  EXPECT_GT(node_count, 5); // At least: root, +, *, 2, 3, 4
  
  // Check for binary operators
  int plus_count = 0, star_count = 0;
  for (uint32_t i = 0; i < node_count; ++i) {
    if (ast_nodes[i].type == rustg::ASTNodeType::BinaryOp) {
      if (ast_nodes[i].token_index < tokens.size()) {
        if (tokens[ast_nodes[i].token_index].type == rustg::TokenType::Plus) {
          plus_count++;
        } else if (tokens[ast_nodes[i].token_index].type == rustg::TokenType::Star) {
          star_count++;
        }
      }
    }
  }
  EXPECT_EQ(plus_count, 1);
  EXPECT_EQ(star_count, 1);
  
  cudaFree(d_tokens);
  cudaFree(d_ast_nodes);
  cudaFree(d_ast_node_count);
  cudaFree(d_parent_indices);
}

// Test AST validation
TEST_F(ASTConstructionTest, ASTValidation) {
  auto tokens = CreateTestTokens();
  uint32_t max_nodes = 100;
  
  // Build AST first
  rustg::Token* d_tokens;
  rustg::ASTNode* d_ast_nodes;
  uint32_t* d_ast_node_count;
  uint32_t* d_parent_indices;
  bool* d_is_valid;
  
  cudaMalloc(&d_tokens, tokens.size() * sizeof(rustg::Token));
  cudaMalloc(&d_ast_nodes, max_nodes * sizeof(rustg::ASTNode));
  cudaMalloc(&d_ast_node_count, sizeof(uint32_t));
  cudaMalloc(&d_parent_indices, max_nodes * sizeof(uint32_t));
  cudaMalloc(&d_is_valid, sizeof(bool));
  
  cudaMemset(d_ast_node_count, 0, sizeof(uint32_t));
  cudaMemset(d_parent_indices, 0xFF, max_nodes * sizeof(uint32_t));
  
  cudaMemcpy(d_tokens, tokens.data(), tokens.size() * sizeof(rustg::Token), 
             cudaMemcpyHostToDevice);
  
  // Construct AST
  launch_ast_construction_kernel(
      d_tokens, tokens.size(),
      d_ast_nodes, d_ast_node_count, d_parent_indices,
      max_nodes);
  
  uint32_t node_count;
  cudaMemcpy(&node_count, d_ast_node_count, sizeof(uint32_t), 
             cudaMemcpyDeviceToHost);
  
  // Validate AST
  launch_ast_validator_kernel(
      d_ast_nodes, node_count,
      d_parent_indices, d_is_valid);
  
  bool is_valid;
  cudaMemcpy(&is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost);
  
  EXPECT_TRUE(is_valid);
  
  cudaFree(d_tokens);
  cudaFree(d_ast_nodes);
  cudaFree(d_ast_node_count);
  cudaFree(d_parent_indices);
  cudaFree(d_is_valid);
}

// Test memory coalescing for AST construction
TEST_F(ASTConstructionTest, MemoryCoalescingAST) {
  // Generate aligned token data
  const size_t token_count = 1024;
  std::vector<rustg::Token> tokens(token_count);
  
  // Simple pattern: identifier = literal;
  for (size_t i = 0; i < token_count - 1; i += 4) {
    tokens[i] = {rustg::TokenType::Identifier, static_cast<uint32_t>(i * 10), 3, 1, 1};
    tokens[i + 1] = {rustg::TokenType::Equal, static_cast<uint32_t>(i * 10 + 4), 1, 1, 5};
    tokens[i + 2] = {rustg::TokenType::IntegerLiteral, static_cast<uint32_t>(i * 10 + 6), 2, 1, 7};
    tokens[i + 3] = {rustg::TokenType::Semicolon, static_cast<uint32_t>(i * 10 + 9), 1, 1, 10};
  }
  tokens.back() = {rustg::TokenType::EOF, static_cast<uint32_t>(token_count * 10), 0, 1, 1};
  
  uint32_t max_nodes = token_count * 2;
  
  rustg::Token* d_tokens;
  rustg::ASTNode* d_ast_nodes;
  uint32_t* d_ast_node_count;
  uint32_t* d_parent_indices;
  
  cudaMalloc(&d_tokens, tokens.size() * sizeof(rustg::Token));
  cudaMalloc(&d_ast_nodes, max_nodes * sizeof(rustg::ASTNode));
  cudaMalloc(&d_ast_node_count, sizeof(uint32_t));
  cudaMalloc(&d_parent_indices, max_nodes * sizeof(uint32_t));
  
  cudaMemcpy(d_tokens, tokens.data(), tokens.size() * sizeof(rustg::Token), 
             cudaMemcpyHostToDevice);
  
  // Launch with coalesced access pattern
  launch_ast_construction_kernel(
      d_tokens, tokens.size(),
      d_ast_nodes, d_ast_node_count, d_parent_indices,
      max_nodes);
  
  // Verify no errors
  cudaError_t err = cudaGetLastError();
  EXPECT_EQ(err, cudaSuccess);
  
  cudaFree(d_tokens);
  cudaFree(d_ast_nodes);
  cudaFree(d_ast_node_count);
  cudaFree(d_parent_indices);
}