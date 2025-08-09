#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <cassert>
#include "../include/gpu_types.h"

namespace rustg {

// Crate graph structures
struct CrateNode {
    uint32_t crate_id;
    uint32_t name_hash;
    uint32_t version_major;
    uint32_t version_minor;
    uint32_t version_patch;
    uint32_t dependency_start;  // Index in edge array
    uint32_t dependency_count;
    uint32_t symbol_table_offset;
    uint32_t feature_flags;
};

struct DependencyEdge {
    uint32_t from_crate;
    uint32_t to_crate;
    uint32_t edge_type;  // 0=normal, 1=dev, 2=build
    uint32_t version_req; // Simplified version requirement
};

struct Symbol {
    uint32_t name_hash;
    uint32_t crate_id;
    uint32_t module_id;
    uint32_t symbol_type;  // function, struct, enum, etc.
    uint32_t visibility;   // public, private, pub(crate), etc.
    uint32_t definition_loc;
};

// External kernel functions (to be implemented)
extern "C" void launch_crate_graph_builder(
    const CrateNode* crates, uint32_t num_crates,
    const DependencyEdge* edges, uint32_t num_edges,
    uint32_t* csr_row_offsets, uint32_t* csr_col_indices,
    uint32_t* csr_values);

extern "C" void launch_parallel_bfs(
    const uint32_t* csr_row_offsets,
    const uint32_t* csr_col_indices,
    uint32_t num_nodes,
    uint32_t start_node,
    int32_t* distances,
    uint32_t* predecessors);

extern "C" void launch_cycle_detection(
    const uint32_t* csr_row_offsets,
    const uint32_t* csr_col_indices,
    uint32_t num_nodes,
    bool* has_cycle,
    uint32_t* cycle_nodes);

extern "C" void launch_symbol_table_builder(
    const Symbol* symbols, uint32_t num_symbols,
    uint32_t* hash_table, uint32_t table_size,
    uint32_t* collision_count);

extern "C" void launch_symbol_resolver(
    uint32_t query_hash,
    const uint32_t* hash_table,
    const Symbol* symbols,
    uint32_t table_size,
    Symbol* result,
    bool* found);

class CrateGraphTests {
private:
    // Test data
    std::vector<CrateNode> test_crates;
    std::vector<DependencyEdge> test_edges;
    std::vector<Symbol> test_symbols;
    
    void setup_test_data() {
        // Create a simple crate graph
        // my_app -> lib_a -> lib_b
        //        -> lib_c
        
        test_crates = {
            {0, hash("my_app"), 1, 0, 0, 0, 2, 0, 0},
            {1, hash("lib_a"), 1, 0, 0, 2, 1, 100, 0},
            {2, hash("lib_b"), 1, 0, 0, 3, 0, 200, 0},
            {3, hash("lib_c"), 1, 0, 0, 3, 0, 300, 0}
        };
        
        test_edges = {
            {0, 1, 0, 0}, // my_app -> lib_a
            {0, 3, 0, 0}, // my_app -> lib_c
            {1, 2, 0, 0}  // lib_a -> lib_b
        };
        
        test_symbols = {
            {hash("main"), 0, 0, 0, 0, 0},
            {hash("process"), 1, 0, 0, 1, 100},
            {hash("calculate"), 2, 0, 0, 1, 200},
            {hash("utility"), 3, 0, 0, 1, 300}
        };
    }
    
    uint32_t hash(const std::string& str) {
        uint32_t hash = 5381;
        for (char c : str) {
            hash = ((hash << 5) + hash) + c;
        }
        return hash;
    }
    
public:
    CrateGraphTests() {
        setup_test_data();
    }
    
    // Test CSR graph construction
    bool test_csr_construction() {
        std::cout << "\n=== Testing CSR Graph Construction ===\n";
        
        // Allocate GPU memory
        CrateNode* d_crates;
        DependencyEdge* d_edges;
        uint32_t* d_csr_row_offsets;
        uint32_t* d_csr_col_indices;
        uint32_t* d_csr_values;
        
        uint32_t num_crates = test_crates.size();
        uint32_t num_edges = test_edges.size();
        
        cudaMalloc(&d_crates, num_crates * sizeof(CrateNode));
        cudaMalloc(&d_edges, num_edges * sizeof(DependencyEdge));
        cudaMalloc(&d_csr_row_offsets, (num_crates + 1) * sizeof(uint32_t));
        cudaMalloc(&d_csr_col_indices, num_edges * sizeof(uint32_t));
        cudaMalloc(&d_csr_values, num_edges * sizeof(uint32_t));
        
        // Copy data to GPU
        cudaMemcpy(d_crates, test_crates.data(), 
                  num_crates * sizeof(CrateNode), cudaMemcpyHostToDevice);
        cudaMemcpy(d_edges, test_edges.data(), 
                  num_edges * sizeof(DependencyEdge), cudaMemcpyHostToDevice);
        
        // Launch kernel
        launch_crate_graph_builder(
            d_crates, num_crates,
            d_edges, num_edges,
            d_csr_row_offsets, d_csr_col_indices, d_csr_values
        );
        
        // Check results
        std::vector<uint32_t> row_offsets(num_crates + 1);
        cudaMemcpy(row_offsets.data(), d_csr_row_offsets,
                  (num_crates + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        // Verify CSR structure
        bool passed = true;
        if (row_offsets[0] != 0 || row_offsets[num_crates] != num_edges) {
            passed = false;
        }
        
        std::cout << "  [" << (passed ? "PASS" : "FAIL") 
                 << "] CSR structure validation\n";
        
        // Cleanup
        cudaFree(d_crates);
        cudaFree(d_edges);
        cudaFree(d_csr_row_offsets);
        cudaFree(d_csr_col_indices);
        cudaFree(d_csr_values);
        
        return passed;
    }
    
    // Test parallel BFS
    bool test_parallel_bfs() {
        std::cout << "\n=== Testing Parallel BFS ===\n";
        
        // Create CSR representation
        uint32_t num_nodes = 4;
        std::vector<uint32_t> row_offsets = {0, 2, 3, 3, 3};
        std::vector<uint32_t> col_indices = {1, 3, 2};
        
        // Allocate GPU memory
        uint32_t* d_row_offsets;
        uint32_t* d_col_indices;
        int32_t* d_distances;
        uint32_t* d_predecessors;
        
        cudaMalloc(&d_row_offsets, row_offsets.size() * sizeof(uint32_t));
        cudaMalloc(&d_col_indices, col_indices.size() * sizeof(uint32_t));
        cudaMalloc(&d_distances, num_nodes * sizeof(int32_t));
        cudaMalloc(&d_predecessors, num_nodes * sizeof(uint32_t));
        
        cudaMemcpy(d_row_offsets, row_offsets.data(),
                  row_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, col_indices.data(),
                  col_indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // Initialize distances to -1
        cudaMemset(d_distances, -1, num_nodes * sizeof(int32_t));
        
        // Launch BFS from node 0
        launch_parallel_bfs(
            d_row_offsets, d_col_indices,
            num_nodes, 0,
            d_distances, d_predecessors
        );
        
        // Check results
        std::vector<int32_t> distances(num_nodes);
        cudaMemcpy(distances.data(), d_distances,
                  num_nodes * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        // Expected distances: [0, 1, 2, 1]
        bool passed = (distances[0] == 0 && distances[1] == 1 && 
                      distances[2] == 2 && distances[3] == 1);
        
        std::cout << "  [" << (passed ? "PASS" : "FAIL") 
                 << "] BFS distance calculation\n";
        
        // Cleanup
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_distances);
        cudaFree(d_predecessors);
        
        return passed;
    }
    
    // Test cycle detection
    bool test_cycle_detection() {
        std::cout << "\n=== Testing Cycle Detection ===\n";
        
        // Create graph with cycle: 0->1->2->0
        uint32_t num_nodes = 3;
        std::vector<uint32_t> row_offsets = {0, 1, 2, 3};
        std::vector<uint32_t> col_indices = {1, 2, 0};
        
        // GPU allocation
        uint32_t* d_row_offsets;
        uint32_t* d_col_indices;
        bool* d_has_cycle;
        uint32_t* d_cycle_nodes;
        
        cudaMalloc(&d_row_offsets, row_offsets.size() * sizeof(uint32_t));
        cudaMalloc(&d_col_indices, col_indices.size() * sizeof(uint32_t));
        cudaMalloc(&d_has_cycle, sizeof(bool));
        cudaMalloc(&d_cycle_nodes, num_nodes * sizeof(uint32_t));
        
        cudaMemcpy(d_row_offsets, row_offsets.data(),
                  row_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, col_indices.data(),
                  col_indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // Launch cycle detection
        launch_cycle_detection(
            d_row_offsets, d_col_indices,
            num_nodes,
            d_has_cycle, d_cycle_nodes
        );
        
        // Check result
        bool has_cycle;
        cudaMemcpy(&has_cycle, d_has_cycle, sizeof(bool), cudaMemcpyDeviceToHost);
        
        std::cout << "  [" << (has_cycle ? "PASS" : "FAIL") 
                 << "] Cycle detected correctly\n";
        
        // Cleanup
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_has_cycle);
        cudaFree(d_cycle_nodes);
        
        return has_cycle;
    }
    
    // Test symbol table
    bool test_symbol_table() {
        std::cout << "\n=== Testing Symbol Table ===\n";
        
        uint32_t table_size = 128;
        uint32_t num_symbols = test_symbols.size();
        
        // GPU allocation
        Symbol* d_symbols;
        uint32_t* d_hash_table;
        uint32_t* d_collision_count;
        
        cudaMalloc(&d_symbols, num_symbols * sizeof(Symbol));
        cudaMalloc(&d_hash_table, table_size * sizeof(uint32_t));
        cudaMalloc(&d_collision_count, sizeof(uint32_t));
        
        cudaMemcpy(d_symbols, test_symbols.data(),
                  num_symbols * sizeof(Symbol), cudaMemcpyHostToDevice);
        cudaMemset(d_hash_table, 0xFF, table_size * sizeof(uint32_t)); // Initialize to -1
        cudaMemset(d_collision_count, 0, sizeof(uint32_t));
        
        // Build hash table
        launch_symbol_table_builder(
            d_symbols, num_symbols,
            d_hash_table, table_size,
            d_collision_count
        );
        
        // Test lookup
        Symbol* d_result;
        bool* d_found;
        cudaMalloc(&d_result, sizeof(Symbol));
        cudaMalloc(&d_found, sizeof(bool));
        
        uint32_t query_hash = hash("process");
        launch_symbol_resolver(
            query_hash,
            d_hash_table, d_symbols, table_size,
            d_result, d_found
        );
        
        bool found;
        Symbol result;
        cudaMemcpy(&found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result, d_result, sizeof(Symbol), cudaMemcpyDeviceToHost);
        
        bool passed = found && result.crate_id == 1;
        std::cout << "  [" << (passed ? "PASS" : "FAIL") 
                 << "] Symbol lookup\n";
        
        // Cleanup
        cudaFree(d_symbols);
        cudaFree(d_hash_table);
        cudaFree(d_collision_count);
        cudaFree(d_result);
        cudaFree(d_found);
        
        return passed;
    }
    
    // Run all tests
    bool run_all_tests() {
        std::cout << "\n🎯 Phase 3: Crate Graph Resolution Tests\n";
        std::cout << "=========================================\n";
        
        bool csr_pass = test_csr_construction();
        bool bfs_pass = test_parallel_bfs();
        bool cycle_pass = test_cycle_detection();
        bool symbol_pass = test_symbol_table();
        
        std::cout << "\n=== Test Summary ===\n";
        std::cout << "CSR Construction: " << (csr_pass ? "✅" : "❌") << "\n";
        std::cout << "Parallel BFS:     " << (bfs_pass ? "✅" : "❌") << "\n";
        std::cout << "Cycle Detection:  " << (cycle_pass ? "✅" : "❌") << "\n";
        std::cout << "Symbol Table:     " << (symbol_pass ? "✅" : "❌") << "\n";
        
        bool all_passed = csr_pass && bfs_pass && cycle_pass && symbol_pass;
        std::cout << "\nOverall: " << (all_passed ? "✅ ALL TESTS PASSED" : "❌ SOME TESTS FAILED") << "\n";
        
        return all_passed;
    }
};

} // namespace rustg

int main() {
    rustg::CrateGraphTests tests;
    return tests.run_all_tests() ? 0 : 1;
}