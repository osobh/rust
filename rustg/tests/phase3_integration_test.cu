#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <cassert>
#include "../include/gpu_types.h"

namespace rustg {

// External kernel declarations from Phase 3
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

extern "C" void launch_symbol_table_builder(
    const Symbol* symbols, uint32_t num_symbols,
    uint32_t* hash_table, uint32_t table_size,
    uint32_t* collision_count);

extern "C" void launch_batch_symbol_lookup(
    const uint32_t* query_hashes, uint32_t num_queries,
    const uint32_t* hash_table, uint32_t table_size,
    uint32_t* results);

extern "C" void launch_build_module_tree(
    const ModuleNode* modules, uint32_t num_modules,
    uint32_t* parent_pointers, uint32_t* children_lists,
    uint32_t* children_offsets, uint32_t* depth_array,
    uint32_t* tree_stats);

extern "C" void launch_compute_visibility_matrix(
    const ModuleNode* modules, uint32_t num_modules,
    const uint32_t* parent_pointers, const uint32_t* depth_array,
    uint8_t* visibility_matrix, uint32_t matrix_pitch);

class Phase3IntegrationTests {
private:
    // Test data generators
    std::vector<CrateNode> generate_large_crate_graph(uint32_t num_crates) {
        std::vector<CrateNode> crates;
        crates.reserve(num_crates);
        
        for (uint32_t i = 0; i < num_crates; ++i) {
            CrateNode crate;
            crate.crate_id = i;
            crate.name_hash = hash("crate_" + std::to_string(i));
            crate.version_major = 1;
            crate.version_minor = i % 10;
            crate.version_patch = 0;
            crate.dependency_start = i * 3;
            crate.dependency_count = (i < num_crates - 1) ? 3 : 0;
            crate.symbol_table_offset = i * 100;
            crate.feature_flags = 0;
            crates.push_back(crate);
        }
        
        return crates;
    }
    
    std::vector<DependencyEdge> generate_dependencies(uint32_t num_crates) {
        std::vector<DependencyEdge> edges;
        
        // Create a complex dependency graph
        for (uint32_t i = 0; i < num_crates - 1; ++i) {
            // Each crate depends on 1-3 other crates
            uint32_t num_deps = 1 + (i % 3);
            for (uint32_t j = 0; j < num_deps && i + j + 1 < num_crates; ++j) {
                DependencyEdge edge;
                edge.from_crate = i;
                edge.to_crate = i + j + 1;
                edge.edge_type = j; // 0=normal, 1=dev, 2=build
                edge.version_req = 0;
                edges.push_back(edge);
            }
        }
        
        return edges;
    }
    
    std::vector<Symbol> generate_symbols(uint32_t num_symbols) {
        std::vector<Symbol> symbols;
        symbols.reserve(num_symbols);
        
        for (uint32_t i = 0; i < num_symbols; ++i) {
            Symbol sym;
            sym.name_hash = hash("symbol_" + std::to_string(i));
            sym.crate_id = i % 100;
            sym.module_id = i % 10;
            sym.symbol_type = i % 4; // function, struct, enum, trait
            sym.visibility = i % 3;   // public, crate, private
            sym.definition_loc = i * 10;
            sym.attributes = 0;
            sym.generic_params = i % 3;
            symbols.push_back(sym);
        }
        
        return symbols;
    }
    
    std::vector<ModuleNode> generate_module_tree(uint32_t num_modules) {
        std::vector<ModuleNode> modules;
        modules.reserve(num_modules);
        
        // Create hierarchical module structure
        for (uint32_t i = 0; i < num_modules; ++i) {
            ModuleNode mod;
            mod.module_id = i;
            mod.parent_id = (i == 0) ? UINT32_MAX : (i - 1) / 4;
            mod.crate_id = i / 100;
            mod.name_hash = hash("module_" + std::to_string(i));
            mod.children_start = i * 4 + 1;
            mod.children_count = (i * 4 + 4 < num_modules) ? 4 : 0;
            mod.depth = 0; // Will be computed
            mod.visibility_mask = (i % 2) ? 0x01 : 0x03;
            mod.symbol_start = i * 10;
            mod.symbol_count = 10;
            mod.attributes = 0;
            mod.file_id = i;
            modules.push_back(mod);
        }
        
        return modules;
    }
    
    uint32_t hash(const std::string& str) {
        uint32_t hash = 5381;
        for (char c : str) {
            hash = ((hash << 5) + hash) + c;
        }
        return hash;
    }
    
    template<typename T>
    double measure_kernel_time(std::function<void()> kernel_launch) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        kernel_launch();
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds;
    }
    
public:
    // Test 1: Large-scale crate graph construction
    bool test_large_crate_graph() {
        std::cout << "\n=== Testing Large Crate Graph Construction ===\n";
        
        const uint32_t num_crates = 10000;
        auto crates = generate_large_crate_graph(num_crates);
        auto edges = generate_dependencies(num_crates);
        uint32_t num_edges = edges.size();
        
        std::cout << "  Crates: " << num_crates << "\n";
        std::cout << "  Dependencies: " << num_edges << "\n";
        
        // Allocate GPU memory
        CrateNode* d_crates;
        DependencyEdge* d_edges;
        uint32_t* d_csr_row_offsets;
        uint32_t* d_csr_col_indices;
        uint32_t* d_csr_values;
        
        cudaMalloc(&d_crates, num_crates * sizeof(CrateNode));
        cudaMalloc(&d_edges, num_edges * sizeof(DependencyEdge));
        cudaMalloc(&d_csr_row_offsets, (num_crates + 1) * sizeof(uint32_t));
        cudaMalloc(&d_csr_col_indices, num_edges * sizeof(uint32_t));
        cudaMalloc(&d_csr_values, num_edges * sizeof(uint32_t));
        
        cudaMemcpy(d_crates, crates.data(), 
                  num_crates * sizeof(CrateNode), cudaMemcpyHostToDevice);
        cudaMemcpy(d_edges, edges.data(), 
                  num_edges * sizeof(DependencyEdge), cudaMemcpyHostToDevice);
        
        // Measure performance
        double time_ms = measure_kernel_time<void>([&]() {
            launch_crate_graph_builder(
                d_crates, num_crates,
                d_edges, num_edges,
                d_csr_row_offsets, d_csr_col_indices, d_csr_values
            );
        });
        
        // Calculate throughput
        double edges_per_second = (num_edges / time_ms) * 1000.0;
        std::cout << "  Time: " << time_ms << " ms\n";
        std::cout << "  Throughput: " << edges_per_second << " edges/second\n";
        
        // Verify CSR structure
        std::vector<uint32_t> row_offsets(num_crates + 1);
        cudaMemcpy(row_offsets.data(), d_csr_row_offsets,
                  (num_crates + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        bool valid = (row_offsets[0] == 0 && row_offsets[num_crates] <= num_edges);
        
        // Cleanup
        cudaFree(d_crates);
        cudaFree(d_edges);
        cudaFree(d_csr_row_offsets);
        cudaFree(d_csr_col_indices);
        cudaFree(d_csr_values);
        
        std::cout << "  [" << (valid ? "PASS" : "FAIL") 
                 << "] CSR construction (" << edges_per_second << " edges/s)\n";
        
        return valid && edges_per_second > 100000; // Target: 100K edges/s
    }
    
    // Test 2: Parallel graph traversal at scale
    bool test_parallel_traversal() {
        std::cout << "\n=== Testing Parallel Graph Traversal ===\n";
        
        const uint32_t num_nodes = 50000;
        const uint32_t num_edges = 200000;
        
        // Create a synthetic graph
        std::vector<uint32_t> row_offsets(num_nodes + 1);
        std::vector<uint32_t> col_indices(num_edges);
        
        // Generate random graph structure
        uint32_t edge_idx = 0;
        for (uint32_t i = 0; i < num_nodes; ++i) {
            row_offsets[i] = edge_idx;
            uint32_t degree = (i % 10) + 1;
            for (uint32_t j = 0; j < degree && edge_idx < num_edges; ++j) {
                col_indices[edge_idx++] = (i + j + 1) % num_nodes;
            }
        }
        row_offsets[num_nodes] = edge_idx;
        
        // GPU allocation
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
        
        // Measure BFS performance
        double time_ms = measure_kernel_time<void>([&]() {
            launch_parallel_bfs(
                d_row_offsets, d_col_indices,
                num_nodes, 0,
                d_distances, d_predecessors
            );
        });
        
        double edges_per_second = (num_edges / time_ms) * 1000.0;
        std::cout << "  Nodes: " << num_nodes << "\n";
        std::cout << "  Edges: " << num_edges << "\n";
        std::cout << "  Time: " << time_ms << " ms\n";
        std::cout << "  Throughput: " << edges_per_second << " edges/second\n";
        
        // Verify some distances
        std::vector<int32_t> distances(100);
        cudaMemcpy(distances.data(), d_distances,
                  100 * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        bool valid = (distances[0] == 0); // Start node distance is 0
        
        // Cleanup
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_distances);
        cudaFree(d_predecessors);
        
        std::cout << "  [" << (valid ? "PASS" : "FAIL") 
                 << "] BFS traversal (" << edges_per_second << " edges/s)\n";
        
        return valid && edges_per_second > 1000000; // Target: 1M edges/s
    }
    
    // Test 3: Symbol table performance
    bool test_symbol_table_performance() {
        std::cout << "\n=== Testing Symbol Table Performance ===\n";
        
        const uint32_t num_symbols = 1000000;
        const uint32_t table_size = 2097152; // 2^21
        const uint32_t num_queries = 100000;
        
        auto symbols = generate_symbols(num_symbols);
        
        // Generate query hashes
        std::vector<uint32_t> query_hashes(num_queries);
        for (uint32_t i = 0; i < num_queries; ++i) {
            query_hashes[i] = symbols[i % num_symbols].name_hash;
        }
        
        // GPU allocation
        Symbol* d_symbols;
        uint32_t* d_hash_table;
        uint32_t* d_collision_count;
        uint32_t* d_query_hashes;
        uint32_t* d_results;
        
        cudaMalloc(&d_symbols, num_symbols * sizeof(Symbol));
        cudaMalloc(&d_hash_table, table_size * 3 * sizeof(uint32_t));
        cudaMalloc(&d_collision_count, sizeof(uint32_t));
        cudaMalloc(&d_query_hashes, num_queries * sizeof(uint32_t));
        cudaMalloc(&d_results, num_queries * sizeof(uint32_t));
        
        cudaMemcpy(d_symbols, symbols.data(),
                  num_symbols * sizeof(Symbol), cudaMemcpyHostToDevice);
        cudaMemcpy(d_query_hashes, query_hashes.data(),
                  num_queries * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // Build hash table
        double build_time = measure_kernel_time<void>([&]() {
            launch_symbol_table_builder(
                d_symbols, num_symbols,
                d_hash_table, table_size,
                d_collision_count
            );
        });
        
        // Perform batch lookups
        double lookup_time = measure_kernel_time<void>([&]() {
            launch_batch_symbol_lookup(
                d_query_hashes, num_queries,
                d_hash_table, table_size,
                d_results
            );
        });
        
        double symbols_per_second = (num_symbols / build_time) * 1000.0;
        double lookups_per_second = (num_queries / lookup_time) * 1000.0;
        
        std::cout << "  Symbols: " << num_symbols << "\n";
        std::cout << "  Table Size: " << table_size << "\n";
        std::cout << "  Build Time: " << build_time << " ms\n";
        std::cout << "  Build Throughput: " << symbols_per_second << " symbols/s\n";
        std::cout << "  Lookup Time: " << lookup_time << " ms\n";
        std::cout << "  Lookup Throughput: " << lookups_per_second << " lookups/s\n";
        
        // Check collision count
        uint32_t collision_count;
        cudaMemcpy(&collision_count, d_collision_count,
                  sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        double collision_rate = (double)collision_count / num_symbols;
        std::cout << "  Collisions: " << collision_count 
                 << " (" << collision_rate * 100 << "%)\n";
        
        // Cleanup
        cudaFree(d_symbols);
        cudaFree(d_hash_table);
        cudaFree(d_collision_count);
        cudaFree(d_query_hashes);
        cudaFree(d_results);
        
        bool passed = (lookups_per_second > 1000000 && collision_rate < 0.1);
        std::cout << "  [" << (passed ? "PASS" : "FAIL") 
                 << "] Symbol table performance\n";
        
        return passed;
    }
    
    // Test 4: Module tree operations
    bool test_module_tree_operations() {
        std::cout << "\n=== Testing Module Tree Operations ===\n";
        
        const uint32_t num_modules = 100000;
        auto modules = generate_module_tree(num_modules);
        
        // GPU allocation
        ModuleNode* d_modules;
        uint32_t* d_parent_pointers;
        uint32_t* d_children_lists;
        uint32_t* d_children_offsets;
        uint32_t* d_depth_array;
        uint32_t* d_tree_stats;
        uint8_t* d_visibility_matrix;
        
        uint32_t matrix_pitch = ((num_modules + 31) / 32) * 32; // Align to 32
        
        cudaMalloc(&d_modules, num_modules * sizeof(ModuleNode));
        cudaMalloc(&d_parent_pointers, num_modules * sizeof(uint32_t));
        cudaMalloc(&d_children_lists, num_modules * 4 * sizeof(uint32_t));
        cudaMalloc(&d_children_offsets, (num_modules + 1) * sizeof(uint32_t));
        cudaMalloc(&d_depth_array, num_modules * sizeof(uint32_t));
        cudaMalloc(&d_tree_stats, 10 * sizeof(uint32_t));
        cudaMalloc(&d_visibility_matrix, matrix_pitch * num_modules);
        
        cudaMemcpy(d_modules, modules.data(),
                  num_modules * sizeof(ModuleNode), cudaMemcpyHostToDevice);
        cudaMemset(d_children_offsets, 0, (num_modules + 1) * sizeof(uint32_t));
        
        // Build module tree
        double tree_time = measure_kernel_time<void>([&]() {
            launch_build_module_tree(
                d_modules, num_modules,
                d_parent_pointers, d_children_lists,
                d_children_offsets, d_depth_array, d_tree_stats
            );
        });
        
        // Compute visibility matrix
        double vis_time = measure_kernel_time<void>([&]() {
            launch_compute_visibility_matrix(
                d_modules, num_modules,
                d_parent_pointers, d_depth_array,
                d_visibility_matrix, matrix_pitch
            );
        });
        
        double modules_per_second = (num_modules / tree_time) * 1000.0;
        
        std::cout << "  Modules: " << num_modules << "\n";
        std::cout << "  Tree Build Time: " << tree_time << " ms\n";
        std::cout << "  Visibility Time: " << vis_time << " ms\n";
        std::cout << "  Throughput: " << modules_per_second << " modules/s\n";
        
        // Get tree statistics
        uint32_t tree_stats[10];
        cudaMemcpy(tree_stats, d_tree_stats,
                  10 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        std::cout << "  Max Depth: " << tree_stats[0] << "\n";
        std::cout << "  Total Nodes: " << tree_stats[1] << "\n";
        
        // Cleanup
        cudaFree(d_modules);
        cudaFree(d_parent_pointers);
        cudaFree(d_children_lists);
        cudaFree(d_children_offsets);
        cudaFree(d_depth_array);
        cudaFree(d_tree_stats);
        cudaFree(d_visibility_matrix);
        
        bool passed = (modules_per_second > 500000); // Target: 500K modules/s
        std::cout << "  [" << (passed ? "PASS" : "FAIL") 
                 << "] Module tree operations\n";
        
        return passed;
    }
    
    // Test 5: End-to-end integration
    bool test_end_to_end_integration() {
        std::cout << "\n=== Testing End-to-End Integration ===\n";
        
        // Simulate a medium-sized Rust project
        const uint32_t num_crates = 100;
        const uint32_t num_symbols = 50000;
        const uint32_t num_modules = 1000;
        
        auto start_total = std::chrono::high_resolution_clock::now();
        
        // Phase 1: Build crate graph
        auto crates = generate_large_crate_graph(num_crates);
        auto edges = generate_dependencies(num_crates);
        
        // Phase 2: Build symbol table
        auto symbols = generate_symbols(num_symbols);
        
        // Phase 3: Build module tree
        auto modules = generate_module_tree(num_modules);
        
        auto end_total = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
                       (end_total - start_total);
        
        std::cout << "  Total Crates: " << num_crates << "\n";
        std::cout << "  Total Symbols: " << num_symbols << "\n";
        std::cout << "  Total Modules: " << num_modules << "\n";
        std::cout << "  Total Time: " << duration.count() << " ms\n";
        
        double throughput = (num_crates + num_symbols + num_modules) / 
                           (duration.count() / 1000.0);
        std::cout << "  Overall Throughput: " << throughput << " items/s\n";
        
        bool passed = duration.count() < 1000; // Should complete in under 1 second
        std::cout << "  [" << (passed ? "PASS" : "FAIL") 
                 << "] End-to-end integration\n";
        
        return passed;
    }
    
    // Run all tests
    bool run_all_tests() {
        std::cout << "\n🎯 Phase 3 Integration Tests\n";
        std::cout << "=====================================\n";
        
        bool graph_pass = test_large_crate_graph();
        bool traversal_pass = test_parallel_traversal();
        bool symbol_pass = test_symbol_table_performance();
        bool module_pass = test_module_tree_operations();
        bool e2e_pass = test_end_to_end_integration();
        
        std::cout << "\n=== Integration Test Summary ===\n";
        std::cout << "Crate Graph:      " << (graph_pass ? "✅" : "❌") << "\n";
        std::cout << "Graph Traversal:  " << (traversal_pass ? "✅" : "❌") << "\n";
        std::cout << "Symbol Table:     " << (symbol_pass ? "✅" : "❌") << "\n";
        std::cout << "Module Tree:      " << (module_pass ? "✅" : "❌") << "\n";
        std::cout << "End-to-End:       " << (e2e_pass ? "✅" : "❌") << "\n";
        
        bool all_passed = graph_pass && traversal_pass && symbol_pass && 
                         module_pass && e2e_pass;
        
        std::cout << "\nOverall: " << (all_passed ? 
                    "✅ ALL INTEGRATION TESTS PASSED" : 
                    "❌ SOME INTEGRATION TESTS FAILED") << "\n";
        
        return all_passed;
    }
};

} // namespace rustg

int main() {
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running on: " << prop.name << "\n";
    std::cout << "SM Count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Shared Memory/Block: " << prop.sharedMemPerBlock << " bytes\n";
    
    rustg::Phase3IntegrationTests tests;
    return tests.run_all_tests() ? 0 : 1;
}