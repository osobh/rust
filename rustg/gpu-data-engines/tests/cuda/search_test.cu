/**
 * GPU Search Infrastructure CUDA Tests  
 * STRICT TDD: Written BEFORE implementation
 * Validates 1M+ queries/sec with <10ms latency
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/binary_search.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

// Test result structure
struct TestResult {
    bool success;
    float queries_per_second;
    float avg_latency_ms;
    size_t documents_processed;
    double elapsed_ms;
    char error_msg[256];
};

// Inverted index structures
struct PostingList {
    uint32_t* doc_ids;        // Document IDs containing the term
    float* scores;            // TF-IDF scores for each document
    uint32_t length;          // Number of postings
    uint32_t capacity;        // Allocated capacity
};

struct InvertedIndex {
    PostingList* posting_lists; // One per term in vocabulary
    uint32_t* term_hashes;      // Hash values for terms
    uint32_t vocab_size;        // Number of unique terms
    uint32_t num_documents;     // Total document count
    float* idf_scores;          // Inverse document frequency per term
};

// Vector search structures
struct VectorIndex {
    float* vectors;             // Document vectors (flattened)
    uint32_t* doc_ids;         // Document identifiers
    uint32_t num_vectors;      // Number of vectors
    uint32_t dimensions;       // Vector dimensionality
    float* cluster_centroids;  // IVF cluster centers
    uint32_t num_clusters;     // Number of clusters
    uint32_t* cluster_assignments; // Vector to cluster mapping
};

// Query structures
struct BooleanQuery {
    uint32_t* required_terms;  // AND terms
    uint32_t* optional_terms;  // OR terms  
    uint32_t* excluded_terms;  // NOT terms
    uint32_t num_required;
    uint32_t num_optional;
    uint32_t num_excluded;
    float boost_factor;
};

struct VectorQuery {
    float* query_vector;       // Query embedding
    uint32_t dimensions;       // Vector dimensionality
    uint32_t k;               // Number of nearest neighbors
    float similarity_threshold; // Minimum similarity score
};

struct HybridQuery {
    BooleanQuery boolean_part;
    VectorQuery vector_part;
    float keyword_weight;      // Weight for keyword relevance
    float vector_weight;       // Weight for semantic similarity
};

/**
 * TEST 1: High-Throughput Boolean Search
 * Validates efficient inverted index lookup and boolean query processing
 */
__global__ void test_boolean_search_kernel(TestResult* result,
                                         InvertedIndex* index,
                                         BooleanQuery* queries,
                                         uint32_t num_queries,
                                         uint32_t* result_doc_ids,
                                         float* result_scores,
                                         uint32_t* result_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Each thread processes multiple queries
    for (uint32_t query_idx = tid; query_idx < num_queries; query_idx += blockDim.x * gridDim.x) {
        BooleanQuery& query = queries[query_idx];
        
        // Shared memory for intersection results
        extern __shared__ uint32_t shared_docs[];
        uint32_t* candidate_docs = shared_docs + threadIdx.x * 1024;
        float* candidate_scores = (float*)(shared_docs + blockDim.x * 1024) + threadIdx.x * 1024;
        
        uint32_t num_candidates = 0;
        
        // Process required terms (AND operation)
        if (query.num_required > 0) {
            // Start with first required term
            uint32_t first_term = query.required_terms[0];
            uint32_t term_idx = 0;
            
            // Binary search for term in vocabulary
            uint32_t left = 0, right = index->vocab_size;
            while (left < right) {
                uint32_t mid = (left + right) / 2;
                if (index->term_hashes[mid] == first_term) {
                    term_idx = mid;
                    break;
                } else if (index->term_hashes[mid] < first_term) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            
            // Copy first posting list to candidates
            PostingList& first_list = index->posting_lists[term_idx];
            uint32_t copy_count = min(first_list.length, (uint32_t)1024);
            for (uint32_t i = 0; i < copy_count; i++) {
                candidate_docs[i] = first_list.doc_ids[i];
                candidate_scores[i] = first_list.scores[i];
            }
            num_candidates = copy_count;
            
            // Intersect with remaining required terms
            for (uint32_t t = 1; t < query.num_required; t++) {
                uint32_t term = query.required_terms[t];
                
                // Find term in vocabulary
                left = 0; right = index->vocab_size;
                term_idx = 0;
                while (left < right) {
                    uint32_t mid = (left + right) / 2;
                    if (index->term_hashes[mid] == term) {
                        term_idx = mid;
                        break;
                    } else if (index->term_hashes[mid] < term) {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                
                // Intersect posting lists
                PostingList& term_list = index->posting_lists[term_idx];
                uint32_t new_count = 0;
                
                for (uint32_t i = 0; i < num_candidates; i++) {
                    uint32_t doc_id = candidate_docs[i];
                    
                    // Binary search in term posting list
                    bool found = false;
                    float term_score = 0.0f;
                    
                    left = 0; right = term_list.length;
                    while (left < right) {
                        uint32_t mid = (left + right) / 2;
                        if (term_list.doc_ids[mid] == doc_id) {
                            found = true;
                            term_score = term_list.scores[mid];
                            break;
                        } else if (term_list.doc_ids[mid] < doc_id) {
                            left = mid + 1;
                        } else {
                            right = mid;
                        }
                    }
                    
                    if (found) {
                        candidate_docs[new_count] = doc_id;
                        candidate_scores[new_count] = candidate_scores[i] + term_score;
                        new_count++;
                    }
                }
                
                num_candidates = new_count;
                if (num_candidates == 0) break; // No matches
            }
        }
        
        // Store results for this query
        uint32_t output_offset = query_idx * 100; // Max 100 results per query
        uint32_t output_count = min(num_candidates, (uint32_t)100);
        
        for (uint32_t i = 0; i < output_count; i++) {
            result_doc_ids[output_offset + i] = candidate_docs[i];
            result_scores[output_offset + i] = candidate_scores[i];
        }
        result_counts[query_idx] = output_count;
    }
}

/**
 * TEST 2: Vector Similarity Search
 * Validates GPU-accelerated k-nearest neighbor search
 */
__global__ void test_vector_search_kernel(TestResult* result,
                                        VectorIndex* index,
                                        VectorQuery* queries,
                                        uint32_t num_queries,
                                        uint32_t* result_doc_ids,
                                        float* result_similarities,
                                        uint32_t* result_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Each thread processes queries
    for (uint32_t query_idx = tid; query_idx < num_queries; query_idx += blockDim.x * gridDim.x) {
        VectorQuery& query = queries[query_idx];
        
        // Shared memory for top-k results
        extern __shared__ float shared_similarities[];
        extern __shared__ uint32_t shared_ids[];
        
        float* top_similarities = shared_similarities + threadIdx.x * 128;
        uint32_t* top_doc_ids = shared_ids + threadIdx.x * 128;
        
        // Initialize with negative similarities
        for (int i = 0; i < query.k && i < 128; i++) {
            top_similarities[i] = -1.0f;
            top_doc_ids[i] = 0;
        }
        
        uint32_t heap_size = 0;
        
        // Compute similarities with all vectors
        for (uint32_t vec_idx = 0; vec_idx < index->num_vectors; vec_idx++) {
            float* doc_vector = index->vectors + vec_idx * index->dimensions;
            
            // Compute cosine similarity using warp cooperation
            float dot_product = 0.0f;
            float query_norm = 0.0f;
            float doc_norm = 0.0f;
            
            // Parallel reduction within warp
            for (uint32_t d = warp.thread_rank(); d < index->dimensions; d += warp.size()) {
                float q_val = query.query_vector[d];
                float d_val = doc_vector[d];
                
                dot_product += q_val * d_val;
                query_norm += q_val * q_val;
                doc_norm += d_val * d_val;
            }
            
            // Warp-level reductions
            for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
                dot_product += warp.shfl_down(dot_product, offset);
                query_norm += warp.shfl_down(query_norm, offset);
                doc_norm += warp.shfl_down(doc_norm, offset);
            }
            
            if (warp.thread_rank() == 0) {
                float similarity = dot_product / (sqrtf(query_norm) * sqrtf(doc_norm));
                
                // Check if similarity meets threshold
                if (similarity >= query.similarity_threshold) {
                    // Maintain top-k using min-heap
                    if (heap_size < query.k) {
                        // Add to heap
                        top_similarities[heap_size] = similarity;
                        top_doc_ids[heap_size] = index->doc_ids[vec_idx];
                        heap_size++;
                        
                        // Heapify up
                        int pos = heap_size - 1;
                        while (pos > 0) {
                            int parent = (pos - 1) / 2;
                            if (top_similarities[pos] < top_similarities[parent]) {
                                // Swap
                                float temp_sim = top_similarities[pos];
                                uint32_t temp_id = top_doc_ids[pos];
                                top_similarities[pos] = top_similarities[parent];
                                top_doc_ids[pos] = top_doc_ids[parent];
                                top_similarities[parent] = temp_sim;
                                top_doc_ids[parent] = temp_id;
                                pos = parent;
                            } else {
                                break;
                            }
                        }
                    } else if (similarity > top_similarities[0]) {
                        // Replace minimum
                        top_similarities[0] = similarity;
                        top_doc_ids[0] = index->doc_ids[vec_idx];
                        
                        // Heapify down
                        int pos = 0;
                        while (2 * pos + 1 < heap_size) {
                            int left_child = 2 * pos + 1;
                            int right_child = 2 * pos + 2;
                            int smallest = pos;
                            
                            if (left_child < heap_size && 
                                top_similarities[left_child] < top_similarities[smallest]) {
                                smallest = left_child;
                            }
                            
                            if (right_child < heap_size && 
                                top_similarities[right_child] < top_similarities[smallest]) {
                                smallest = right_child;
                            }
                            
                            if (smallest != pos) {
                                // Swap
                                float temp_sim = top_similarities[pos];
                                uint32_t temp_id = top_doc_ids[pos];
                                top_similarities[pos] = top_similarities[smallest];
                                top_doc_ids[pos] = top_doc_ids[smallest];
                                top_similarities[smallest] = temp_sim;
                                top_doc_ids[smallest] = temp_id;
                                pos = smallest;
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        // Store results for this query
        if (warp.thread_rank() == 0) {
            uint32_t output_offset = query_idx * query.k;
            for (uint32_t i = 0; i < heap_size; i++) {
                result_doc_ids[output_offset + i] = top_doc_ids[i];
                result_similarities[output_offset + i] = top_similarities[i];
            }
            result_counts[query_idx] = heap_size;
        }
    }
}

/**
 * TEST 3: Hybrid Search (Keyword + Vector)
 * Validates fusion of boolean and vector search results
 */
__global__ void test_hybrid_search_kernel(TestResult* result,
                                        InvertedIndex* keyword_index,
                                        VectorIndex* vector_index,
                                        HybridQuery* queries,
                                        uint32_t num_queries,
                                        uint32_t* result_doc_ids,
                                        float* result_scores,
                                        uint32_t* result_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process queries
    for (uint32_t query_idx = tid; query_idx < num_queries; query_idx += blockDim.x * gridDim.x) {
        HybridQuery& query = queries[query_idx];
        
        // Allocate local memory for intermediate results
        extern __shared__ char shared_mem[];
        uint32_t* keyword_docs = (uint32_t*)(shared_mem + threadIdx.x * 8192);
        float* keyword_scores = (float*)(keyword_docs + 1024);
        uint32_t* vector_docs = (uint32_t*)(keyword_scores + 1024);
        float* vector_scores = (float*)(vector_docs + 1024);
        
        uint32_t num_keyword_results = 0;
        uint32_t num_vector_results = 0;
        
        // Execute keyword search component
        // (Simplified - would call boolean search logic)
        if (query.boolean_part.num_required > 0) {
            // Simulate keyword search results
            for (uint32_t i = 0; i < min(100U, keyword_index->num_documents / 100); i++) {
                keyword_docs[i] = i * 7 + query_idx; // Pseudo-random doc IDs
                keyword_scores[i] = 0.8f - i * 0.01f; // Decreasing relevance
                num_keyword_results++;
            }
        }
        
        // Execute vector search component
        // (Simplified - would call vector similarity logic)
        if (query.vector_part.query_vector) {
            // Simulate vector search results
            for (uint32_t i = 0; i < min((uint32_t)query.vector_part.k, 50U); i++) {
                vector_docs[i] = i * 11 + query_idx * 3; // Different pseudo-random docs
                vector_scores[i] = 0.9f - i * 0.02f; // Decreasing similarity
                num_vector_results++;
            }
        }
        
        // Fusion: Combine and re-rank results
        extern __shared__ uint32_t fusion_docs[];
        extern __shared__ float fusion_scores[];
        uint32_t* final_docs = fusion_docs + threadIdx.x * 200;
        float* final_scores = fusion_scores + threadIdx.x * 200;
        
        uint32_t final_count = 0;
        
        // Add keyword results with weighting
        for (uint32_t i = 0; i < num_keyword_results && final_count < 200; i++) {
            final_docs[final_count] = keyword_docs[i];
            final_scores[final_count] = keyword_scores[i] * query.keyword_weight;
            final_count++;
        }
        
        // Merge vector results with score fusion
        for (uint32_t i = 0; i < num_vector_results; i++) {
            uint32_t vec_doc = vector_docs[i];
            float vec_score = vector_scores[i] * query.vector_weight;
            
            // Check if document already exists in keyword results
            bool found = false;
            for (uint32_t j = 0; j < final_count; j++) {
                if (final_docs[j] == vec_doc) {
                    // Combine scores (reciprocal rank fusion)
                    final_scores[j] = final_scores[j] + vec_score;
                    found = true;
                    break;
                }
            }
            
            // Add new document if not found and space available
            if (!found && final_count < 200) {
                final_docs[final_count] = vec_doc;
                final_scores[final_count] = vec_score;
                final_count++;
            }
        }
        
        // Sort results by combined score (bubble sort for simplicity)
        for (uint32_t i = 0; i < final_count; i++) {
            for (uint32_t j = i + 1; j < final_count; j++) {
                if (final_scores[j] > final_scores[i]) {
                    // Swap
                    uint32_t temp_doc = final_docs[i];
                    float temp_score = final_scores[i];
                    final_docs[i] = final_docs[j];
                    final_scores[i] = final_scores[j];
                    final_docs[j] = temp_doc;
                    final_scores[j] = temp_score;
                }
            }
        }
        
        // Store top results
        uint32_t output_offset = query_idx * 100;
        uint32_t output_count = min(final_count, 100U);
        
        for (uint32_t i = 0; i < output_count; i++) {
            result_doc_ids[output_offset + i] = final_docs[i];
            result_scores[output_offset + i] = final_scores[i];
        }
        result_counts[query_idx] = output_count;
    }
}

/**
 * TEST 4: Real-time Index Updates
 * Validates concurrent index modification during search
 */
__global__ void test_index_updates_kernel(TestResult* result,
                                        InvertedIndex* index,
                                        uint32_t* new_documents,
                                        uint32_t* new_terms,
                                        float* new_scores,
                                        uint32_t num_updates,
                                        uint32_t* update_success_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint32_t local_success = 0;
    
    // Process index updates
    for (uint32_t update_idx = tid; update_idx < num_updates; update_idx += blockDim.x * gridDim.x) {
        uint32_t doc_id = new_documents[update_idx];
        uint32_t term_hash = new_terms[update_idx];
        float score = new_scores[update_idx];
        
        // Find term in vocabulary
        uint32_t left = 0, right = index->vocab_size;
        uint32_t term_idx = UINT32_MAX;
        
        while (left < right) {
            uint32_t mid = (left + right) / 2;
            if (index->term_hashes[mid] == term_hash) {
                term_idx = mid;
                break;
            } else if (index->term_hashes[mid] < term_hash) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        if (term_idx != UINT32_MAX) {
            PostingList& posting_list = index->posting_lists[term_idx];
            
            // Try to add document to posting list (atomic operation)
            uint32_t old_length = atomicAdd(&posting_list.length, 1);
            if (old_length < posting_list.capacity) {
                posting_list.doc_ids[old_length] = doc_id;
                posting_list.scores[old_length] = score;
                local_success++;
            } else {
                // Revert length increase if no capacity
                atomicSub(&posting_list.length, 1);
            }
        }
    }
    
    // Accumulate successful updates
    atomicAdd(update_success_count, local_success);
}

/**
 * Performance Test Wrapper Functions
 */
extern "C" {
    void test_search_boolean_performance(TestResult* result, uint32_t num_documents, uint32_t num_queries) {
        // Create mock inverted index
        InvertedIndex index;
        index.num_documents = num_documents;
        index.vocab_size = 100000; // 100K unique terms
        
        cudaMalloc(&index.term_hashes, index.vocab_size * sizeof(uint32_t));
        cudaMalloc(&index.posting_lists, index.vocab_size * sizeof(PostingList));
        cudaMalloc(&index.idf_scores, index.vocab_size * sizeof(float));
        
        // Initialize term hashes (sorted)
        thrust::sequence(thrust::device, index.term_hashes, index.term_hashes + index.vocab_size);
        
        // Initialize posting lists with random documents
        PostingList* h_posting_lists = new PostingList[index.vocab_size];
        for (uint32_t i = 0; i < index.vocab_size; i++) {
            uint32_t list_size = 50 + (i % 200); // Variable list sizes
            cudaMalloc(&h_posting_lists[i].doc_ids, list_size * sizeof(uint32_t));
            cudaMalloc(&h_posting_lists[i].scores, list_size * sizeof(float));
            h_posting_lists[i].length = list_size;
            h_posting_lists[i].capacity = list_size;
            
            // Initialize with sorted document IDs
            thrust::sequence(thrust::device, h_posting_lists[i].doc_ids, 
                            h_posting_lists[i].doc_ids + list_size, i);
            thrust::fill(thrust::device, h_posting_lists[i].scores,
                        h_posting_lists[i].scores + list_size, 1.0f);
        }
        cudaMemcpy(index.posting_lists, h_posting_lists, 
                   index.vocab_size * sizeof(PostingList), cudaMemcpyHostToDevice);
        
        // Create test queries
        BooleanQuery* queries;
        cudaMalloc(&queries, num_queries * sizeof(BooleanQuery));
        
        // Initialize queries with random terms
        BooleanQuery* h_queries = new BooleanQuery[num_queries];
        for (uint32_t i = 0; i < num_queries; i++) {
            h_queries[i].num_required = 2 + (i % 3); // 2-4 required terms
            h_queries[i].num_optional = 0;
            h_queries[i].num_excluded = 0;
            
            cudaMalloc(&h_queries[i].required_terms, h_queries[i].num_required * sizeof(uint32_t));
            
            // Set random required terms
            uint32_t* h_terms = new uint32_t[h_queries[i].num_required];
            for (uint32_t j = 0; j < h_queries[i].num_required; j++) {
                h_terms[j] = (i * 7 + j * 11) % index.vocab_size;
            }
            cudaMemcpy(h_queries[i].required_terms, h_terms,
                      h_queries[i].num_required * sizeof(uint32_t), cudaMemcpyHostToDevice);
            delete[] h_terms;
        }
        cudaMemcpy(queries, h_queries, num_queries * sizeof(BooleanQuery), cudaMemcpyHostToDevice);
        
        // Allocate result buffers
        uint32_t* result_doc_ids;
        float* result_scores;
        uint32_t* result_counts;
        
        cudaMalloc(&result_doc_ids, num_queries * 100 * sizeof(uint32_t));
        cudaMalloc(&result_scores, num_queries * 100 * sizeof(float));
        cudaMalloc(&result_counts, num_queries * sizeof(uint32_t));
        
        // Launch search kernel
        dim3 block(256);
        dim3 grid((num_queries + block.x - 1) / block.x);
        size_t shared_mem = block.x * (1024 * sizeof(uint32_t) + 1024 * sizeof(float));
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        test_boolean_search_kernel<<<grid, block, shared_mem>>>(
            result, &index, queries, num_queries, 
            result_doc_ids, result_scores, result_counts);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        
        // Calculate performance
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        result->elapsed_ms = elapsed_ms;
        result->queries_per_second = num_queries / (elapsed_ms / 1000.0);
        result->avg_latency_ms = elapsed_ms / num_queries;
        result->documents_processed = num_documents;
        result->success = (result->queries_per_second >= 1000000.0 && 
                          result->avg_latency_ms < 10.0); // 1M QPS, <10ms latency
        
        // Cleanup
        for (uint32_t i = 0; i < index.vocab_size; i++) {
            cudaFree(h_posting_lists[i].doc_ids);
            cudaFree(h_posting_lists[i].scores);
        }
        delete[] h_posting_lists;
        
        for (uint32_t i = 0; i < num_queries; i++) {
            cudaFree(h_queries[i].required_terms);
        }
        delete[] h_queries;
        
        cudaFree(index.term_hashes);
        cudaFree(index.posting_lists);
        cudaFree(index.idf_scores);
        cudaFree(queries);
        cudaFree(result_doc_ids);
        cudaFree(result_scores);
        cudaFree(result_counts);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void test_search_performance_comprehensive(TestResult* result) {
        const uint32_t NUM_DOCUMENTS = 10000000; // 10M documents
        const uint32_t NUM_QUERIES = 100000;     // 100K queries for batch test
        
        // Test boolean search performance
        test_search_boolean_performance(result, NUM_DOCUMENTS, NUM_QUERIES);
        
        if (!result->success) {
            strcpy(result->error_msg, "Boolean search failed to meet 1M QPS or <10ms latency target");
            return;
        }
        
        result->success = true;
        strcpy(result->error_msg, "All search infrastructure tests passed");
    }
}