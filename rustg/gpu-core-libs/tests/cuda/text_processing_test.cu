// GPU-Native Text Processing Tests
// SIMD tokenization, regex, and parsing
// NO STUBS OR MOCKS - Real GPU operations only

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Test result structure
struct TestResult {
    bool passed;
    int test_id;
    float throughput_gbps;
    int matches_found;
    float elapsed_cycles;
    char error_msg[256];
};

// Token types for tokenizer
enum TokenType {
    TOKEN_WORD,
    TOKEN_NUMBER,
    TOKEN_PUNCTUATION,
    TOKEN_WHITESPACE,
    TOKEN_UNKNOWN
};

struct Token {
    unsigned int start;
    unsigned int length;
    TokenType type;
};

// Character classification lookup table
__constant__ unsigned char char_class_table[256];

// Initialize character classes on host
void init_char_classes() {
    unsigned char table[256];
    memset(table, TOKEN_UNKNOWN, 256);
    
    // Whitespace
    table[' '] = table['\t'] = table['\n'] = table['\r'] = TOKEN_WHITESPACE;
    
    // Letters
    for (int i = 'a'; i <= 'z'; i++) table[i] = TOKEN_WORD;
    for (int i = 'A'; i <= 'Z'; i++) table[i] = TOKEN_WORD;
    
    // Numbers
    for (int i = '0'; i <= '9'; i++) table[i] = TOKEN_NUMBER;
    
    // Punctuation
    const char* punct = ".,;:!?\"'()[]{}";
    for (const char* p = punct; *p; p++) {
        table[(unsigned char)*p] = TOKEN_PUNCTUATION;
    }
    
    cudaMemcpyToSymbol(char_class_table, table, 256);
}

// SIMD Tokenizer
struct GPUTokenizer {
    char* text;
    unsigned int* text_len;
    Token* tokens;
    unsigned int* token_count;
    unsigned int* max_tokens;
    
    __device__ void tokenize_parallel() {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        unsigned int len = *text_len;
        
        // Each thread processes a chunk
        for (unsigned int i = tid; i < len; i += stride) {
            // Check for token boundary
            if (i == 0 || char_class_table[text[i-1]] != char_class_table[text[i]]) {
                // Start of new token
                unsigned int token_idx = atomicAdd(token_count, 1);
                if (token_idx < *max_tokens) {
                    tokens[token_idx].start = i;
                    tokens[token_idx].type = (TokenType)char_class_table[text[i]];
                    
                    // Find token length
                    unsigned int j = i + 1;
                    while (j < len && char_class_table[text[j]] == char_class_table[text[i]]) {
                        j++;
                    }
                    tokens[token_idx].length = j - i;
                }
            }
        }
    }
    
    __device__ void tokenize_warp_cooperative() {
        unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        unsigned int lane_id = threadIdx.x % 32;
        unsigned int num_warps = (blockDim.x * gridDim.x) / 32;
        unsigned int len = *text_len;
        
        // Each warp processes a text segment
        unsigned int chunk_size = (len + num_warps - 1) / num_warps;
        unsigned int start = warp_id * chunk_size;
        unsigned int end = min(start + chunk_size, len);
        
        for (unsigned int i = start + lane_id; i < end; i += 32) {
            if (i < len) {
                // Warp-level voting for token boundaries
                bool is_boundary = (i == 0) || 
                    (char_class_table[text[i-1]] != char_class_table[text[i]]);
                
                unsigned int ballot = __ballot_sync(0xFFFFFFFF, is_boundary);
                
                if (is_boundary) {
                    unsigned int token_idx = atomicAdd(token_count, 1);
                    if (token_idx < *max_tokens) {
                        tokens[token_idx].start = i;
                        tokens[token_idx].type = (TokenType)char_class_table[text[i]];
                        
                        // Cooperative length finding
                        unsigned int j = i + 1;
                        while (j < len && char_class_table[text[j]] == char_class_table[text[i]]) {
                            j++;
                        }
                        tokens[token_idx].length = j - i;
                    }
                }
            }
        }
    }
};

// Simple NFA for regex matching
struct GPURegex {
    struct State {
        char match_char;
        int next1;
        int next2;
        bool is_final;
    };
    
    State* states;
    unsigned int* num_states;
    unsigned int* start_state;
    
    __device__ bool match(const char* text, unsigned int len) {
        // Simplified NFA simulation
        bool current[64] = {false};  // Current state set
        bool next[64] = {false};     // Next state set
        
        current[*start_state] = true;
        
        for (unsigned int i = 0; i < len; i++) {
            memset(next, false, sizeof(next));
            
            for (unsigned int s = 0; s < *num_states; s++) {
                if (!current[s]) continue;
                
                State& state = states[s];
                
                if (state.match_char == '.' || state.match_char == text[i]) {
                    if (state.next1 >= 0) next[state.next1] = true;
                    if (state.next2 >= 0) next[state.next2] = true;
                }
            }
            
            memcpy(current, next, sizeof(current));
        }
        
        // Check for accepting state
        for (unsigned int s = 0; s < *num_states; s++) {
            if (current[s] && states[s].is_final) {
                return true;
            }
        }
        
        return false;
    }
    
    __device__ int find_all_matches(const char* text, unsigned int len,
                                    unsigned int* match_positions,
                                    unsigned int max_matches) {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        __shared__ unsigned int match_count;
        
        if (threadIdx.x == 0) {
            match_count = 0;
        }
        __syncthreads();
        
        // Parallel search at different starting positions
        for (unsigned int start = tid; start < len; start += stride) {
            // Try to match from this position
            for (unsigned int end = start + 1; end <= len && end <= start + 100; end++) {
                if (match(text + start, end - start)) {
                    unsigned int idx = atomicAdd(&match_count, 1);
                    if (idx < max_matches) {
                        match_positions[idx * 2] = start;
                        match_positions[idx * 2 + 1] = end;
                    }
                    break;  // Found match starting at this position
                }
            }
        }
        
        __syncthreads();
        return match_count;
    }
};

// JSON parser state machine
struct GPUJsonParser {
    enum JsonState {
        JSON_START,
        JSON_OBJECT,
        JSON_ARRAY,
        JSON_STRING,
        JSON_NUMBER,
        JSON_BOOL,
        JSON_NULL,
        JSON_ERROR
    };
    
    __device__ bool validate_structure(const char* json, unsigned int len) {
        unsigned int depth = 0;
        JsonState state = JSON_START;
        bool in_string = false;
        bool escape_next = false;
        
        for (unsigned int i = 0; i < len; i++) {
            char c = json[i];
            
            if (escape_next) {
                escape_next = false;
                continue;
            }
            
            if (in_string) {
                if (c == '\\') {
                    escape_next = true;
                } else if (c == '"') {
                    in_string = false;
                }
                continue;
            }
            
            switch (c) {
                case '{':
                case '[':
                    depth++;
                    if (depth > 100) return false;  // Max depth
                    break;
                case '}':
                case ']':
                    if (depth == 0) return false;
                    depth--;
                    break;
                case '"':
                    in_string = true;
                    break;
            }
        }
        
        return depth == 0 && !in_string;
    }
    
    __device__ int parallel_parse_array(const char* json, unsigned int len,
                                        float* output, unsigned int max_elements) {
        // Simple number array parser
        unsigned int tid = threadIdx.x;
        unsigned int count = 0;
        bool in_number = false;
        float current_num = 0;
        float sign = 1;
        
        for (unsigned int i = tid; i < len; i += blockDim.x) {
            char c = json[i];
            
            if (c >= '0' && c <= '9') {
                if (!in_number) {
                    in_number = true;
                    current_num = c - '0';
                } else {
                    current_num = current_num * 10 + (c - '0');
                }
            } else if (c == '-' && !in_number) {
                sign = -1;
            } else if (in_number) {
                // End of number
                if (count < max_elements) {
                    output[count] = current_num * sign;
                }
                count++;
                in_number = false;
                current_num = 0;
                sign = 1;
            }
        }
        
        return count;
    }
};

// Test 1: SIMD Tokenization
__global__ void test_simd_tokenization(TestResult* result,
                                       GPUTokenizer* tokenizer,
                                       const char* test_text,
                                       int text_len) {
    // Copy text to device memory
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        memcpy(tokenizer->text, test_text, text_len);
        *tokenizer->text_len = text_len;
        *tokenizer->token_count = 0;
    }
    __syncthreads();
    
    clock_t start = clock();
    tokenizer->tokenize_parallel();
    __syncthreads();
    clock_t end = clock();
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result->passed = (*tokenizer->token_count > 0);
        result->matches_found = *tokenizer->token_count;
        result->elapsed_cycles = (float)(end - start);
        
        // Calculate throughput
        float bytes_processed = text_len;
        float time_ms = result->elapsed_cycles / 1000.0f;
        result->throughput_gbps = (bytes_processed / time_ms) / 1e6;  // GB/s
        
        if (!result->passed) {
            sprintf(result->error_msg, "No tokens found");
        }
    }
}

// Test 2: Warp-cooperative tokenization
__global__ void test_warp_tokenization(TestResult* result,
                                       GPUTokenizer* tokenizer,
                                       const char* test_text,
                                       int text_len) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        memcpy(tokenizer->text, test_text, text_len);
        *tokenizer->text_len = text_len;
        *tokenizer->token_count = 0;
    }
    __syncthreads();
    
    clock_t start = clock();
    tokenizer->tokenize_warp_cooperative();
    __syncthreads();
    clock_t end = clock();
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result->passed = (*tokenizer->token_count > 0);
        result->matches_found = *tokenizer->token_count;
        result->elapsed_cycles = (float)(end - start);
        
        float bytes_processed = text_len;
        float time_ms = result->elapsed_cycles / 1000.0f;
        result->throughput_gbps = (bytes_processed / time_ms) / 1e6;
    }
}

// Test 3: Regex matching
__global__ void test_regex_matching(TestResult* result,
                                    GPURegex* regex,
                                    const char* text,
                                    int text_len) {
    unsigned int* match_positions;
    cudaMalloc(&match_positions, sizeof(unsigned int) * 1000 * 2);
    
    clock_t start = clock();
    int matches = regex->find_all_matches(text, text_len, match_positions, 1000);
    clock_t end = clock();
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result->passed = (matches > 0);
        result->matches_found = matches;
        result->elapsed_cycles = (float)(end - start);
        
        float ops_per_sec = (text_len * matches) / result->elapsed_cycles * 1000;
        result->throughput_gbps = ops_per_sec / 1e9;
        
        if (!result->passed) {
            sprintf(result->error_msg, "No regex matches found");
        }
    }
    
    cudaFree(match_positions);
}

// Test 4: JSON validation
__global__ void test_json_validation(TestResult* result,
                                     GPUJsonParser* parser,
                                     const char* json,
                                     int json_len) {
    clock_t start = clock();
    bool valid = parser->validate_structure(json, json_len);
    clock_t end = clock();
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result->passed = valid;
        result->elapsed_cycles = (float)(end - start);
        
        float bytes_processed = json_len;
        float time_ms = result->elapsed_cycles / 1000.0f;
        result->throughput_gbps = (bytes_processed / time_ms) / 1e6;
        
        if (!result->passed) {
            sprintf(result->error_msg, "Invalid JSON structure");
        }
    }
}

// Test 5: Parallel JSON array parsing
__global__ void test_json_array_parsing(TestResult* result,
                                        GPUJsonParser* parser,
                                        const char* json,
                                        int json_len) {
    float* numbers;
    cudaMalloc(&numbers, sizeof(float) * 1000);
    
    clock_t start = clock();
    int count = parser->parallel_parse_array(json, json_len, numbers, 1000);
    clock_t end = clock();
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result->passed = (count > 0);
        result->matches_found = count;
        result->elapsed_cycles = (float)(end - start);
        
        float bytes_processed = json_len;
        float time_ms = result->elapsed_cycles / 1000.0f;
        result->throughput_gbps = (bytes_processed / time_ms) / 1e6;
    }
    
    cudaFree(numbers);
}

// Test 6: Performance target (10GB/s throughput)
__global__ void test_performance_target(TestResult* result,
                                        char* text,
                                        int text_size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    // Simple character processing benchmark
    clock_t start = clock();
    
    unsigned int count = 0;
    for (unsigned int i = tid; i < text_size; i += stride) {
        // Simulate text processing
        char c = text[i];
        if (c >= 'a' && c <= 'z') {
            text[i] = c - 32;  // To uppercase
            count++;
        }
    }
    
    __syncthreads();
    clock_t end = clock();
    
    if (tid == 0) {
        float elapsed_ms = (float)(end - start) / 1000.0f;
        float throughput = (text_size / elapsed_ms) / 1e6;  // GB/s
        
        result->passed = (throughput > 10.0f);  // 10GB/s target
        result->throughput_gbps = throughput;
        result->elapsed_cycles = (float)(end - start);
        
        if (!result->passed) {
            sprintf(result->error_msg, "Throughput: %.2f GB/s (target: 10 GB/s)",
                   throughput);
        }
    }
}

// Main test runner
int main() {
    printf("GPU-Native Text Processing Tests\n");
    printf("================================\n\n");
    
    // Initialize character classes
    init_char_classes();
    
    TestResult* d_results;
    cudaMalloc(&d_results, sizeof(TestResult) * 10);
    
    TestResult h_results[10];
    
    // Test 1: SIMD Tokenization
    {
        printf("Test 1: SIMD Tokenization...\n");
        
        GPUTokenizer* d_tokenizer;
        cudaMalloc(&d_tokenizer, sizeof(GPUTokenizer));
        
        char* d_text;
        Token* d_tokens;
        unsigned int* d_meta;
        
        cudaMalloc(&d_text, 10000);
        cudaMalloc(&d_tokens, sizeof(Token) * 5000);
        cudaMalloc(&d_meta, sizeof(unsigned int) * 3);
        
        GPUTokenizer h_tokenizer = {d_text, d_meta, d_tokens, d_meta + 1, d_meta + 2};
        cudaMemcpy(d_tokenizer, &h_tokenizer, sizeof(GPUTokenizer), cudaMemcpyHostToDevice);
        
        unsigned int max_tokens = 5000;
        cudaMemcpy(d_meta + 2, &max_tokens, sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        const char* test_text = "Hello world! This is a test 123 of GPU tokenization.";
        
        test_simd_tokenization<<<256, 256>>>(d_results, d_tokenizer, test_text, strlen(test_text));
        cudaDeviceSynchronize();
        
        cudaMemcpy(&h_results[0], d_results, sizeof(TestResult), cudaMemcpyDeviceToHost);
        printf("  Result: %s\n", h_results[0].passed ? "PASSED" : "FAILED");
        printf("  Tokens found: %d\n", h_results[0].matches_found);
        printf("  Throughput: %.2f GB/s\n\n", h_results[0].throughput_gbps);
        
        cudaFree(d_text);
        cudaFree(d_tokens);
        cudaFree(d_meta);
        cudaFree(d_tokenizer);
    }
    
    // Test 2: Warp-cooperative tokenization
    {
        printf("Test 2: Warp-Cooperative Tokenization...\n");
        
        GPUTokenizer* d_tokenizer;
        cudaMalloc(&d_tokenizer, sizeof(GPUTokenizer));
        
        char* d_text;
        Token* d_tokens;
        unsigned int* d_meta;
        
        cudaMalloc(&d_text, 10000);
        cudaMalloc(&d_tokens, sizeof(Token) * 5000);
        cudaMalloc(&d_meta, sizeof(unsigned int) * 3);
        
        GPUTokenizer h_tokenizer = {d_text, d_meta, d_tokens, d_meta + 1, d_meta + 2};
        cudaMemcpy(d_tokenizer, &h_tokenizer, sizeof(GPUTokenizer), cudaMemcpyHostToDevice);
        
        unsigned int max_tokens = 5000;
        cudaMemcpy(d_meta + 2, &max_tokens, sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        const char* test_text = "The quick brown fox jumps over the lazy dog. 1234567890!";
        
        test_warp_tokenization<<<256, 256>>>(d_results + 1, d_tokenizer, test_text, strlen(test_text));
        cudaDeviceSynchronize();
        
        cudaMemcpy(&h_results[1], d_results + 1, sizeof(TestResult), cudaMemcpyDeviceToHost);
        printf("  Result: %s\n", h_results[1].passed ? "PASSED" : "FAILED");
        printf("  Tokens found: %d\n", h_results[1].matches_found);
        printf("  Throughput: %.2f GB/s\n\n", h_results[1].throughput_gbps);
        
        cudaFree(d_text);
        cudaFree(d_tokens);
        cudaFree(d_meta);
        cudaFree(d_tokenizer);
    }
    
    // Test 3: JSON validation
    {
        printf("Test 3: JSON Structure Validation...\n");
        
        GPUJsonParser* d_parser;
        cudaMalloc(&d_parser, sizeof(GPUJsonParser));
        
        const char* json = "{\"name\":\"test\",\"value\":123,\"array\":[1,2,3]}";
        char* d_json;
        cudaMalloc(&d_json, strlen(json) + 1);
        cudaMemcpy(d_json, json, strlen(json) + 1, cudaMemcpyHostToDevice);
        
        test_json_validation<<<1, 256>>>(d_results + 2, d_parser, d_json, strlen(json));
        cudaDeviceSynchronize();
        
        cudaMemcpy(&h_results[2], d_results + 2, sizeof(TestResult), cudaMemcpyDeviceToHost);
        printf("  Result: %s\n", h_results[2].passed ? "PASSED" : "FAILED");
        if (!h_results[2].passed) {
            printf("  Error: %s\n", h_results[2].error_msg);
        }
        printf("  Throughput: %.2f GB/s\n\n", h_results[2].throughput_gbps);
        
        cudaFree(d_json);
        cudaFree(d_parser);
    }
    
    // Summary
    printf("Test Summary\n");
    printf("============\n");
    
    int passed = 0;
    float total_throughput = 0;
    
    for (int i = 0; i < 3; i++) {
        if (h_results[i].passed) {
            passed++;
            total_throughput += h_results[i].throughput_gbps;
        }
    }
    
    printf("Passed: %d/3\n", passed);
    printf("Average Throughput: %.2f GB/s\n", total_throughput / 3);
    
    if (passed == 3 && total_throughput / 3 > 10.0f) {
        printf("\n✓ All tests passed with >10GB/s throughput!\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed or performance target not met\n");
        return 1;
    }
    
    cudaFree(d_results);
}