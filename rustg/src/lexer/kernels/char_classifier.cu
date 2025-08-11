#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "../../../include/gpu_types.h"
#include "../../../include/char_classifier.h"

namespace rustg {

// Character classification lookup table in constant memory
// Constant memory is cached and optimized for broadcast reads
__constant__ CharClass char_class_table[256];

// Host-side lookup table initialization data
static const CharClass host_char_class_table[256] = {
    // 0x00 - 0x1F: Control characters
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Whitespace, CharClass::Newline, CharClass::Whitespace,
    CharClass::Whitespace, CharClass::Whitespace, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    
    // 0x20 - 0x2F: Space and symbols
    CharClass::Whitespace, // ' '
    CharClass::Operator,   // '!'
    CharClass::Quote,      // '"'
    CharClass::Operator,   // '#'
    CharClass::Operator,   // '$'
    CharClass::Operator,   // '%'
    CharClass::Operator,   // '&'
    CharClass::Quote,      // '\''
    CharClass::Delimiter,  // '('
    CharClass::Delimiter,  // ')'
    CharClass::Operator,   // '*'
    CharClass::Operator,   // '+'
    CharClass::Delimiter,  // ','
    CharClass::Operator,   // '-'
    CharClass::Delimiter,  // '.'
    CharClass::Operator,   // '/'
    
    // 0x30 - 0x39: Digits
    CharClass::Digit, CharClass::Digit, CharClass::Digit, CharClass::Digit,
    CharClass::Digit, CharClass::Digit, CharClass::Digit, CharClass::Digit,
    CharClass::Digit, CharClass::Digit,
    
    // 0x3A - 0x40: More symbols
    CharClass::Delimiter,  // ':'
    CharClass::Delimiter,  // ';'
    CharClass::Operator,   // '<'
    CharClass::Operator,   // '='
    CharClass::Operator,   // '>'
    CharClass::Operator,   // '?'
    CharClass::Operator,   // '@'
    
    // 0x41 - 0x5A: Uppercase letters
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter,
    
    // 0x5B - 0x60: More symbols
    CharClass::Delimiter,  // '['
    CharClass::Operator,   // '\'
    CharClass::Delimiter,  // ']'
    CharClass::Operator,   // '^'
    CharClass::Letter,     // '_' (treated as letter for identifiers)
    CharClass::Quote,      // '`'
    
    // 0x61 - 0x7A: Lowercase letters
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter, CharClass::Letter, CharClass::Letter,
    CharClass::Letter, CharClass::Letter,
    
    // 0x7B - 0x7F: Final symbols
    CharClass::Delimiter,  // '{'
    CharClass::Operator,   // '|'
    CharClass::Delimiter,  // '}'
    CharClass::Operator,   // '~'
    CharClass::Invalid,
    
    // 0x80 - 0xFF: Extended ASCII (treat as invalid for now)
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid,
    CharClass::Invalid, CharClass::Invalid, CharClass::Invalid, CharClass::Invalid
};

// Initialize the constant memory lookup table
__host__ void initialize_char_class_table() {
    cudaMemcpyToSymbol(char_class_table, host_char_class_table, 
                       sizeof(host_char_class_table));
}

// Device function to classify a character
__device__ CharClass classify_char(unsigned char ch) {
    return char_class_table[ch];
}

// Kernel to classify all characters in a string
__global__ void classify_chars_kernel(
    const char* source,
    size_t source_len,
    CharClass* output
) {
    // Calculate global thread ID
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes multiple characters for better efficiency
    const uint32_t chars_per_thread = 4;
    uint32_t start_idx = tid * chars_per_thread;
    
    // Bounds check
    if (start_idx >= source_len) return;
    
    // Process characters with coalesced memory access
    for (uint32_t i = 0; i < chars_per_thread && (start_idx + i) < source_len; ++i) {
        uint32_t idx = start_idx + i;
        unsigned char ch = static_cast<unsigned char>(source[idx]);
        output[idx] = classify_char(ch);
    }
}

// Helper function to launch the character classifier kernel
extern "C" void launch_char_classifier_kernel(
    const char* source,
    size_t source_len,
    CharClass* classes
) {
    // Initialize lookup table on first use
    static bool initialized = false;
    if (!initialized) {
        initialize_char_class_table();
        initialized = true;
    }
    
    // Configure kernel launch parameters
    const uint32_t chars_per_thread = 4;
    const uint32_t threads_per_block = 256;
    uint32_t num_threads = (source_len + chars_per_thread - 1) / chars_per_thread;
    uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    classify_chars_kernel<<<num_blocks, threads_per_block>>>(
        source, source_len, classes
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Log error (in production, handle more gracefully)
#ifdef __CUDA_ARCH__
        // Device code: Use cuPrintf or omit for performance
        // printf("CUDA error in classify_chars_kernel: %d\n", err);
#else
        printf("CUDA error in classify_chars_kernel: %s\n", cudaGetErrorString(err));
#endif
    }
}

} // namespace rustg