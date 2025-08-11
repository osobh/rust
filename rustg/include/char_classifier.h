#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "gpu_types.h"

namespace rustg {

// Device function declarations for character classification
// These functions are defined in char_classifier.cu and can be called
// from other CUDA compilation units

/**
 * @brief Classify a character using the constant memory lookup table
 * @param ch The character to classify (0-255)
 * @return The character class for the given character
 * 
 * This is a high-performance device function that uses constant memory
 * for O(1) character classification. The lookup table must be initialized
 * by calling initialize_char_class_table() before use.
 */
__device__ CharClass classify_char(unsigned char ch);

/**
 * @brief Initialize the character classification lookup table in constant memory
 * 
 * This function must be called on the host before using classify_char().
 * It copies the classification data to GPU constant memory for fast access.
 * This initialization only needs to be done once per program run.
 */
__host__ void initialize_char_class_table();

/**
 * @brief Launch the character classification kernel
 * @param source Input source code string
 * @param source_len Length of the source string
 * @param classes Output array for character classifications
 * 
 * This function launches a CUDA kernel that classifies all characters
 * in the input string using parallel threads. Each thread processes
 * multiple characters for optimal performance.
 */
extern "C" void launch_char_classifier_kernel(
    const char* source,
    size_t source_len,
    CharClass* classes
);

} // namespace rustg