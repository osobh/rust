/**
 * CUDA Runtime Initialization for Tests
 * Provides proper GPU context initialization for test harness
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Global variable to track CUDA initialization
static bool cuda_initialized = false;

extern "C" {
    /**
     * Initialize CUDA runtime
     * Returns 0 on success, error code otherwise
     */
    int cuda_init() {
        if (cuda_initialized) {
            return 0;  // Already initialized
        }
        
        // Set device to 0 (first GPU)
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to set CUDA device: %s\n", cudaGetErrorString(err));
            return (int)err;
        }
        
        // Force runtime initialization
        cudaFree(0);
        
        // Reset any prior errors
        cudaGetLastError();
        
        cuda_initialized = true;
        return 0;
    }
    
    /**
     * Check if CUDA is initialized
     */
    bool cuda_is_initialized() {
        return cuda_initialized;
    }
    
    /**
     * Get number of available CUDA devices
     */
    int cuda_device_count() {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
            return 0;
        }
        return count;
    }
    
    /**
     * Reset CUDA device
     */
    int cuda_reset() {
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to reset CUDA device: %s\n", cudaGetErrorString(err));
            return (int)err;
        }
        return 0;
    }
}