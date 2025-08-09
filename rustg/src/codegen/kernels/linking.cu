#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Symbol table entry
struct Symbol {
    uint32_t name_hash;
    uint32_t address;
    uint32_t size;
    uint16_t section;
    uint8_t binding;     // LOCAL, GLOBAL, WEAK
    uint8_t type;        // FUNC, OBJECT, SECTION
};

// Relocation entry
struct Relocation {
    uint32_t offset;     // Where to apply relocation
    uint32_t symbol;     // Symbol index
    uint32_t type;       // Relocation type
    int64_t addend;      // Addend for RELA relocations
    uint32_t section;    // Which section this applies to
};

// Section information
struct Section {
    char name[32];
    uint32_t type;       // SHT_PROGBITS, SHT_RELA, etc.
    uint64_t flags;      // SHF_ALLOC, SHF_EXECINSTR, etc.
    uint32_t addr;       // Virtual address
    uint32_t offset;     // File offset
    uint32_t size;       // Section size
    uint32_t align;      // Alignment
};

// Object file representation
struct ObjectFile {
    Section* sections;
    uint32_t num_sections;
    Symbol* symbols;
    uint32_t num_symbols;
    Relocation* relocations;
    uint32_t num_relocations;
    uint8_t* data;       // Raw section data
    uint32_t data_size;
};

// ELF header constants
#define EI_NIDENT 16
#define ET_REL    1
#define ET_EXEC   2
#define ET_DYN    3
#define EM_X86_64 62

// Relocation types for x86_64
#define R_X86_64_NONE     0
#define R_X86_64_64       1    // Direct 64 bit
#define R_X86_64_PC32     2    // PC relative 32 bit signed
#define R_X86_64_GOT32    3    // 32 bit GOT entry
#define R_X86_64_PLT32    4    // 32 bit PLT address
#define R_X86_64_32       10   // Direct 32 bit zero extended
#define R_X86_64_32S      11   // Direct 32 bit sign extended

// ELF header
struct ELFHeader {
    uint8_t e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};

// Shared memory for linking operations
struct LinkingSharedMem {
    uint32_t symbol_addresses[256];  // Resolved addresses
    uint32_t reloc_queue[256];       // Relocations to process
    uint32_t queue_size;
    uint32_t section_offsets[32];    // Section base addresses
    bool resolution_complete;
    uint32_t undefined_symbols[64];
    uint32_t num_undefined;
};

// Hash function for symbol names (using same as previous phases)
__device__ uint32_t symbol_hash(const char* name, uint32_t len) {
    uint32_t hash = 5381;
    for (uint32_t i = 0; i < len; ++i) {
        hash = ((hash << 5) + hash) + name[i];
    }
    return hash;
}

// Symbol resolution kernel
__global__ void symbol_resolution_kernel(
    const ObjectFile* objects,
    uint32_t num_objects,
    Symbol* global_symbols,
    uint32_t* global_addresses,
    uint32_t* num_resolved,
    uint32_t* undefined_symbols
) {
    extern __shared__ char shared_mem_raw[];
    LinkingSharedMem* shared = reinterpret_cast<LinkingSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->num_undefined = 0;
        shared->resolution_complete = false;
    }
    __syncthreads();
    
    // Build global symbol table from all objects
    for (uint32_t obj = 0; obj < num_objects; ++obj) {
        const ObjectFile& object = objects[obj];
        
        // Process symbols in parallel
        for (uint32_t sym = tid; sym < object.num_symbols; sym += blockDim.x) {
            const Symbol& symbol = object.symbols[sym];
            
            // Only process global and weak symbols
            if (symbol.binding == 1 || symbol.binding == 2) { // GLOBAL or WEAK
                
                // Check for existing definition
                bool found_existing = false;
                for (uint32_t g = 0; g < 1000; ++g) { // Limit check
                    if (global_symbols[g].name_hash == symbol.name_hash) {
                        found_existing = true;
                        
                        // Handle symbol conflicts
                        if (global_symbols[g].binding == 2 && symbol.binding == 1) {
                            // Weak replaced by strong
                            global_symbols[g] = symbol;
                            global_addresses[g] = symbol.address;
                        }
                        break;
                    }
                    
                    if (global_symbols[g].name_hash == 0) {
                        // Empty slot - insert
                        if (atomicCAS(&global_symbols[g].name_hash, 0, symbol.name_hash) == 0) {
                            global_symbols[g] = symbol;
                            global_addresses[g] = symbol.address;
                            atomicAdd(num_resolved, 1);
                        }
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Check for undefined symbols
    for (uint32_t obj = 0; obj < num_objects; ++obj) {
        const ObjectFile& object = objects[obj];
        
        for (uint32_t rel = tid; rel < object.num_relocations; rel += blockDim.x) {
            const Relocation& reloc = object.relocations[rel];
            const Symbol& ref_symbol = object.symbols[reloc.symbol];
            
            // Find symbol in global table
            bool found = false;
            for (uint32_t g = 0; g < 1000; ++g) {
                if (global_symbols[g].name_hash == ref_symbol.name_hash) {
                    found = true;
                    break;
                }
            }
            
            if (!found && ref_symbol.binding == 1) { // GLOBAL undefined
                uint32_t idx = atomicAdd(&shared->num_undefined, 1);
                if (idx < 64) {
                    shared->undefined_symbols[idx] = ref_symbol.name_hash;
                }
            }
        }
    }
    __syncthreads();
    
    // Copy undefined symbols to global memory
    for (uint32_t i = tid; i < shared->num_undefined; i += blockDim.x) {
        undefined_symbols[i] = shared->undefined_symbols[i];
    }
}

// Relocation processing kernel
__global__ void relocation_kernel(
    ObjectFile* objects,
    uint32_t num_objects,
    const Symbol* global_symbols,
    const uint32_t* global_addresses,
    uint32_t num_global_symbols,
    uint32_t* relocations_applied
) {
    extern __shared__ char shared_mem_raw[];
    LinkingSharedMem* shared = reinterpret_cast<LinkingSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    
    // Initialize section base addresses
    if (tid == 0) {
        uint32_t base = 0x400000;  // Standard ELF base
        for (uint32_t s = 0; s < 32; ++s) {
            shared->section_offsets[s] = base;
            base += 0x10000;  // 64KB per section
        }
    }
    __syncthreads();
    
    // Process each object file
    for (uint32_t obj = bid; obj < num_objects; obj += gridDim.x) {
        ObjectFile& object = objects[obj];
        
        // Process relocations in parallel
        for (uint32_t rel = tid; rel < object.num_relocations; rel += blockDim.x) {
            Relocation& relocation = object.relocations[rel];
            const Symbol& symbol = object.symbols[relocation.symbol];
            
            // Find symbol address
            uint32_t symbol_addr = 0;
            bool found = false;
            
            for (uint32_t g = 0; g < num_global_symbols; ++g) {
                if (global_symbols[g].name_hash == symbol.name_hash) {
                    symbol_addr = global_addresses[g];
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                continue; // Skip undefined symbols
            }
            
            // Calculate final address
            uint32_t section_base = shared->section_offsets[relocation.section];
            uint32_t patch_location = section_base + relocation.offset;
            
            // Get pointer to patch location
            uint8_t* patch_ptr = object.data + relocation.offset;
            
            // Apply relocation based on type
            switch (relocation.type) {
                case R_X86_64_64: {
                    // Direct 64-bit address
                    uint64_t value = symbol_addr + relocation.addend;
                    *reinterpret_cast<uint64_t*>(patch_ptr) = value;
                    break;
                }
                
                case R_X86_64_PC32: {
                    // PC-relative 32-bit
                    int64_t value = symbol_addr + relocation.addend - patch_location;
                    *reinterpret_cast<int32_t*>(patch_ptr) = static_cast<int32_t>(value);
                    break;
                }
                
                case R_X86_64_32: {
                    // Direct 32-bit (zero-extended)
                    uint32_t value = static_cast<uint32_t>(symbol_addr + relocation.addend);
                    *reinterpret_cast<uint32_t*>(patch_ptr) = value;
                    break;
                }
                
                case R_X86_64_32S: {
                    // Direct 32-bit (sign-extended)
                    int32_t value = static_cast<int32_t>(symbol_addr + relocation.addend);
                    *reinterpret_cast<int32_t*>(patch_ptr) = value;
                    break;
                }
                
                case R_X86_64_PLT32: {
                    // PLT entry (simplified - direct call)
                    int64_t value = symbol_addr + relocation.addend - patch_location;
                    *reinterpret_cast<int32_t*>(patch_ptr) = static_cast<int32_t>(value);
                    break;
                }
            }
            
            atomicAdd(relocations_applied, 1);
        }
    }
}

// ELF file generation kernel
__global__ void elf_generation_kernel(
    const ObjectFile* objects,
    uint32_t num_objects,
    const Section* merged_sections,
    uint32_t num_sections,
    uint8_t* elf_data,
    uint32_t* elf_size
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    
    // Generate ELF header
    if (tid == 0 && bid == 0) {
        ELFHeader* header = reinterpret_cast<ELFHeader*>(elf_data);
        
        // ELF magic number
        header->e_ident[0] = 0x7f;
        header->e_ident[1] = 'E';
        header->e_ident[2] = 'L';
        header->e_ident[3] = 'F';
        header->e_ident[4] = 2;    // 64-bit
        header->e_ident[5] = 1;    // Little endian
        header->e_ident[6] = 1;    // ELF version
        header->e_ident[7] = 0;    // System V ABI
        
        // Fill rest with zeros
        for (int i = 8; i < EI_NIDENT; ++i) {
            header->e_ident[i] = 0;
        }
        
        header->e_type = ET_EXEC;      // Executable file
        header->e_machine = EM_X86_64; // x86_64
        header->e_version = 1;
        header->e_entry = 0x400000;    // Entry point
        header->e_phoff = sizeof(ELFHeader);
        header->e_shoff = sizeof(ELFHeader) + num_sections * 56; // Program headers
        header->e_flags = 0;
        header->e_ehsize = sizeof(ELFHeader);
        header->e_phentsize = 56;      // Program header size
        header->e_phnum = num_sections;
        header->e_shentsize = 64;      // Section header size
        header->e_shnum = num_sections;
        header->e_shstrndx = num_sections - 1; // String table index
        
        *elf_size = sizeof(ELFHeader);
    }
    __syncthreads();
    
    // Copy section data
    for (uint32_t s = tid; s < num_sections; s += blockDim.x) {
        const Section& section = merged_sections[s];
        
        if (section.size > 0) {
            uint32_t offset = section.offset;
            
            // Find source data from objects
            for (uint32_t obj = 0; obj < num_objects; ++obj) {
                const ObjectFile& object = objects[obj];
                
                for (uint32_t os = 0; os < object.num_sections; ++os) {
                    if (object.sections[os].type == section.type) {
                        // Copy section data
                        uint8_t* src = object.data + object.sections[os].offset;
                        uint8_t* dst = elf_data + offset;
                        
                        for (uint32_t i = 0; i < object.sections[os].size; ++i) {
                            dst[i] = src[i];
                        }
                        
                        atomicAdd(elf_size, object.sections[os].size);
                        break;
                    }
                }
            }
        }
    }
}

// Section merging kernel
__global__ void section_merging_kernel(
    const ObjectFile* objects,
    uint32_t num_objects,
    Section* merged_sections,
    uint32_t* num_merged,
    uint32_t* layout_offsets
) {
    __shared__ uint32_t section_map[64];  // Hash -> merged index
    __shared__ uint32_t next_section;
    
    const uint32_t tid = threadIdx.x;
    
    if (tid == 0) {
        next_section = 0;
        for (int i = 0; i < 64; ++i) {
            section_map[i] = UINT32_MAX;
        }
    }
    __syncthreads();
    
    // Collect unique sections
    for (uint32_t obj = 0; obj < num_objects; ++obj) {
        const ObjectFile& object = objects[obj];
        
        for (uint32_t s = tid; s < object.num_sections; s += blockDim.x) {
            const Section& section = object.sections[s];
            
            // Hash section name
            uint32_t hash = symbol_hash(section.name, 32) % 64;
            
            // Check if section already exists
            if (section_map[hash] == UINT32_MAX) {
                // Try to claim this slot
                uint32_t old = atomicCAS(&section_map[hash], UINT32_MAX, next_section);
                if (old == UINT32_MAX) {
                    // Successfully claimed
                    uint32_t idx = atomicAdd(&next_section, 1);
                    if (idx < 32) {
                        merged_sections[idx] = section;
                        merged_sections[idx].offset = 0;  // Will be set later
                        merged_sections[idx].size = 0;    // Will be accumulated
                    }
                }
            }
        }
    }
    __syncthreads();
    
    // Accumulate section sizes
    for (uint32_t obj = 0; obj < num_objects; ++obj) {
        const ObjectFile& object = objects[obj];
        
        for (uint32_t s = tid; s < object.num_sections; s += blockDim.x) {
            const Section& section = object.sections[s];
            uint32_t hash = symbol_hash(section.name, 32) % 64;
            uint32_t merged_idx = section_map[hash];
            
            if (merged_idx != UINT32_MAX && merged_idx < 32) {
                atomicAdd(&merged_sections[merged_idx].size, section.size);
            }
        }
    }
    __syncthreads();
    
    // Calculate layout offsets
    if (tid == 0) {
        uint32_t offset = sizeof(ELFHeader) + next_section * 64; // After headers
        
        for (uint32_t s = 0; s < next_section; ++s) {
            merged_sections[s].offset = offset;
            layout_offsets[s] = offset;
            offset += merged_sections[s].size;
            
            // Align to page boundary for executable sections
            if (merged_sections[s].flags & 0x4) { // SHF_EXECINSTR
                offset = (offset + 4095) & ~4095;
            }
        }
        
        *num_merged = next_section;
    }
}

// Host launchers
extern "C" void launch_symbol_resolution(
    const ObjectFile* objects,
    uint32_t num_objects,
    Symbol* global_symbols,
    uint32_t* global_addresses,
    uint32_t* num_resolved,
    uint32_t* undefined_symbols
) {
    uint32_t threads = 256;
    uint32_t blocks = 1;
    size_t shared_mem = sizeof(LinkingSharedMem);
    
    symbol_resolution_kernel<<<blocks, threads, shared_mem>>>(
        objects, num_objects,
        global_symbols, global_addresses,
        num_resolved, undefined_symbols
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in symbol_resolution: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_relocation_processing(
    ObjectFile* objects,
    uint32_t num_objects,
    const Symbol* global_symbols,
    const uint32_t* global_addresses,
    uint32_t num_global_symbols,
    uint32_t* relocations_applied
) {
    uint32_t threads = 256;
    uint32_t blocks = num_objects;
    size_t shared_mem = sizeof(LinkingSharedMem);
    
    relocation_kernel<<<blocks, threads, shared_mem>>>(
        objects, num_objects,
        global_symbols, global_addresses,
        num_global_symbols, relocations_applied
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in relocation_processing: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_elf_generation(
    const ObjectFile* objects,
    uint32_t num_objects,
    const Section* merged_sections,
    uint32_t num_sections,
    uint8_t* elf_data,
    uint32_t* elf_size
) {
    uint32_t threads = 256;
    uint32_t blocks = 1;
    
    elf_generation_kernel<<<blocks, threads>>>(
        objects, num_objects,
        merged_sections, num_sections,
        elf_data, elf_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in elf_generation: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg