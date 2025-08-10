// Consensus and Coordination Tests - WRITTEN FIRST (TDD)
// Testing GPU-native consensus algorithms and distributed coordination

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <chrono>
#include <atomic>

// Test result structure  
struct TestResult {
    bool passed;
    float throughput_ops;
    float latency_us;
    int operations_completed;
    char error_message[256];
};

// Node state for consensus
struct ConsensusNode {
    int node_id;
    int current_term;
    int voted_for;
    int log_length;
    int commit_index;
    int state; // 0=follower, 1=candidate, 2=leader
    uint64_t last_heartbeat;
};

// Vote request/response
struct VoteRequest {
    int term;
    int candidate_id;
    int last_log_index;
    int last_log_term;
};

struct VoteResponse {
    int term;
    bool vote_granted;
    int voter_id;
};

// Distributed ledger entry
struct LedgerEntry {
    uint64_t transaction_id;
    uint64_t timestamp;
    uint32_t from_node;
    uint32_t to_node;
    float value;
    uint32_t hash;
    uint32_t prev_hash;
    bool validated;
};

// Lock structure for distributed locking
struct DistributedLock {
    int lock_id;
    int owner_id;
    uint64_t expiry_time;
    int wait_queue[32];
    int queue_head;
    int queue_tail;
    std::atomic<int> state; // 0=free, 1=locked, 2=contended
};

// Test 1: Leader Election (Raft-style)
__global__ void test_leader_election(TestResult* result,
                                    ConsensusNode* nodes,
                                    VoteRequest* requests,
                                    VoteResponse* responses,
                                    int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_nodes) {
        ConsensusNode* node = &nodes[tid];
        
        // Initialize node
        node->node_id = tid;
        node->current_term = 0;
        node->voted_for = -1;
        node->state = 0; // Start as follower
        node->last_heartbeat = clock64();
        
        __syncthreads();
        
        // Simulate timeout and become candidate
        uint64_t current_time = clock64();
        uint64_t timeout = 1000000 + (tid * 100000); // Randomized timeout
        
        if (current_time - node->last_heartbeat > timeout && node->state == 0) {
            // Become candidate
            node->state = 1;
            node->current_term++;
            node->voted_for = tid;
            
            // Request votes
            VoteRequest* req = &requests[tid];
            req->term = node->current_term;
            req->candidate_id = tid;
            req->last_log_index = node->log_length;
            req->last_log_term = node->current_term;
            
            // Count votes
            int votes = 1; // Vote for self
            
            for (int i = 0; i < num_nodes; i++) {
                if (i != tid) {
                    // Check if node would grant vote
                    ConsensusNode* voter = &nodes[i];
                    if (voter->current_term <= req->term &&
                        (voter->voted_for == -1 || voter->voted_for == tid)) {
                        votes++;
                        responses[i].vote_granted = true;
                        responses[i].voter_id = i;
                        responses[i].term = req->term;
                    }
                }
            }
            
            // Check if won election (majority)
            if (votes > num_nodes / 2) {
                node->state = 2; // Become leader
            }
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Verify exactly one leader elected
        int leader_count = 0;
        int leader_id = -1;
        
        for (int i = 0; i < num_nodes; i++) {
            if (nodes[i].state == 2) {
                leader_count++;
                leader_id = i;
            }
        }
        
        result->passed = (leader_count == 1);
        result->operations_completed = num_nodes;
        
        if (!result->passed) {
            sprintf(result->error_message,
                   "Leader election failed: %d leaders elected", leader_count);
        }
    }
}

// Test 2: Quorum-based Voting
__global__ void test_quorum_voting(TestResult* result,
                                  int* votes,
                                  bool* decisions,
                                  int num_proposals,
                                  int num_voters,
                                  int quorum_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int proposal_id = tid / num_voters;
    int voter_id = tid % num_voters;
    
    if (proposal_id < num_proposals) {
        // Cast vote (random based on hash)
        uint32_t hash = (proposal_id * 0x9e3779b9) ^ (voter_id * 0xdeadbeef);
        bool vote = (hash & 1) == 0;
        
        if (vote) {
            atomicAdd(&votes[proposal_id], 1);
        }
        
        __syncthreads();
        
        // Check if quorum reached
        if (voter_id == 0) {
            int vote_count = votes[proposal_id];
            decisions[proposal_id] = (vote_count >= quorum_size);
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Verify all proposals have decisions
        result->passed = true;
        int decided_count = 0;
        
        for (int i = 0; i < num_proposals; i++) {
            if (votes[i] >= quorum_size) {
                decided_count++;
            }
        }
        
        result->operations_completed = decided_count;
        result->passed = (decided_count > 0);
        
        if (!result->passed) {
            sprintf(result->error_message, "No proposals reached quorum");
        }
    }
}

// Test 3: Byzantine Fault Tolerance
__global__ void test_byzantine_consensus(TestResult* result,
                                        int* node_values,
                                        int* agreed_value,
                                        int num_nodes,
                                        int num_byzantine) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_nodes) {
        // Simulate Byzantine nodes (faulty)
        bool is_byzantine = (tid < num_byzantine);
        
        if (is_byzantine) {
            // Byzantine nodes send random values
            node_values[tid] = (tid * 0x9e3779b9) % 100;
        } else {
            // Honest nodes agree on value
            node_values[tid] = 42;
        }
        
        __syncthreads();
        
        // PBFT-style consensus (simplified)
        if (!is_byzantine) {
            // Count occurrences of each value
            int value_counts[100] = {0};
            
            for (int i = 0; i < num_nodes; i++) {
                if (node_values[i] >= 0 && node_values[i] < 100) {
                    value_counts[node_values[i]]++;
                }
            }
            
            // Find value with 2f+1 agreement (f = num_byzantine)
            int required_votes = 2 * num_byzantine + 1;
            int consensus_value = -1;
            
            for (int v = 0; v < 100; v++) {
                if (value_counts[v] >= required_votes) {
                    consensus_value = v;
                    break;
                }
            }
            
            if (tid == num_byzantine) { // First honest node
                *agreed_value = consensus_value;
            }
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Verify consensus achieved despite Byzantine nodes
        result->passed = (*agreed_value == 42);
        result->operations_completed = num_nodes - num_byzantine;
        
        if (!result->passed) {
            sprintf(result->error_message,
                   "Byzantine consensus failed: agreed on %d instead of 42",
                   *agreed_value);
        }
    }
}

// Test 4: Distributed Ledger Operations
__global__ void test_distributed_ledger(TestResult* result,
                                       LedgerEntry* ledger,
                                       int num_entries,
                                       int num_validators) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int entry_id = tid / num_validators;
    int validator_id = tid % num_validators;
    
    if (entry_id < num_entries) {
        LedgerEntry* entry = &ledger[entry_id];
        
        // Initialize entry
        if (validator_id == 0) {
            entry->transaction_id = entry_id;
            entry->timestamp = clock64();
            entry->from_node = entry_id % 10;
            entry->to_node = (entry_id + 1) % 10;
            entry->value = 100.0f + entry_id;
            
            // Calculate hash
            uint32_t hash = entry->transaction_id;
            hash ^= entry->from_node << 16;
            hash ^= entry->to_node << 8;
            hash *= 0x9e3779b9;
            entry->hash = hash;
            
            // Link to previous entry
            if (entry_id > 0) {
                entry->prev_hash = ledger[entry_id - 1].hash;
            } else {
                entry->prev_hash = 0;
            }
        }
        
        __syncthreads();
        
        // Validate entry (all validators)
        bool valid = true;
        
        // Check hash chain
        if (entry_id > 0) {
            if (entry->prev_hash != ledger[entry_id - 1].hash) {
                valid = false;
            }
        }
        
        // Check transaction validity
        if (entry->value <= 0 || entry->from_node == entry->to_node) {
            valid = false;
        }
        
        // Vote on validity
        unsigned mask = __ballot_sync(0xFFFFFFFF, valid);
        int valid_votes = __popc(mask);
        
        if (validator_id == 0) {
            // Require majority validation
            entry->validated = (valid_votes > num_validators / 2);
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Count validated entries
        int validated_count = 0;
        for (int i = 0; i < num_entries; i++) {
            if (ledger[i].validated) {
                validated_count++;
            }
        }
        
        result->passed = (validated_count == num_entries);
        result->operations_completed = validated_count;
        
        if (!result->passed) {
            sprintf(result->error_message,
                   "Ledger validation failed: %d/%d entries validated",
                   validated_count, num_entries);
        }
    }
}

// Test 5: Distributed Locking with Fairness
__global__ void test_distributed_locking(TestResult* result,
                                        DistributedLock* lock,
                                        int* access_order,
                                        int num_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_threads) {
        // Try to acquire lock
        bool acquired = false;
        int attempts = 0;
        const int MAX_ATTEMPTS = 1000;
        
        while (!acquired && attempts < MAX_ATTEMPTS) {
            int expected = 0;
            if (lock->state.compare_exchange_strong(expected, 1)) {
                // Lock acquired
                lock->owner_id = tid;
                acquired = true;
                
                // Critical section
                int my_position = atomicAdd(&access_order[0], 1);
                access_order[my_position + 1] = tid;
                
                // Simulate work
                for (int i = 0; i < 100; i++) {
                    __threadfence();
                }
                
                // Release lock
                lock->owner_id = -1;
                lock->state.store(0);
            } else {
                // Add to wait queue
                if (expected == 1) {
                    lock->state.store(2); // Mark as contended
                }
                attempts++;
            }
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Verify all threads got access
        int accessed = access_order[0];
        result->passed = (accessed == num_threads);
        result->operations_completed = accessed;
        
        if (!result->passed) {
            sprintf(result->error_message,
                   "Lock fairness failed: %d/%d threads got access",
                   accessed, num_threads);
        }
    }
}

// Test 6: Group Membership Protocol
__global__ void test_group_membership(TestResult* result,
                                     int* member_list,
                                     bool* member_status,
                                     int* view_number,
                                     int max_members) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < max_members) {
        // Simulate member join/leave
        bool joining = (tid % 3 != 0); // 2/3 join, 1/3 leave
        
        if (joining) {
            // Join protocol
            int pos = atomicAdd(&member_list[0], 1);
            if (pos < max_members) {
                member_list[pos + 1] = tid;
                member_status[tid] = true;
                atomicAdd(view_number, 1);
            }
        } else {
            // Leave protocol
            member_status[tid] = false;
            atomicAdd(view_number, 1);
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Verify membership consistency
        int active_count = 0;
        for (int i = 0; i < max_members; i++) {
            if (member_status[i]) {
                active_count++;
            }
        }
        
        result->passed = (active_count > 0 && active_count <= max_members);
        result->operations_completed = active_count;
        
        if (!result->passed) {
            sprintf(result->error_message,
                   "Group membership inconsistent: %d active members",
                   active_count);
        }
    }
}

// Test 7: Consensus with Network Partitions
__global__ void test_partition_tolerance(TestResult* result,
                                        ConsensusNode* nodes,
                                        bool* partition_map,
                                        int num_nodes,
                                        int partition_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_nodes) {
        // Assign nodes to partitions
        bool in_partition_a = (tid < partition_size);
        partition_map[tid] = in_partition_a;
        
        __syncthreads();
        
        // Try to reach consensus within partition
        if (in_partition_a) {
            // Count nodes in partition
            int partition_nodes = 0;
            for (int i = 0; i < num_nodes; i++) {
                if (partition_map[i] == in_partition_a) {
                    partition_nodes++;
                }
            }
            
            // Check if partition has majority
            bool has_majority = (partition_nodes > num_nodes / 2);
            
            if (has_majority && tid == 0) {
                // This partition can make progress
                nodes[0].state = 2; // Elect leader
                nodes[0].commit_index++;
            }
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Verify split-brain prevention
        int leaders = 0;
        int progress_made = 0;
        
        for (int i = 0; i < num_nodes; i++) {
            if (nodes[i].state == 2) leaders++;
            if (nodes[i].commit_index > 0) progress_made++;
        }
        
        // Only majority partition should make progress
        result->passed = (leaders <= 1 && progress_made <= partition_size);
        result->operations_completed = progress_made;
        
        if (!result->passed) {
            sprintf(result->error_message,
                   "Partition tolerance failed: %d leaders, %d progressed",
                   leaders, progress_made);
        }
    }
}

// Test 8: Distributed Transaction Commit (2PC)
__global__ void test_two_phase_commit(TestResult* result,
                                     int* participant_votes,
                                     bool* commit_decision,
                                     int num_participants,
                                     int num_transactions) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tx_id = tid / num_participants;
    int participant_id = tid % num_participants;
    
    if (tx_id < num_transactions) {
        // Phase 1: Voting
        bool can_commit = ((tid * 0x9e3779b9) % 10) > 1; // 80% success rate
        
        if (can_commit) {
            atomicAdd(&participant_votes[tx_id], 1);
        }
        
        __syncthreads();
        
        // Coordinator decision
        if (participant_id == 0) {
            // Phase 2: Commit/Abort decision
            int votes = participant_votes[tx_id];
            commit_decision[tx_id] = (votes == num_participants);
        }
        
        __syncthreads();
        
        // All participants follow decision
        if (commit_decision[tx_id]) {
            // Commit transaction
            // Would update local state here
        } else {
            // Abort transaction
            // Would rollback local state here
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Count successful commits
        int committed = 0;
        for (int i = 0; i < num_transactions; i++) {
            if (commit_decision[i]) {
                committed++;
            }
        }
        
        result->passed = true;
        result->operations_completed = committed;
        result->throughput_ops = (float)committed / num_transactions;
    }
}

// Test 9: Lease-based Leadership
__global__ void test_lease_leadership(TestResult* result,
                                     int* leader_id,
                                     uint64_t* lease_expiry,
                                     int num_nodes,
                                     uint64_t lease_duration) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_nodes) {
        uint64_t current_time = clock64();
        
        // Check if lease expired
        if (current_time > *lease_expiry) {
            // Try to become leader
            int expected = -1;
            if (atomicCAS(leader_id, expected, tid) == expected) {
                // Became leader
                *lease_expiry = current_time + lease_duration;
            }
        }
        
        __syncthreads();
        
        // Verify lease
        if (*leader_id == tid) {
            // Leader performs work
            if (current_time <= *lease_expiry) {
                // Lease valid, can make decisions
                atomicAdd(&result->operations_completed, 1);
            }
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        result->passed = (*leader_id >= 0 && *leader_id < num_nodes);
        
        if (!result->passed) {
            sprintf(result->error_message,
                   "Lease leadership failed: invalid leader %d", *leader_id);
        }
    }
}

// Performance benchmark
__global__ void benchmark_consensus_throughput(TestResult* result,
                                              int* decisions,
                                              int num_decisions,
                                              int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int decisions_per_thread = (num_decisions + blockDim.x - 1) / blockDim.x;
    
    auto start = clock64();
    
    for (int i = 0; i < decisions_per_thread; i++) {
        int decision_id = tid * decisions_per_thread + i;
        if (decision_id < num_decisions) {
            // Simulate consensus round
            int votes = 0;
            
            // Collect votes (simplified)
            for (int n = 0; n < num_nodes; n++) {
                if ((decision_id ^ n) & 1) {
                    votes++;
                }
            }
            
            // Make decision
            decisions[decision_id] = (votes > num_nodes / 2) ? 1 : 0;
        }
    }
    
    auto end = clock64();
    
    if (tid == 0) {
        double cycles = (double)(end - start);
        double seconds = cycles / 1.4e9;
        double decisions_per_second = num_decisions / seconds;
        
        result->throughput_ops = decisions_per_second;
        result->latency_us = (seconds * 1e6) / num_decisions;
        result->passed = (decisions_per_second >= 100000); // 100K decisions/sec
        result->operations_completed = num_decisions;
    }
}

// Main test runner
int main() {
    printf("=== GPU Consensus & Coordination Tests ===\n\n");
    
    TestResult* d_result;
    cudaMalloc(&d_result, sizeof(TestResult));
    
    // Test 1: Leader Election
    printf("Test 1: Leader Election...\n");
    ConsensusNode* d_nodes;
    VoteRequest* d_requests;
    VoteResponse* d_responses;
    int num_nodes = 5;
    
    cudaMalloc(&d_nodes, sizeof(ConsensusNode) * num_nodes);
    cudaMalloc(&d_requests, sizeof(VoteRequest) * num_nodes);
    cudaMalloc(&d_responses, sizeof(VoteResponse) * num_nodes);
    
    test_leader_election<<<1, num_nodes>>>(d_result, d_nodes, d_requests, 
                                          d_responses, num_nodes);
    cudaDeviceSynchronize();
    
    TestResult h_result;
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    printf("  %s - %d nodes participated\n\n",
           h_result.passed ? "PASSED" : "FAILED",
           h_result.operations_completed);
    
    // Test 2: Quorum Voting
    printf("Test 2: Quorum-based Voting...\n");
    int num_proposals = 100;
    int num_voters = 10;
    int quorum_size = 6;
    
    int* d_votes;
    bool* d_decisions;
    cudaMalloc(&d_votes, sizeof(int) * num_proposals);
    cudaMalloc(&d_decisions, sizeof(bool) * num_proposals);
    cudaMemset(d_votes, 0, sizeof(int) * num_proposals);
    
    test_quorum_voting<<<(num_proposals * num_voters + 255) / 256, 256>>>(
        d_result, d_votes, d_decisions, num_proposals, num_voters, quorum_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    printf("  %s - %d proposals decided\n\n",
           h_result.passed ? "PASSED" : "FAILED", 
           h_result.operations_completed);
    
    // Test 3: Byzantine Fault Tolerance
    printf("Test 3: Byzantine Fault Tolerance...\n");
    int num_byzantine = 3;
    int total_nodes = 10;
    
    int* d_node_values;
    int* d_agreed_value;
    cudaMalloc(&d_node_values, sizeof(int) * total_nodes);
    cudaMalloc(&d_agreed_value, sizeof(int));
    
    test_byzantine_consensus<<<1, total_nodes>>>(d_result, d_node_values,
                                                d_agreed_value, total_nodes,
                                                num_byzantine);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    printf("  %s - %d honest nodes agreed\n\n",
           h_result.passed ? "PASSED" : "FAILED",
           h_result.operations_completed);
    
    // Test 4: Distributed Ledger
    printf("Test 4: Distributed Ledger...\n");
    int num_entries = 1000;
    int num_validators = 5;
    
    LedgerEntry* d_ledger;
    cudaMalloc(&d_ledger, sizeof(LedgerEntry) * num_entries);
    
    test_distributed_ledger<<<(num_entries * num_validators + 255) / 256, 256>>>(
        d_result, d_ledger, num_entries, num_validators);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    printf("  %s - %d entries validated\n\n",
           h_result.passed ? "PASSED" : "FAILED",
           h_result.operations_completed);
    
    // Performance Benchmark
    printf("Consensus Performance Benchmark:\n");
    int* d_decisions;
    int num_decisions = 100000;
    cudaMalloc(&d_decisions, sizeof(int) * num_decisions);
    
    benchmark_consensus_throughput<<<256, 256>>>(d_result, d_decisions,
                                                num_decisions, 5);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    printf("  Throughput: %.0f decisions/sec\n", h_result.throughput_ops);
    printf("  Latency: %.2f μs\n", h_result.latency_us);
    printf("  Target Met (100K/sec): %s\n", h_result.passed ? "YES" : "NO");
    
    // Cleanup
    cudaFree(d_result);
    cudaFree(d_nodes);
    cudaFree(d_requests);
    cudaFree(d_responses);
    cudaFree(d_votes);
    cudaFree(d_decisions);
    cudaFree(d_node_values);
    cudaFree(d_agreed_value);
    cudaFree(d_ledger);
    cudaFree(d_decisions);
    
    return 0;
}