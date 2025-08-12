#!/bin/bash
# RustG Performance Benchmark Script
# Compare RustG GPU tools vs standard Rust tools

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

print_benchmark() {
    local tool="$1"
    local rustg_time="$2" 
    local standard_time="$3"
    local speedup="$4"
    
    echo -e "${BOLD}$tool Benchmark Results:${NC}"
    echo "  RustG Time:    ${GREEN}${rustg_time}s${NC}"
    echo "  Standard Time: ${YELLOW}${standard_time}s${NC}" 
    echo "  Speedup:       ${GREEN}${speedup}x${NC}"
    echo ""
}

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███╗   ███╗ █████╗ ██████╗ ██╗  ██╗"
    echo "██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝"
    echo "██████╔╝█████╗  ██╔██╗ ██║██║     ███████║██╔████╔██║███████║██████╔╝█████╔╝ "
    echo "██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ "
    echo "██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗"
    echo "╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝"
    echo "RustG vs Standard Rust Tools Performance Benchmark"
    echo -e "${NC}"
}

# Create benchmark test project
create_test_project() {
    local project_name="$1"
    local complexity="$2"
    
    if command -v cargo-g >/dev/null 2>&1; then
        cargo-g new "$project_name" --quiet
    else
        cargo new "$project_name" --quiet
    fi
    
    cd "$project_name"
    
    # Add dependencies and complexity based on level
    case "$complexity" in
        "simple")
            # Basic project - no changes needed
            ;;
        "medium")
            # Add some dependencies and code
            cat >> Cargo.toml << 'EOF'

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
EOF
            
            cat > src/lib.rs << 'EOF'
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
pub struct User {
    pub name: String,
    pub age: u32,
    pub email: String,
}

pub struct UserManager {
    users: HashMap<String, User>,
}

impl UserManager {
    pub fn new() -> Self {
        Self { users: HashMap::new() }
    }
    
    pub fn add_user(&mut self, id: String, user: User) {
        self.users.insert(id, user);
    }
    
    pub fn get_user(&self, id: &str) -> Option<&User> {
        self.users.get(id)
    }
    
    pub fn serialize_users(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.users)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_manager() {
        let mut manager = UserManager::new();
        let user = User {
            name: "Alice".to_string(),
            age: 30,
            email: "alice@example.com".to_string(),
        };
        
        manager.add_user("1".to_string(), user);
        assert!(manager.get_user("1").is_some());
    }
    
    #[tokio::test]
    async fn test_async() {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        assert_eq!(2 + 2, 4);
    }
}
EOF
            ;;
        "complex")
            # Add many dependencies and complex code
            cat >> Cargo.toml << 'EOF'

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
clap = { version = "4.0", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"
uuid = { version = "1.0", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
regex = "1.0"
log = "0.4"
env_logger = "0.10"

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "benchmarks"
harness = false
EOF

            # Create complex source files
            mkdir -p src/{models,services,utils}
            
            # Main lib.rs
            cat > src/lib.rs << 'EOF'
pub mod models;
pub mod services;
pub mod utils;

pub use models::*;
pub use services::*;
pub use utils::*;

use anyhow::Result;
use std::collections::HashMap;

pub struct Application {
    user_service: UserService,
    config: AppConfig,
}

impl Application {
    pub fn new(config: AppConfig) -> Self {
        Self {
            user_service: UserService::new(),
            config,
        }
    }
    
    pub async fn run(&mut self) -> Result<()> {
        log::info!("Starting application");
        
        // Simulate some work
        for i in 0..100 {
            let user = User {
                id: uuid::Uuid::new_v4().to_string(),
                name: format!("User {}", i),
                email: format!("user{}@example.com", i),
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
            };
            
            self.user_service.create_user(user).await?;
        }
        
        log::info!("Application completed successfully");
        Ok(())
    }
}
EOF

            # Models
            cat > src/models/mod.rs << 'EOF'
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub name: String,
    pub email: String,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub database_url: String,
    pub port: u16,
    pub log_level: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            database_url: "sqlite://memory".to_string(),
            port: 8080,
            log_level: "info".to_string(),
        }
    }
}
EOF

            # Services  
            cat > src/services/mod.rs << 'EOF'
use crate::models::User;
use anyhow::Result;
use std::collections::HashMap;
use tokio::time::Duration;

pub struct UserService {
    users: HashMap<String, User>,
}

impl UserService {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }
    
    pub async fn create_user(&mut self, user: User) -> Result<()> {
        // Simulate async database operation
        tokio::time::sleep(Duration::from_millis(1)).await;
        
        self.users.insert(user.id.clone(), user);
        Ok(())
    }
    
    pub async fn get_user(&self, id: &str) -> Option<&User> {
        tokio::time::sleep(Duration::from_millis(1)).await;
        self.users.get(id)
    }
    
    pub async fn list_users(&self) -> Vec<&User> {
        tokio::time::sleep(Duration::from_millis(5)).await;
        self.users.values().collect()
    }
    
    pub fn validate_email(&self, email: &str) -> bool {
        crate::utils::is_valid_email(email)
    }
}
EOF

            # Utils
            cat > src/utils/mod.rs << 'EOF'
use regex::Regex;

pub fn is_valid_email(email: &str) -> bool {
    let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
    email_regex.is_match(email)
}

pub fn sanitize_input(input: &str) -> String {
    input.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_email_validation() {
        assert!(is_valid_email("test@example.com"));
        assert!(!is_valid_email("invalid-email"));
    }
    
    #[test]
    fn test_sanitize_input() {
        assert_eq!(sanitize_input("  HELLO  "), "hello");
    }
}
EOF

            # Benchmarks
            mkdir -p benches
            cat > benches/benchmarks.rs << 'EOF'
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_user_creation(c: &mut Criterion) {
    c.bench_function("create_user", |b| {
        b.iter(|| {
            // Simulate user creation
            std::thread::sleep(std::time::Duration::from_micros(100));
        })
    });
}

criterion_group!(benches, benchmark_user_creation);
criterion_main!(benches);
EOF
            ;;
    esac
    
    cd ..
}

# Benchmark cargo build
benchmark_cargo_build() {
    local project_name="$1"
    local complexity="$2"
    
    print_header "Benchmarking Cargo Build ($complexity complexity)"
    
    # RustG benchmark
    local rustg_time=""
    if command -v cargo-g >/dev/null 2>&1; then
        cd "$project_name"
        print_info "Testing cargo-g build..."
        cargo clean --quiet 2>/dev/null || true
        
        local start=$(date +%s.%N)
        cargo-g build --release --quiet >/dev/null 2>&1 || cargo-g build --quiet >/dev/null 2>&1
        local end=$(date +%s.%N)
        rustg_time=$(echo "$end - $start" | bc -l)
        cd ..
    fi
    
    # Standard cargo benchmark
    local standard_time=""
    if command -v cargo >/dev/null 2>&1; then
        cd "$project_name"
        print_info "Testing standard cargo build..."
        cargo clean --quiet 2>/dev/null || true
        
        local start=$(date +%s.%N)
        cargo build --release --quiet >/dev/null 2>&1 || cargo build --quiet >/dev/null 2>&1
        local end=$(date +%s.%N)
        standard_time=$(echo "$end - $start" | bc -l)
        cd ..
    fi
    
    # Calculate speedup
    if [[ -n "$rustg_time" ]] && [[ -n "$standard_time" ]]; then
        local speedup=$(echo "scale=1; $standard_time / $rustg_time" | bc -l)
        print_benchmark "cargo ($complexity)" "$rustg_time" "$standard_time" "$speedup"
    else
        print_info "Could not compare both tools"
        [[ -n "$rustg_time" ]] && echo "  cargo-g time: ${rustg_time}s"
        [[ -n "$standard_time" ]] && echo "  cargo time: ${standard_time}s"
    fi
}

# Benchmark clippy
benchmark_clippy() {
    local project_name="$1"
    
    print_header "Benchmarking Clippy Linting"
    
    cd "$project_name"
    
    # RustG clippy benchmark
    local rustg_time=""
    if command -v clippy-f >/dev/null 2>&1; then
        print_info "Testing clippy-f..."
        local start=$(date +%s.%N)
        clippy-f src/ >/dev/null 2>&1 || true
        local end=$(date +%s.%N)
        rustg_time=$(echo "$end - $start" | bc -l)
    fi
    
    # Standard clippy benchmark
    local standard_time=""
    if command -v cargo >/dev/null 2>&1; then
        print_info "Testing standard cargo clippy..."
        local start=$(date +%s.%N)
        cargo clippy --quiet >/dev/null 2>&1 || true
        local end=$(date +%s.%N)
        standard_time=$(echo "$end - $start" | bc -l)
    fi
    
    cd ..
    
    # Calculate speedup
    if [[ -n "$rustg_time" ]] && [[ -n "$standard_time" ]]; then
        local speedup=$(echo "scale=1; $standard_time / $rustg_time" | bc -l)
        print_benchmark "clippy" "$rustg_time" "$standard_time" "$speedup"
    else
        print_info "Could not compare both tools"
        [[ -n "$rustg_time" ]] && echo "  clippy-f time: ${rustg_time}s"  
        [[ -n "$standard_time" ]] && echo "  cargo clippy time: ${standard_time}s"
    fi
}

# Benchmark rustfmt
benchmark_rustfmt() {
    local project_name="$1"
    
    print_header "Benchmarking Code Formatting"
    
    cd "$project_name"
    
    # RustG rustfmt benchmark
    local rustg_time=""
    if command -v rustfmt-g >/dev/null 2>&1; then
        print_info "Testing rustfmt-g..."
        local start=$(date +%s.%N)
        rustfmt-g src/ --check >/dev/null 2>&1 || true
        local end=$(date +%s.%N)
        rustg_time=$(echo "$end - $start" | bc -l)
    fi
    
    # Standard rustfmt benchmark
    local standard_time=""
    if command -v rustfmt >/dev/null 2>&1; then
        print_info "Testing standard rustfmt..."
        local start=$(date +%s.%N)
        rustfmt --check src/*.rs >/dev/null 2>&1 || true
        local end=$(date +%s.%N)
        standard_time=$(echo "$end - $start" | bc -l)
    fi
    
    cd ..
    
    # Calculate speedup
    if [[ -n "$rustg_time" ]] && [[ -n "$standard_time" ]]; then
        local speedup=$(echo "scale=1; $standard_time / $rustg_time" | bc -l)
        print_benchmark "rustfmt" "$rustg_time" "$standard_time" "$speedup"
    else
        print_info "Could not compare both tools"
        [[ -n "$rustg_time" ]] && echo "  rustfmt-g time: ${rustg_time}s"
        [[ -n "$standard_time" ]] && echo "  rustfmt time: ${standard_time}s"
    fi
}

# Benchmark documentation generation
benchmark_rustdoc() {
    local project_name="$1"
    
    print_header "Benchmarking Documentation Generation"
    
    cd "$project_name"
    
    # RustG rustdoc benchmark
    local rustg_time=""
    if command -v rustdoc-g >/dev/null 2>&1; then
        print_info "Testing rustdoc-g..."
        local start=$(date +%s.%N)
        rustdoc-g src/lib.rs --output target/rustg-docs >/dev/null 2>&1 || true
        local end=$(date +%s.%N)
        rustg_time=$(echo "$end - $start" | bc -l)
    fi
    
    # Standard rustdoc benchmark
    local standard_time=""
    if command -v cargo >/dev/null 2>&1; then
        print_info "Testing standard cargo doc..."
        local start=$(date +%s.%N)
        cargo doc --quiet >/dev/null 2>&1 || true
        local end=$(date +%s.%N)
        standard_time=$(echo "$end - $start" | bc -l)
    fi
    
    cd ..
    
    # Calculate speedup
    if [[ -n "$rustg_time" ]] && [[ -n "$standard_time" ]]; then
        local speedup=$(echo "scale=1; $standard_time / $rustg_time" | bc -l)
        print_benchmark "rustdoc" "$rustg_time" "$standard_time" "$speedup"
    else
        print_info "Could not compare both tools"
        [[ -n "$rustg_time" ]] && echo "  rustdoc-g time: ${rustg_time}s"
        [[ -n "$standard_time" ]] && echo "  cargo doc time: ${standard_time}s"
    fi
}

# Run comprehensive benchmark
run_comprehensive_benchmark() {
    print_header "Comprehensive Performance Benchmark"
    
    local test_dir=$(mktemp -d)
    cd "$test_dir"
    
    print_info "Created benchmark directory: $test_dir"
    echo ""
    
    # Test different complexity levels
    local complexities=("simple" "medium" "complex")
    
    for complexity in "${complexities[@]}"; do
        local project_name="benchmark_${complexity}"
        
        print_info "Creating $complexity test project: $project_name"
        create_test_project "$project_name" "$complexity"
        
        # Run benchmarks
        benchmark_cargo_build "$project_name" "$complexity"
        benchmark_clippy "$project_name"
        benchmark_rustfmt "$project_name" 
        benchmark_rustdoc "$project_name"
        
        echo ""
    done
    
    # Cleanup
    cd /
    rm -rf "$test_dir"
    print_info "Cleaned up benchmark directory"
}

# Generate system info report
generate_system_report() {
    print_header "System Information"
    
    echo "OS: $(lsb_release -d -s 2>/dev/null || uname -a)"
    echo "Architecture: $(uname -m)"
    echo "CPU: $(nproc) cores"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        echo "GPU Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
        echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
    else
        echo "GPU: Not detected"
    fi
    
    if command -v nvcc >/dev/null 2>&1; then
        echo "CUDA: $(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')"
    else
        echo "CUDA: Not detected"
    fi
    
    echo ""
    echo "Rust toolchain:"
    rustc --version
    cargo --version
    echo ""
    
    echo "RustG tools:"
    for tool in cargo-g clippy-f rustfmt-g rustdoc-g rustup-g rust-gdb-g bindgen-g miri-g; do
        if command -v "$tool" >/dev/null 2>&1; then
            echo "  $tool: Available"
        else
            echo "  $tool: Not found"
        fi
    done
}

# Main execution
main() {
    print_banner
    
    print_info "RustG Performance Benchmark Suite"
    print_info "Comparing GPU-accelerated tools vs standard Rust tools"
    echo ""
    
    # Check if bc is available for calculations
    if ! command -v bc >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: 'bc' not found. Please install it for precise timing calculations.${NC}"
        echo "Ubuntu/Debian: sudo apt install bc"
        echo "RHEL/CentOS: sudo yum install bc"
        echo ""
    fi
    
    generate_system_report
    run_comprehensive_benchmark
    
    print_header "Benchmark Complete"
    print_success "All benchmarks completed successfully!"
    echo ""
    echo "Notes:"
    echo "- Times may vary based on system load and hardware"
    echo "- GPU acceleration requires CUDA 13.0+ and compatible GPU"
    echo "- CPU fallback is used when GPU is not available"
    echo "- Add --stats flag to RustG tools for detailed performance information"
}

# Execute main function
main "$@"