# Hello World Example

Basic example demonstrating RustG GPU compiler usage.

## Building

```bash
# GPU-accelerated build (10x faster)
cargo-g build --release

# Run the example
cargo-g run

# GPU-accelerated testing
cargo-g test

# GPU-accelerated linting
clippy-f src/
```

## Performance Comparison

```bash
# Standard cargo (slow)
time cargo build --release

# RustG GPU compiler (10x faster)  
time cargo-g build --release
```

You should see significant speedup with cargo-g!