# thePuzzler - Future Enhancements

## Brainwallet Improvements

### WarpWallet Support
- [ ] Implement scrypt GPU kernel (memory-hard, N=2^18, r=8, p=1)
- [ ] Implement PBKDF2-SHA256 with 2^16 iterations
- [ ] XOR combination of scrypt + PBKDF2 outputs
- [ ] Salt support (email-based)
- [ ] Expected speed: ~1,000-10,000 keys/sec per GPU (vs billions for SHA256)

### BrainV2 Support
- [ ] PBKDF2-SHA512 with configurable iterations (100,000+)
- [ ] Argon2id support (memory-hard, highly resistant to GPU)
- [ ] Salt parameter support
- [ ] Expected speed: ~100-100,000 keys/sec depending on parameters

### Brainwallet Detection
- [ ] Auto-detect brainwallet type from address patterns
- [ ] Support mixed-mode scanning (SHA256 + WarpWallet + BrainV2)

---

## Performance Optimizations

### Double-Buffering Pipeline
- [ ] Implement true async double-buffering for brainwallet mode
- [ ] Overlap GPU compute with CPU passphrase generation
- [ ] Use CUDA streams and events for pipelining

### Multi-GPU Improvements
- [ ] Parallelize bloom filter loading across GPUs
- [ ] Auto-tune batch size per GPU based on memory/compute ratio
- [ ] Better load balancing for heterogeneous GPU configurations

---

## User Experience

### Web Interface
- [ ] Local web UI for monitoring progress
- [ ] Real-time charts for speed/progress
- [ ] Remote monitoring capability

### Distributed Solving
- [ ] Server mode for coordinating multiple machines
- [ ] Work unit distribution and result collection
- [ ] Checkpoint synchronization across nodes

---

## Algorithm Enhancements

### Kangaroo Improvements
- [ ] Tame kangaroo pre-generation and caching
- [ ] Work file import/export (JLP compatibility)
- [ ] Distributed Kangaroo with central DP server

### PCFG Enhancements
- [ ] Train custom PCFG models from leaked password databases
- [ ] Markov chain integration for better passphrase ordering
- [ ] Context-aware generation (crypto-specific patterns)

---

## Platform Support

### OpenCL Backend
- [ ] AMD GPU support via OpenCL
- [ ] Intel GPU support
- [ ] Cross-platform fallback when CUDA unavailable

### ARM Support
- [ ] Optimize Metal backend for Apple Silicon
- [ ] NEON SIMD optimizations for CPU fallback
