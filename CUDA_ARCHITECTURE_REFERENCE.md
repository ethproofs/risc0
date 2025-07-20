# CUDA Architecture Quick Reference

## Architecture Detection & Optimization Matrix

### Compute Capability Mapping

| Architecture | Compute Capability | Common GPUs | Optimization Strategy |
|-------------|-------------------|-------------|----------------------|
| **Volta** | 7.0 | V100, Titan V | High register availability |
| **Turing** | 7.5 | RTX 2080, Quadro RTX | Balanced approach |
| **Ampere** | 8.0-8.6 | A100, RTX 3090, A4500 | Register pressure aware |
| **Ada Lovelace** | 8.9 | RTX 4090, RTX 6000 Ada | Similar to Ampere |
| **Blackwell** | 10.x | H100, H200 | Conservative new arch |

### Block Size Configuration Matrix

| Kernel Type | Volta | Turing | Ampere | Ada | Blackwell |
|------------|-------|--------|--------|-----|-----------|
| **NTT/FFT** | 384 | 320 | 256 | 256 | 384 |
| **Witness Generation** | 256 | 224 | 192 | 192 | 256 |
| **Accumulation** | 320 | 320 | 256 | 256 | 256 |
| **Hash/Memory** | 128 | 128 | 128 | 128 | 128 |

### Grid Sizing Multipliers

| Architecture | SM Multiplier | Reasoning |
|-------------|---------------|-----------|
| **Volta** | 4× | High concurrency capability |
| **Turing** | 3× | Balanced approach |
| **Ampere** | 2× | Conservative due to register pressure |
| **Ada Lovelace** | 3× | Slightly more aggressive than Ampere |
| **Blackwell** | 2× | Conservative for new architecture |

### Max Safe Threads per Block

| Architecture | Max Safe Threads | Hardware Limit | Safety Margin |
|-------------|------------------|----------------|---------------|
| **Volta** | 768 | 1024 | 25% |
| **Turing** | 768 | 1024 | 25% |
| **Ampere** | 512 | 1024 | 50% |
| **Ada Lovelace** | 512 | 1024 | 50% |
| **Blackwell** | 768 | 1024+ | 25% |

## Performance Expectations

### Expected GPU Utilization

| Architecture | Target Utilization | Typical Range |
|-------------|-------------------|---------------|
| **Volta** | 90% | 85-95% |
| **Turing** | 85% | 80-90% |
| **Ampere** | 87% | 80-95% |
| **Ada Lovelace** | 90% | 85-95% |
| **Blackwell** | 85% | 80-90% |

### Performance Relative to Hardware Peak

| Architecture | Expected Performance | Notes |
|-------------|---------------------|-------|
| **Volta** | 95-98% | Excellent register utilization |
| **Turing** | 92-96% | Balanced performance |
| **Ampere** | 90-95% | Conservative due to register limits |
| **Ada Lovelace** | 92-97% | High efficiency |
| **Blackwell** | 90-95% | Conservative initial tuning |

## Monitoring Commands by Architecture

### Volta (V100)

```bash
# High register availability - monitor occupancy
ncu --metrics smsp__occupancy_pct.avg
nvidia-smi dmon -s u
```

### Turing (RTX 2080)

```bash
# Balanced monitoring
ncu --metrics smsp__occupancy_pct.avg,smsp__registers_per_thread_allocated.avg
nvidia-smi dmon -s u,m
```

### Ampere (A4500, A100)

```bash
# Focus on register pressure
ncu --metrics smsp__registers_per_thread_allocated.avg,smsp__occupancy_pct.avg
nvidia-smi dmon -s u,m,t
```

### Ada Lovelace (RTX 4090)

```bash
# Similar to Ampere monitoring
ncu --metrics smsp__registers_per_thread_allocated.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed
nvidia-smi dmon -s u,m
```

### Blackwell (H100, H200)

```bash
# Comprehensive monitoring for new architecture
ncu --metrics smsp__occupancy_pct.avg,smsp__registers_per_thread_allocated.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed
nvidia-smi dmon -s u,m,t,p
```

## Troubleshooting by Architecture

### Volta Issues

* **Low occupancy**: Increase block size to 384-512
* **Register spilling**: Reduce complexity, not block size
* **Memory bound**: Focus on coalescing

### Turing Issues

* **Balanced problems**: Adjust between 256-384 threads
* **Power limits**: Monitor temperature and clocks
* **Memory bandwidth**: Optimize access patterns

### Ampere Issues

* **Register pressure**: Keep blocks ≤ 256 for complex kernels
* **Resource errors**: Use fallback to 192 threads
* **Thermal throttling**: Monitor temperature closely

### Ada Lovelace Issues

* **Similar to Ampere**: Register pressure management
* **Memory bandwidth**: Leverage high bandwidth efficiently
* **Power efficiency**: Monitor power consumption

### Blackwell Issues

* **New architecture**: Start conservative, profile extensively
* **Unknown limits**: Use fallback mechanisms
* **Future optimization**: Collect data for tuning

## Quick Diagnostic Commands

```bash
# Identify your GPU architecture
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Check current utilization
nvidia-smi dmon -s u -c 10

# Profile register usage
ncu --metrics smsp__registers_per_thread_allocated.avg your_command

# Check for resource errors
dmesg | grep -i cuda

# Monitor thermal throttling
nvidia-smi dmon -s t,p -c 20
```

## Architecture-Specific Optimization Tips

### Volta Optimization

* Leverage high register count
* Use larger block sizes when possible
* Focus on compute-intensive kernels

### Turing Optimization

* Balance register usage and occupancy
* Optimize memory access patterns
* Use moderate block sizes

### Ampere Optimization

* Conservative register usage
* Monitor for resource exhaustion
* Use smaller block sizes for complex kernels

### Ada Lovelace Optimization

* Similar strategy to Ampere
* Leverage high memory bandwidth
* Monitor power efficiency

### Blackwell Optimization

* Start conservative
* Profile extensively
* Prepare for future tuning opportunities
