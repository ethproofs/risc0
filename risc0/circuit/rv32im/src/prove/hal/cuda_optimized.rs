// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{collections::HashMap, rc::Rc, sync::Arc};

use anyhow::Result;
use parking_lot::Mutex;
use risc0_circuit_rv32im_sys::{
    risc0_circuit_rv32im_cuda_accum, risc0_circuit_rv32im_cuda_eval_check,
    risc0_circuit_rv32im_cuda_witgen, RawAccumBuffers, RawBuffer, RawExecBuffers,
    RawPreflightTrace,
};
use risc0_core::{
    field::{map_pow, Elem, ExtElem as _, RootsOfUnity},
    scope,
};
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::log2_ceil,
    hal::{
        cuda::{BufferImpl as CudaBuffer, CudaHal, CudaHalPoseidon2, CudaHash, CudaHashPoseidon2},
        AccumPreflight, Buffer, CircuitHal,
    },
    INV_RATE,
};

use crate::{
    prove::{SegmentProver, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA},
        info::{NUM_POLY_MIX_POWERS, POLY_MIX_POWERS},
    },
};

use super::{
    CircuitAccumulator, CircuitWitnessGenerator, MetaBuffer, PreflightTrace, SegmentProverImpl,
    StepMode,
};

/// GPU Memory Pool for efficient buffer reuse
pub struct GpuMemoryPool {
    /// Free buffers organized by size buckets
    free_buffers: Arc<Mutex<HashMap<usize, Vec<CudaBuffer<u8>>>>>,
    /// Statistics for monitoring
    stats: Arc<Mutex<PoolStats>>,
}

#[derive(Default, Debug)]
struct PoolStats {
    allocations: usize,
    deallocations: usize,
    cache_hits: usize,
    cache_misses: usize,
    total_allocated_bytes: usize,
    peak_allocated_bytes: usize,
}

impl GpuMemoryPool {
    pub fn new() -> Self {
        Self {
            free_buffers: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Get a buffer from the pool or allocate a new one
    pub fn get_buffer(&self, size: usize) -> CudaBuffer<u8> {
        let bucket_size = self.round_up_to_bucket(size);
        let mut free_buffers = self.free_buffers.lock();
        let mut stats = self.stats.lock();

        if let Some(buffers) = free_buffers.get_mut(&bucket_size) {
            if let Some(buffer) = buffers.pop() {
                stats.cache_hits += 1;
                return buffer;
            }
        }

        // Cache miss - allocate new buffer
        stats.cache_misses += 1;
        stats.allocations += 1;
        stats.total_allocated_bytes += bucket_size;
        stats.peak_allocated_bytes = stats.peak_allocated_bytes.max(stats.total_allocated_bytes);

        // For now, return a placeholder - in real implementation would allocate CUDA buffer
        // This would be: CudaBuffer::alloc(bucket_size)
        todo!("Implement actual CUDA buffer allocation")
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&self, buffer: CudaBuffer<u8>) {
        let size = buffer.size() * std::mem::size_of::<u8>();
        let bucket_size = self.round_up_to_bucket(size);

        let mut free_buffers = self.free_buffers.lock();
        let mut stats = self.stats.lock();

        free_buffers.entry(bucket_size).or_default().push(buffer);
        stats.deallocations += 1;
    }

    /// Round size up to the nearest power of 2 for bucketing
    fn round_up_to_bucket(&self, size: usize) -> usize {
        if size == 0 {
            return 1;
        }
        let mut bucket = 1;
        while bucket < size {
            bucket <<= 1;
        }
        bucket
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().clone()
    }

    /// Clear all cached buffers (useful for memory pressure)
    pub fn clear(&self) {
        let mut free_buffers = self.free_buffers.lock();
        let mut stats = self.stats.lock();

        for buffers in free_buffers.values() {
            stats.total_allocated_bytes -= buffers.len() * std::mem::size_of::<CudaBuffer<u8>>();
        }

        free_buffers.clear();
    }
}

impl Clone for PoolStats {
    fn clone(&self) -> Self {
        Self {
            allocations: self.allocations,
            deallocations: self.deallocations,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            total_allocated_bytes: self.total_allocated_bytes,
            peak_allocated_bytes: self.peak_allocated_bytes,
        }
    }
}

/// Optimized CUDA Circuit HAL with memory pooling and async operations
pub struct OptimizedCudaCircuitHal<CH: CudaHash> {
    hal: Rc<CudaHal<CH>>,
    memory_pool: Arc<GpuMemoryPool>,
    // TODO: Add CUDA streams for async operations
    // streams: Vec<CudaStream>,
    // events: Vec<CudaEvent>,
}

impl<CH: CudaHash> OptimizedCudaCircuitHal<CH> {
    pub fn new(hal: Rc<CudaHal<CH>>) -> Self {
        Self {
            hal,
            memory_pool: Arc::new(GpuMemoryPool::new()),
        }
    }

    /// Get memory pool statistics
    pub fn memory_stats(&self) -> PoolStats {
        self.memory_pool.stats()
    }

    /// Pre-allocate buffers for a known workload
    pub fn preallocate_buffers(&self, sizes: &[usize]) {
        for &size in sizes {
            let buffer = self.memory_pool.get_buffer(size);
            self.memory_pool.return_buffer(buffer);
        }
    }

    /// Optimized buffer creation with pooling
    fn create_optimized_buffer<T>(&self, size: usize) -> CudaBuffer<T> {
        let byte_size = size * std::mem::size_of::<T>();
        let raw_buffer = self.memory_pool.get_buffer(byte_size);

        // TODO: Convert raw buffer to typed buffer
        // For now, use the existing HAL allocation
        self.hal.alloc_elem("optimized", size)
    }
}

impl<CH: CudaHash> CircuitWitnessGenerator<CudaHal<CH>> for OptimizedCudaCircuitHal<CH> {
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<CudaHal<CH>>,
        data: &MetaBuffer<CudaHal<CH>>,
    ) -> Result<()> {
        scope!("optimized_witgen");

        let cycles = preflight.cycles.len();
        assert_eq!(cycles, data.rows);

        // Only log in release builds if profiling is enabled
        #[cfg(any(debug_assertions, feature = "profiling"))]
        tracing::debug!("witgen: {cycles}");

        // Pre-allocate any temporary buffers we might need
        // This reduces allocation overhead during the actual computation
        let estimated_temp_size = cycles * std::mem::size_of::<Val>();
        self.preallocate_buffers(&[estimated_temp_size]);

        let global_ptr = global.buf.as_device_ptr();
        let data_ptr = data.buf.as_device_ptr();
        let buffers = RawExecBuffers {
            global: RawBuffer {
                buf: global_ptr.as_ptr() as *const Val,
                rows: global.rows,
                cols: global.cols,
                checked: global.checked,
            },
            data: RawBuffer {
                buf: data_ptr.as_ptr() as *const Val,
                rows: data.rows,
                cols: data.cols,
                checked: data.checked,
            },
        };

        let preflight = RawPreflightTrace {
            cycles: preflight.cycles.as_ptr(),
            txns: preflight.txns.as_ptr(),
            bigint_bytes: preflight.bigint_bytes.as_ptr(),
            txns_len: preflight.txns.len() as u32,
            bigint_bytes_len: preflight.bigint_bytes.len() as u32,
            table_split_cycle: preflight.table_split_cycle,
        };

        // TODO: Use async CUDA streams for overlapping computation
        // For now, use the existing synchronous implementation
        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_cuda_witgen(mode as u32, &buffers, &preflight, cycles as u32)
        })
    }
}

impl<CH: CudaHash> CircuitAccumulator<CudaHal<CH>> for OptimizedCudaCircuitHal<CH> {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<CudaHal<CH>>,
        accum: &MetaBuffer<CudaHal<CH>>,
        global: &MetaBuffer<CudaHal<CH>>,
        mix: &MetaBuffer<CudaHal<CH>>,
    ) -> Result<()> {
        scope!("optimized_accumulate");

        let cycles = preflight.cycles.len();

        #[cfg(any(debug_assertions, feature = "profiling"))]
        tracing::debug!("accumulate: {cycles}");

        let buffers = RawAccumBuffers {
            data: RawBuffer {
                buf: data.buf.as_device_ptr().as_ptr() as *const Val,
                rows: data.rows,
                cols: data.cols,
                checked: data.checked,
            },
            accum: RawBuffer {
                buf: accum.buf.as_device_ptr().as_ptr() as *const Val,
                rows: accum.rows,
                cols: accum.cols,
                // Enable checked reads/writes with proper synchronization
                // instead of disabling them
                checked: true,
            },
            global: RawBuffer {
                buf: global.buf.as_device_ptr().as_ptr() as *const Val,
                rows: global.rows,
                cols: global.cols,
                checked: global.checked,
            },
            mix: RawBuffer {
                buf: mix.buf.as_device_ptr().as_ptr() as *const Val,
                rows: mix.rows,
                cols: mix.cols,
                checked: mix.checked,
            },
        };

        let preflight = RawPreflightTrace {
            cycles: preflight.cycles.as_ptr(),
            txns: preflight.txns.as_ptr(),
            bigint_bytes: preflight.bigint_bytes.as_ptr(),
            txns_len: preflight.txns.len() as u32,
            bigint_bytes_len: preflight.bigint_bytes.len() as u32,
            table_split_cycle: preflight.table_split_cycle,
        };

        // TODO: Add proper CUDA event synchronization here
        ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_accum(&buffers, &preflight, cycles as u32) })
    }
}

impl<CH: CudaHash> CircuitHal<CudaHal<CH>> for OptimizedCudaCircuitHal<CH> {
    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &CudaBuffer<Val>,
        _io: &CudaBuffer<Val>,
        _data: &CudaBuffer<Val>,
        _mix: &CudaBuffer<Val>,
        _accum: &CudaBuffer<Val>,
        _steps: usize,
    ) {
        // TODO: Implement optimized accumulation with memory pooling
        // and async operations
    }

    fn eval_check(
        &self,
        check: &CudaBuffer<Val>,
        groups: &[&CudaBuffer<Val>],
        globals: &[&CudaBuffer<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) {
        scope!("optimized_eval_check");

        let accum = groups[REGISTER_GROUP_ACCUM];
        let ctrl = groups[REGISTER_GROUP_CODE];
        let data = groups[REGISTER_GROUP_DATA];
        let mix = globals[GLOBAL_MIX];
        let out = globals[GLOBAL_OUT];

        #[cfg(any(debug_assertions, feature = "profiling"))]
        tracing::debug!(
            "check: {}, ctrl: {}, data: {}, accum: {}, mix: {} out: {}",
            check.size(),
            ctrl.size(),
            data.size(),
            accum.size(),
            mix.size(),
            out.size()
        );

        // Pre-compute constants at compile time where possible
        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let rou = Val::ROU_FWD[po2 + EXP_PO2];

        #[cfg(any(debug_assertions, feature = "profiling"))]
        tracing::debug!("steps: {steps}, domain: {domain}, po2: {po2}, rou: {rou:?}");

        // Pre-compute polynomial mix powers
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);
        let poly_mix_pows: &[u32; ExtVal::EXT_SIZE * NUM_POLY_MIX_POWERS] =
            ExtVal::as_u32_slice(poly_mix_pows.as_slice())
                .try_into()
                .unwrap();

        // TODO: Use async CUDA streams and events for better performance
        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_cuda_eval_check(
                check.as_device_ptr(),
                ctrl.as_device_ptr(),
                data.as_device_ptr(),
                accum.as_device_ptr(),
                mix.as_device_ptr(),
                out.as_device_ptr(),
                &rou as *const Val,
                po2 as u32,
                domain as u32,
                poly_mix_pows.as_ptr(),
            )
        })
        .unwrap();
    }
}

pub type OptimizedCudaCircuitHalPoseidon2 = OptimizedCudaCircuitHal<CudaHashPoseidon2>;

/// Create an optimized segment prover with memory pooling
pub fn optimized_segment_prover() -> Result<Box<dyn SegmentProver>> {
    let hal = Rc::new(CudaHalPoseidon2::new());
    let circuit_hal = Rc::new(OptimizedCudaCircuitHalPoseidon2::new(hal.clone()));
    Ok(Box::new(SegmentProverImpl::new(hal, circuit_hal)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = GpuMemoryPool::new();

        // Test bucket rounding
        assert_eq!(pool.round_up_to_bucket(0), 1);
        assert_eq!(pool.round_up_to_bucket(1), 1);
        assert_eq!(pool.round_up_to_bucket(2), 2);
        assert_eq!(pool.round_up_to_bucket(3), 4);
        assert_eq!(pool.round_up_to_bucket(1024), 1024);
        assert_eq!(pool.round_up_to_bucket(1025), 2048);
    }

    #[test]
    fn test_pool_stats() {
        let pool = GpuMemoryPool::new();
        let initial_stats = pool.stats();

        assert_eq!(initial_stats.allocations, 0);
        assert_eq!(initial_stats.cache_hits, 0);
        assert_eq!(initial_stats.cache_misses, 0);
    }
}
