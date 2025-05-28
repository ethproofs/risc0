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

use cust::device::{Device, DeviceAttribute};
use std::collections::HashMap;

/// Optimized kernel launch configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub shared_mem: usize,
    pub min_blocks_per_sm: u32,
}

/// Kernel configuration optimizer
pub struct KernelOptimizer {
    device_props: DeviceProperties,
    config_cache: HashMap<String, KernelConfig>,
}

#[derive(Debug, Clone)]
struct DeviceProperties {
    max_threads_per_block: u32,
    max_threads_per_sm: u32,
    max_blocks_per_sm: u32,
    shared_mem_per_block: usize,
    shared_mem_per_sm: usize,
    warp_size: u32,
    sm_count: u32,
}

impl KernelOptimizer {
    pub fn new() -> Result<Self, cust::error::CudaError> {
        let device = Device::get_device(0)?;

        let device_props = DeviceProperties {
            max_threads_per_block: device.get_attribute(DeviceAttribute::MaxThreadsPerBlock)? as u32,
            max_threads_per_sm: device.get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor)? as u32,
            max_blocks_per_sm: device.get_attribute(DeviceAttribute::MaxBlocksPerMultiprocessor)? as u32,
            shared_mem_per_block: device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)? as usize,
            shared_mem_per_sm: device.get_attribute(DeviceAttribute::MaxSharedMemoryPerMultiprocessor)? as usize,
            warp_size: device.get_attribute(DeviceAttribute::WarpSize)? as u32,
            sm_count: device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32,
        };

        Ok(Self {
            device_props,
            config_cache: HashMap::new(),
        })
    }

    /// Get optimized configuration for a kernel
    pub fn get_config(&mut self, kernel_name: &str, problem_size: usize, registers_per_thread: u32) -> KernelConfig {
        let cache_key = format!("{}_{}_{}", kernel_name, problem_size, registers_per_thread);

        if let Some(config) = self.config_cache.get(&cache_key) {
            return config.clone();
        }

        let config = self.compute_optimal_config(problem_size, registers_per_thread);
        self.config_cache.insert(cache_key, config.clone());
        config
    }

    fn compute_optimal_config(&self, problem_size: usize, registers_per_thread: u32) -> KernelConfig {
        // Calculate optimal block size based on occupancy
        let max_threads_per_block = self.calculate_max_threads_per_block(registers_per_thread);

        // Find the best block size that's a multiple of warp size
        let mut best_block_size = self.device_props.warp_size;
        let mut best_occupancy = 0.0;

        for block_size in (self.device_props.warp_size..=max_threads_per_block)
            .step_by(self.device_props.warp_size as usize)
        {
            let occupancy = self.calculate_occupancy(block_size, registers_per_thread);
            if occupancy > best_occupancy {
                best_occupancy = occupancy;
                best_block_size = block_size;
            }
        }

        // Calculate grid size
        let grid_size = ((problem_size as u32 + best_block_size - 1) / best_block_size).max(1);

        // Limit grid size to avoid excessive blocks
        let max_grid_size = self.device_props.sm_count * self.device_props.max_blocks_per_sm;
        let grid_size = grid_size.min(max_grid_size);

        KernelConfig {
            grid_size: (grid_size, 1, 1),
            block_size: (best_block_size, 1, 1),
            shared_mem: 0, // Will be set per kernel
            min_blocks_per_sm: (self.device_props.max_threads_per_sm / best_block_size).max(1),
        }
    }

    fn calculate_max_threads_per_block(&self, registers_per_thread: u32) -> u32 {
        // Limit based on register usage
        let max_threads_by_registers = if registers_per_thread > 0 {
            // Assume 65536 registers per SM (typical for modern GPUs)
            let registers_per_sm = 65536;
            let max_threads_by_regs = registers_per_sm / registers_per_thread;
            max_threads_by_regs.min(self.device_props.max_threads_per_block)
        } else {
            self.device_props.max_threads_per_block
        };

        max_threads_by_registers
    }

    fn calculate_occupancy(&self, block_size: u32, registers_per_thread: u32) -> f32 {
        // Calculate theoretical occupancy
        let blocks_per_sm_by_threads = self.device_props.max_threads_per_sm / block_size;
        let blocks_per_sm_by_blocks = self.device_props.max_blocks_per_sm;

        let blocks_per_sm_by_registers = if registers_per_thread > 0 {
            let registers_per_sm = 65536; // Typical value
            let registers_per_block = block_size * registers_per_thread;
            registers_per_sm / registers_per_block
        } else {
            u32::MAX
        };

        let blocks_per_sm = blocks_per_sm_by_threads
            .min(blocks_per_sm_by_blocks)
            .min(blocks_per_sm_by_registers);

        let active_threads = blocks_per_sm * block_size;
        active_threads as f32 / self.device_props.max_threads_per_sm as f32
    }

    /// Get device properties for manual optimization
    pub fn device_properties(&self) -> &DeviceProperties {
        &self.device_props
    }

    /// Clear the configuration cache
    pub fn clear_cache(&mut self) {
        self.config_cache.clear();
    }
}

/// Specialized configurations for common RISC Zero kernels
impl KernelOptimizer {
    /// Configuration for NTT kernels
    pub fn ntt_config(&mut self, size: usize) -> KernelConfig {
        let mut config = self.get_config("ntt", size, 32); // NTT typically uses ~32 registers

        // NTT benefits from larger shared memory
        config.shared_mem = (config.block_size.0 as usize * std::mem::size_of::<u32>() * 4)
            .min(self.device_props.shared_mem_per_block);

        config
    }

    /// Configuration for hash kernels
    pub fn hash_config(&mut self, size: usize) -> KernelConfig {
        let mut config = self.get_config("hash", size, 24); // Hash typically uses ~24 registers

        // Hash kernels benefit from smaller blocks for better occupancy
        config.block_size.0 = config.block_size.0.min(256);
        config.grid_size.0 = ((size as u32 + config.block_size.0 - 1) / config.block_size.0).max(1);

        config
    }

    /// Configuration for polynomial evaluation kernels
    pub fn poly_eval_config(&mut self, size: usize) -> KernelConfig {
        let mut config = self.get_config("poly_eval", size, 40); // Poly eval uses more registers

        // Polynomial evaluation benefits from larger blocks
        config.block_size.0 = config.block_size.0.max(128);
        config.grid_size.0 = ((size as u32 + config.block_size.0 - 1) / config.block_size.0).max(1);

        config
    }

    /// Configuration for memory-bound kernels
    pub fn memory_bound_config(&mut self, size: usize) -> KernelConfig {
        let mut config = self.get_config("memory_bound", size, 16); // Memory-bound uses fewer registers

        // Memory-bound kernels benefit from maximum occupancy
        config.block_size.0 = self.device_props.max_threads_per_block.min(512);
        config.grid_size.0 = ((size as u32 + config.block_size.0 - 1) / config.block_size.0).max(1);

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_config_calculation() {
        // Mock test since we can't guarantee CUDA device in CI
        let optimizer = KernelOptimizer {
            device_props: DeviceProperties {
                max_threads_per_block: 1024,
                max_threads_per_sm: 2048,
                max_blocks_per_sm: 32,
                shared_mem_per_block: 49152,
                shared_mem_per_sm: 98304,
                warp_size: 32,
                sm_count: 80,
            },
            config_cache: HashMap::new(),
        };

        let config = optimizer.compute_optimal_config(1000000, 32);

        // Block size should be a multiple of warp size
        assert_eq!(config.block_size.0 % 32, 0);

        // Grid size should be reasonable
        assert!(config.grid_size.0 > 0);
        assert!(config.grid_size.0 <= 80 * 32); // sm_count * max_blocks_per_sm
    }

    #[test]
    fn test_occupancy_calculation() {
        let optimizer = KernelOptimizer {
            device_props: DeviceProperties {
                max_threads_per_block: 1024,
                max_threads_per_sm: 2048,
                max_blocks_per_sm: 32,
                shared_mem_per_block: 49152,
                shared_mem_per_sm: 98304,
                warp_size: 32,
                sm_count: 80,
            },
            config_cache: HashMap::new(),
        };

        let occupancy = optimizer.calculate_occupancy(256, 32);
        assert!(occupancy > 0.0);
        assert!(occupancy <= 1.0);
    }
}
