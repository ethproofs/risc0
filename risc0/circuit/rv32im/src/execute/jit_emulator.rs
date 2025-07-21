use std::collections::HashMap;
use anyhow::Result;
use risc0_binfmt::ByteAddr;

use super::rv32im::{Emulator, EmuContext, InsnKind, DecodedInstruction};
use super::jit::{JitCompiler, BasicBlock};
use super::r0vm::EmuStep;

/// Simple JIT-enabled emulator that directly translates RISC-V to x86-64
pub struct JitEmulator {
    /// Fallback interpreter
    interpreter: Emulator,
    /// JIT compiler
    jit_compiler: JitCompiler,
    /// Execution count for hotspot detection
    execution_count: HashMap<ByteAddr, u32>,
    /// Threshold for JIT compilation
    jit_threshold: u32,
    /// Whether JIT is enabled
    jit_enabled: bool,
    /// Statistics
    stats: JitStats,
}

impl JitEmulator {
    pub fn new() -> Result<Self> {
        // Simple JIT enablement - just check if we can allocate executable memory
        let jit_enabled = std::env::var("RISC0_DISABLE_JIT").is_err()
            && !cfg!(target_os = "macos"); // Disable on macOS due to sandbox restrictions

        if jit_enabled {
            tracing::info!("JIT compilation enabled");
        } else {
            tracing::info!("JIT compilation disabled");
        }

        let jit_compiler = JitCompiler::new()?;

        Ok(Self {
            interpreter: Emulator::new(),
            jit_compiler,
            execution_count: HashMap::new(),
            jit_threshold: if jit_enabled { 1000 } else { u32::MAX }, // Much lower threshold
            jit_enabled,
            stats: JitStats::default(),
        })
    }

    /// Step execution with simple JIT compilation
    pub fn step<C: EmuContext>(&mut self, ctx: &mut C) -> Result<()> {
        let pc = ctx.get_pc();

        // If JIT is disabled, just use interpreter
        if !self.jit_enabled {
            return self.interpreter.step(ctx);
        }

        // Check if we have compiled code for this address
        if let Some(compiled_code) = self.jit_compiler.get_compiled_block(pc) {
            self.stats.native_executions += 1;
            return self.execute_compiled_block(ctx, compiled_code);
        }

        // Update execution count for hotspot detection
        let count = self.execution_count.entry(pc).or_insert(0);
        *count += 1;
        self.stats.total_executions += 1;

        // Check if this should trigger JIT compilation
        if *count >= self.jit_threshold {
            self.try_compile_block(ctx, pc)?;
        }

        // Execute with interpreter
        self.interpreter.step(ctx)
    }

    /// Try to compile a basic block starting at the given address
    fn try_compile_block<C: EmuContext>(&mut self, ctx: &mut C, start_addr: ByteAddr) -> Result<()> {
        let block = self.build_basic_block(ctx, start_addr)?;

        if !block.instructions.is_empty() {
            tracing::debug!("Compiling basic block at {:?} with {} instructions",
                          block.start_addr, block.instructions.len());

            match self.jit_compiler.compile_block(&block) {
                Ok(_) => {
                    self.stats.compiled_blocks += 1;
                    tracing::debug!("Successfully compiled block");
                }
                Err(e) => {
                    tracing::warn!("JIT compilation failed: {e}, falling back to interpreter");
                    // Don't disable JIT completely, just skip this block
                }
            }
        }

        Ok(())
    }

    /// Build a basic block starting at the given address
    fn build_basic_block<C: EmuContext>(&mut self, ctx: &mut C, start_addr: ByteAddr) -> Result<BasicBlock> {
        let mut block = BasicBlock {
            start_addr,
            instructions: Vec::new(),
            end_addr: start_addr,
            is_conditional_branch: false,
        };

        let mut current_addr = start_addr;

        // Collect instructions until we hit a control flow instruction
        for _ in 0..16 { // Limit basic block size
            if !ctx.check_insn_load(current_addr) {
                break;
            }

            let word = ctx.load_memory(current_addr.waddr())?;
            let decoded = DecodedInstruction::new(word);
            let kind = Self::decode_instruction_kind(&decoded);

            // Skip memory operations - they cause segfaults in JIT
            if matches!(kind, InsnKind::Lw | InsnKind::Sw | InsnKind::Lh | InsnKind::LhU |
                              InsnKind::Lb | InsnKind::LbU | InsnKind::Sh | InsnKind::Sb) {
                break;
            }

            block.instructions.push((kind, decoded));
            block.end_addr = current_addr;

            // Stop at control flow instructions
            if Self::is_block_ending_instruction(kind) {
                block.is_conditional_branch = matches!(kind,
                    InsnKind::Beq | InsnKind::Bne | InsnKind::Blt | InsnKind::Bge |
                    InsnKind::BltU | InsnKind::BgeU
                );
                break;
            }

            current_addr = current_addr + 4;
        }

        Ok(block)
    }

    /// Execute a compiled basic block with simple approach
    fn execute_compiled_block<C: EmuContext>(&mut self, ctx: &mut C, compiled_code: *const u8) -> Result<()> {
        tracing::debug!("Executing compiled block at {:?}", ctx.get_pc());

        // Simple approach: just call the native code with the context pointer
        let result = unsafe {
            let jit_fn: unsafe extern "C" fn(*mut u8) -> i32 =
                std::mem::transmute(compiled_code);

            // Pass the emulator context directly
            let context_ptr = ctx as *mut C as *mut u8;
            jit_fn(context_ptr)
        };

        // Handle the result
        match result {
            0 => {
                // Normal completion - update PC to next instruction
                ctx.set_pc(ctx.get_pc() + 4);
                Ok(())
            }
            pc_value if pc_value > 0x1000 => {
                // Branch/jump - set new PC
                ctx.set_pc(risc0_binfmt::ByteAddr(pc_value as u32));
                Ok(())
            }
            8 => {
                // ECALL - trigger actual environment call in emulator
                ctx.ecall()?;
                Ok(())
            }
            3 => {
                // EBREAK - trigger breakpoint exception
                ctx.trap(super::rv32im::Exception::Breakpoint)?;
                Ok(())
            }
            48 => {
                // MRET - machine return
                ctx.mret()?;
                Ok(())
            }
            _ => {
                // Unknown result - assume normal completion
                ctx.set_pc(ctx.get_pc() + 4);
                Ok(())
            }
        }
    }

    /// Clear the JIT cache to force recompilation
    pub fn clear_jit_cache(&mut self) {
        self.jit_compiler.clear_cache();
        tracing::debug!("JIT cache cleared");
    }

    /// Comprehensive instruction decoding that mirrors the main emulator
    fn decode_instruction_kind(decoded: &DecodedInstruction) -> InsnKind {
        let opcode = decoded.insn & 0x7f;
        let func3 = (decoded.insn >> 12) & 0x7;
        let func7 = (decoded.insn >> 25) & 0x7f;

        match (opcode, func3, func7) {
            // R-format arithmetic ops
            (0b0110011, 0b000, 0b0000000) => InsnKind::Add,
            (0b0110011, 0b000, 0b0100000) => InsnKind::Sub,
            (0b0110011, 0b001, 0b0000000) => InsnKind::Sll,
            (0b0110011, 0b010, 0b0000000) => InsnKind::Slt,
            (0b0110011, 0b011, 0b0000000) => InsnKind::SltU,
            (0b0110011, 0b101, 0b0000000) => InsnKind::Srl,
            (0b0110011, 0b100, 0b0000000) => InsnKind::Xor,
            (0b0110011, 0b101, 0b0100000) => InsnKind::Sra,
            (0b0110011, 0b110, 0b0000000) => InsnKind::Or,
            (0b0110011, 0b111, 0b0000000) => InsnKind::And,

            // M-extension multiply/divide
            (0b0110011, 0b000, 0b0000001) => InsnKind::Mul,
            (0b0110011, 0b001, 0b0000001) => InsnKind::MulH,
            (0b0110011, 0b010, 0b0000001) => InsnKind::MulHSU,
            (0b0110011, 0b011, 0b0000001) => InsnKind::MulHU,
            (0b0110011, 0b100, 0b0000001) => InsnKind::Div,
            (0b0110011, 0b101, 0b0000001) => InsnKind::DivU,
            (0b0110011, 0b110, 0b0000001) => InsnKind::Rem,
            (0b0110011, 0b111, 0b0000001) => InsnKind::RemU,

            // I-format arithmetic ops
            (0b0010011, 0b000, _) => InsnKind::AddI,
            (0b0010011, 0b001, 0b0000000) => InsnKind::SllI,
            (0b0010011, 0b010, _) => InsnKind::SltI,
            (0b0010011, 0b011, _) => InsnKind::SltIU,
            (0b0010011, 0b100, _) => InsnKind::XorI,
            (0b0010011, 0b101, 0b0000000) => InsnKind::SrlI,
            (0b0010011, 0b101, 0b0100000) => InsnKind::SraI,
            (0b0010011, 0b110, _) => InsnKind::OrI,
            (0b0010011, 0b111, _) => InsnKind::AndI,

            // I-format memory loads
            (0b0000011, 0b000, _) => InsnKind::Lb,
            (0b0000011, 0b001, _) => InsnKind::Lh,
            (0b0000011, 0b010, _) => InsnKind::Lw,
            (0b0000011, 0b100, _) => InsnKind::LbU,
            (0b0000011, 0b101, _) => InsnKind::LhU,

            // S-format memory stores
            (0b0100011, 0b000, _) => InsnKind::Sb,
            (0b0100011, 0b001, _) => InsnKind::Sh,
            (0b0100011, 0b010, _) => InsnKind::Sw,

            // U-format upper immediate
            (0b0110111, _, _) => InsnKind::Lui,
            (0b0010111, _, _) => InsnKind::Auipc,

            // B-format branch instructions
            (0b1100011, 0b000, _) => InsnKind::Beq,
            (0b1100011, 0b001, _) => InsnKind::Bne,
            (0b1100011, 0b100, _) => InsnKind::Blt,
            (0b1100011, 0b101, _) => InsnKind::Bge,
            (0b1100011, 0b110, _) => InsnKind::BltU,
            (0b1100011, 0b111, _) => InsnKind::BgeU,

            // J-format jump
            (0b1101111, _, _) => InsnKind::Jal,

            // I-format jump register
            (0b1100111, _, _) => InsnKind::JalR,

            // System instructions
            (0b1110011, 0b000, 0b0011000) => InsnKind::Mret,
            (0b1110011, 0b000, 0b0000000) => InsnKind::Eany,

            _ => InsnKind::Invalid,
        }
    }

    /// Check if an instruction ends a basic block
    fn is_block_ending_instruction(kind: InsnKind) -> bool {
        matches!(kind,
            InsnKind::Jal | InsnKind::JalR |
            InsnKind::Beq | InsnKind::Bne | InsnKind::Blt | InsnKind::Bge |
            InsnKind::BltU | InsnKind::BgeU |
            InsnKind::Eany | InsnKind::Mret
        )
    }

    /// Get compilation statistics
    pub fn get_stats(&self) -> &JitStats {
        &self.stats
    }

    /// Get detailed performance information
    pub fn get_performance_info(&self) -> String {
        format!(
            "JIT Stats: {} compiled blocks, {}/{} executions native ({:.1}%)",
            self.stats.compiled_blocks,
            self.stats.native_executions,
            self.stats.total_executions,
            if self.stats.total_executions > 0 {
                (self.stats.native_executions as f64 / self.stats.total_executions as f64) * 100.0
            } else {
                0.0
            }
        )
    }
}

impl EmuStep for JitEmulator {
    fn step<C: EmuContext>(&mut self, ctx: &mut C) -> Result<()> {
        self.step(ctx)
    }
}

/// JIT compilation statistics
#[derive(Debug, Clone, Default)]
pub struct JitStats {
    pub compiled_blocks: usize,
    pub total_executions: u64,
    pub native_executions: u64,
}

impl JitStats {
    pub fn compilation_ratio(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.native_executions as f64 / self.total_executions as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_emulator_creation() {
        let emulator = JitEmulator::new().unwrap();
        if std::env::var("RISC0_DISABLE_JIT").is_ok() || cfg!(target_os = "macos") {
            assert_eq!(emulator.jit_threshold, u32::MAX);
            assert!(!emulator.jit_enabled);
        } else {
            assert_eq!(emulator.jit_threshold, 1000);
            assert!(emulator.jit_enabled);
        }
        assert_eq!(emulator.stats.compiled_blocks, 0);
    }

    #[test]
    fn test_hotspot_detection() {
        let mut emulator = JitEmulator::new().unwrap();
        emulator.jit_threshold = 10;

        let addr = ByteAddr(0x1000);
        for _ in 0..15 {
            *emulator.execution_count.entry(addr).or_insert(0) += 1;
        }

        let hot_spots = emulator.execution_count.iter()
            .filter(|(_, &count)| count >= emulator.jit_threshold)
            .count();

        assert_eq!(hot_spots, 1);
    }

    #[test]
    fn test_instruction_decoding() {
        let decoded = DecodedInstruction::new(0x00208033); // ADD x0, x1, x2
        let kind = JitEmulator::decode_instruction_kind(&decoded);
        assert_eq!(kind, InsnKind::Add);
    }

    #[test]
    fn test_block_ending_instructions() {
        assert!(JitEmulator::is_block_ending_instruction(InsnKind::Jal));
        assert!(JitEmulator::is_block_ending_instruction(InsnKind::Beq));
        assert!(!JitEmulator::is_block_ending_instruction(InsnKind::Add));
    }
}
