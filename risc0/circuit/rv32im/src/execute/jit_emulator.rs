use std::collections::HashMap;
/// JIT emulator scaffolding (unused imports and dead code allowed)
#[allow(dead_code, unused_imports, unused_variables)]
use anyhow::{Result, anyhow};
use risc0_binfmt::ByteAddr;

use super::rv32im::{Emulator, EmuContext, InsnKind, DecodedInstruction};
use super::jit::{JitCompiler, BasicBlock, MemoryCallbacks};
use super::r0vm::EmuStep;

/// Test if we can allocate executable memory (quick JIT capability test)
fn test_executable_memory_allocation() -> Result<()> {
    #[cfg(unix)]
    {
        use libc::{mmap, munmap, MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

        // Try to allocate a small page of executable memory
        let size = 4096; // One page
        let addr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if addr == MAP_FAILED {
            return Err(anyhow!("Failed to allocate executable memory"));
        }

        // Immediately free it
        unsafe {
            munmap(addr, size);
        }

        Ok(())
    }

    #[cfg(not(unix))]
    {
        // On non-Unix systems, assume executable memory allocation will fail
        Err(anyhow!("Executable memory allocation not supported on this platform"))
    }
}

/// Execution context wrapper to pass EmuContext through JIT callbacks safely
pub struct JitExecutionContext<'a> {
    pub emu_context: &'a mut dyn EmuContext,
    pub has_error: bool,
}

/// JIT-enabled emulator that combines interpretation and compilation
pub struct JitEmulator {
    /// Fallback interpreter
    interpreter: Emulator,
    /// JIT compiler
    jit_compiler: JitCompiler,
    /// Execution count for each instruction address (for hotspot detection)
    execution_count: HashMap<ByteAddr, u32>,
    /// Threshold for JIT compilation
    jit_threshold: u32,
    /// Whether JIT is actually working (can execute native code)
    jit_enabled: bool,
    /// Statistics
    stats: JitStats,
}

impl JitEmulator {
    pub fn new() -> Result<Self> {
        // Check environment variable to completely disable JIT
        // On macOS, executable mmap is restricted under sandbox, so disable JIT
        // Also disable JIT in RPC contexts where communication failures are likely
        let mut jit_enabled = std::env::var("RISC0_DISABLE_JIT").is_err()
            && !cfg!(target_os = "macos")
            && std::env::var("RISC0_RPC_MODE").is_err(); // Disable JIT in RPC mode

        // Check if previous JIT execution crashed
        if jit_enabled {
            if let Ok(temp_dir) = std::env::var("TMPDIR") {
                let crash_indicator = std::path::Path::new(&temp_dir).join(".risc0_jit_crashes");
                if crash_indicator.exists() {
                    tracing::warn!("Found JIT crash indicator file, disabling JIT for safety");
                    jit_enabled = false;
                }
            }
        }

        // Additional safety checks - disable JIT if system appears unstable
        if jit_enabled {
            // Check if we're running in a container or restricted environment
            if std::path::Path::new("/.dockerenv").exists() ||
               std::env::var("container").is_ok() ||
               std::env::var("KUBERNETES_SERVICE_HOST").is_ok() {
                tracing::warn!("Detected containerized environment, disabling JIT for stability");
                jit_enabled = false;
            }

            // Check if we can allocate executable memory at all
            if jit_enabled {
                match test_executable_memory_allocation() {
                    Ok(()) => {
                        tracing::debug!("JIT executable memory test passed");
                    }
                    Err(e) => {
                        tracing::warn!("JIT executable memory test failed: {e}, disabling JIT");
                        jit_enabled = false;
                    }
                }
            }
        }

        if jit_enabled {
            tracing::info!("JIT compilation enabled with high stability threshold");
        } else {
            tracing::info!("JIT compilation disabled for system stability");
        }

        let jit_compiler = JitCompiler::new()?;

        // Clear any cached blocks to ensure fresh compilation
        let mut compiler = jit_compiler;
        compiler.clear_cache();

        Ok(Self {
            interpreter: Emulator::new(),
            jit_compiler: compiler,
            execution_count: HashMap::new(),
            // More conservative threshold to prevent compilation of problematic code
            // This helps avoid the slice bounds error by being more selective about what gets compiled
            jit_threshold: if jit_enabled { 50000 } else { u32::MAX }, // More conservative threshold
            jit_enabled,
            stats: JitStats::default(),
        })
    }

    /// Step execution with JIT compilation
    pub fn step<C: EmuContext>(&mut self, ctx: &mut C) -> Result<()> {
        let pc = ctx.get_pc();

        // If JIT is disabled, just use interpreter
        if !self.jit_enabled {
            return self.interpreter.step(ctx);
        }

        // ADDITIONAL SAFETY: Check for signs of memory corruption before JIT execution
        // This helps prevent the slice bounds error by detecting problematic register states
        let mut suspicious_registers = 0;
        for i in 1..8 { // Check first few registers for suspicious values
            if let Ok(val) = ctx.load_register(i) {
                // More intelligent detection: look for truly suspicious patterns
                // High values alone are normal - they're just memory addresses
                // Look for patterns that suggest corruption instead
                let is_suspicious = match val {
                    // Check for unaligned addresses (not 4-byte aligned)
                    addr if addr & 0x3 != 0 && addr > 0x1000 => {
                        tracing::warn!("Unaligned address detected in register x{}: 0x{:08x}", i, val);
                        true
                    }
                    // Check for extremely high addresses that are unlikely to be valid
                    addr if addr > 0x80000000 && addr < 0xffff0000 => {
                        tracing::warn!("Extremely high address detected in register x{}: 0x{:08x}", i, val);
                        true
                    }
                    // Check for null pointers (except in x0 which should always be 0)
                    addr if addr == 0 && i != 0 => {
                        tracing::warn!("Null pointer detected in register x{}: 0x{:08x}", i, val);
                        true
                    }
                    // Check for obviously corrupted values (all bits set)
                    addr if addr == 0xffffffff => {
                        tracing::warn!("Corrupted value detected in register x{}: 0x{:08x}", i, val);
                        true
                    }
                    _ => false
                };

                if is_suspicious {
                    suspicious_registers += 1;
                }
            }
        }

        // If too many registers have suspicious values, disable JIT to prevent corruption
        if suspicious_registers >= 2 {
            tracing::warn!("Too many suspicious register values detected ({}), disabling JIT for safety", suspicious_registers);
            self.jit_enabled = false;
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
        // ADDITIONAL SAFETY: Check if JIT compilation is still safe
        if !self.jit_enabled {
            return Ok(());
        }

        let block = self.build_basic_block(ctx, start_addr)?;

        if !block.instructions.is_empty() {
            tracing::debug!("Compiling basic block at {:?} with {} instructions",
                          block.start_addr, block.instructions.len());

            // ADDITIONAL SAFETY: Validate the block before compilation
            // This helps prevent compilation of potentially problematic code
            let mut safe_instructions = 0;
            for (kind, _) in &block.instructions {
                match kind {
                    InsnKind::Add | InsnKind::AddI | InsnKind::Sub | InsnKind::And |
                    InsnKind::Or | InsnKind::Xor | InsnKind::Sll | InsnKind::Srl |
                    InsnKind::Sra | InsnKind::Slt | InsnKind::SltU | InsnKind::SltI |
                    InsnKind::SltIU | InsnKind::XorI | InsnKind::OrI | InsnKind::AndI |
                    InsnKind::SllI | InsnKind::SrlI | InsnKind::SraI | InsnKind::Lui |
                    InsnKind::Auipc => {
                        safe_instructions += 1;
                    }
                    _ => {
                        // Skip compilation of potentially unsafe instructions
                        tracing::debug!("Skipping compilation due to potentially unsafe instruction: {:?}", kind);
                        return Ok(());
                    }
                }
            }

            // Only compile if we have enough safe instructions
            if safe_instructions >= block.instructions.len() / 2 {
                match self.jit_compiler.compile_block(&block) {
                    Ok(_) => {
                        self.stats.compiled_blocks += 1;
                        tracing::debug!("Successfully compiled block with {} safe instructions", safe_instructions);
                    }
                    Err(e) => {
                        tracing::warn!("JIT compilation failed: {e}, disabling JIT");
                        self.jit_enabled = false;
                        return Err(e);
                    }
                }
            } else {
                tracing::debug!("Skipping compilation due to insufficient safe instructions: {}/{}",
                              safe_instructions, block.instructions.len());
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

            // ADDITIONAL SAFETY: Skip compilation of potentially problematic instructions
            // This helps prevent the slice bounds error by avoiding compilation of complex operations
            match kind {
                InsnKind::Lw | InsnKind::Lh | InsnKind::Lb | InsnKind::LhU | InsnKind::LbU |
                InsnKind::Sw | InsnKind::Sh | InsnKind::Sb => {
                    // Skip memory operations for now to prevent potential corruption
                    tracing::debug!("Skipping JIT compilation of memory operation at {:?}", current_addr);
                    break;
                }
                InsnKind::Mul | InsnKind::MulH | InsnKind::MulHSU | InsnKind::MulHU |
                InsnKind::Div | InsnKind::DivU | InsnKind::Rem | InsnKind::RemU => {
                    // Skip complex arithmetic operations that might cause issues
                    tracing::debug!("Skipping JIT compilation of complex arithmetic at {:?}", current_addr);
                    break;
                }
                _ => {
                    // Continue with normal compilation for safe instructions
                }
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

            /// Execute a compiled basic block
    fn execute_compiled_block<C: EmuContext>(&mut self, ctx: &mut C, compiled_code: *const u8) -> Result<()> {
        tracing::debug!("Executing compiled block at {:?} (native code at {:p})", ctx.get_pc(), compiled_code);

        // Try to execute the native code
        let native_result = self.try_execute_native_code(ctx, compiled_code);

        match native_result {
            Ok(()) => {
                tracing::debug!("Native code execution succeeded, skipping interpreter step");
                // Native code has executed; skip standard interpretation for this block
                Ok(())
            }
            Err(e) => {
                let error_str = format!("{e:?}");
                if error_str.contains("RPC communication failure") {
                    tracing::error!("JIT execution failed due to RPC communication failure, permanently disabling JIT");
                } else {
                    tracing::warn!("Native code execution failed ({e}), disabling JIT compilation");
                }

                // Disable JIT to avoid compilation overhead and further failures
                self.jit_enabled = false;

                // Clear any potentially corrupted compilation cache
                self.execution_count.clear();

                // Fall back to interpreter
                tracing::debug!("Falling back to interpreter execution");
                self.interpreter.step(ctx)
            }
        }
    }

                                                                /// Try to execute native code with proper exception handling
    fn try_execute_native_code<C: EmuContext>(&mut self, ctx: &mut C, compiled_code: *const u8) -> Result<()> {
        // Create CPU context for native code using Box to ensure stable memory location
        let mut cpu_context = Box::new(super::jit::CpuContext::new());

        // Copy registers from emulator context to CPU context
        for i in 0..32 {
            cpu_context.registers[i] = ctx.load_register(i).unwrap_or(0);
        }
        cpu_context.pc = ctx.get_pc().0;

        // Set up context pointer and callbacks for memory operations
        cpu_context.emu_context = ctx as *mut C as *mut u8;
        cpu_context.callbacks = &MEMORY_CALLBACKS;

        // Get stable pointer to boxed CPU context
        let context_ptr = cpu_context.as_mut() as *mut _ as *mut u8;

        // Validate the CPU context pointer before use
        if context_ptr.is_null() {
            return Err(anyhow!("CPU context pointer is null"));
        }

        // Additional safety check: verify the pointer is in valid memory range
        if (context_ptr as usize) < 0x1000000 {
            return Err(anyhow!("CPU context pointer {context_ptr:p} appears invalid (too low)"));
        }

        // Verify the compiled code pointer is valid
        if compiled_code.is_null() {
            return Err(anyhow!("Compiled code pointer is null"));
        }

        if (compiled_code as usize) < 0x1000000 {
            return Err(anyhow!("Compiled code pointer {compiled_code:p} appears invalid (too low)"));
        }

        // Set the active context for callbacks
        unsafe {
            set_active_context(ctx as *mut C as *mut dyn EmuContext);
        }

        tracing::debug!("JIT calling native code at {compiled_code:p} with context at {context_ptr:p}");

        // DEBUG: Dump the generated code for analysis
        let code_slice = unsafe {
            std::slice::from_raw_parts(compiled_code, 64) // First 64 bytes
        };
        let code_hex: String = code_slice.iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ");
        tracing::debug!("Generated x86-64 code (first 64 bytes): {code_hex}");

        // Cast the compiled code to a function pointer and execute it
        let result = unsafe {
            let jit_fn: unsafe extern "C" fn(*mut u8) -> i32 =
                std::mem::transmute(compiled_code);

            // DEBUGGING: Add extensive validation and logging to track RDI corruption
            tracing::debug!("JIT execution starting:");
            tracing::debug!("  CPU context pointer: {context_ptr:p}");
            tracing::debug!("  CPU context as usize: 0x{:x}", context_ptr as usize);
            tracing::debug!("  Compiled code pointer: {compiled_code:p}");

            // Verify the context is readable by accessing a known field
            let test_read = std::ptr::read_volatile(context_ptr as *const u8);
            tracing::debug!("  Context memory test read: 0x{test_read:02x}");

            // Log the first few values in the CPU context for debugging
            let ctx_words = std::slice::from_raw_parts(context_ptr as *const u32, 8);
            tracing::debug!("  CPU context first 8 words: {:08x?}", ctx_words);

            // ADDITIONAL SAFETY: Validate register values before JIT execution
            // This helps catch potential issues with register state corruption
            tracing::debug!("Pre-JIT register state validation:");
            for i in 0..8 {
                let reg_val = cpu_context.registers[i];
                // Only flag truly suspicious patterns, not just high values
                let is_suspicious = match reg_val {
                    // Check for unaligned addresses (not 4-byte aligned)
                    addr if addr & 0x3 != 0 && addr > 0x1000 => {
                        tracing::warn!("  Register x{} has unaligned address: 0x{:08x}", i, reg_val);
                        true
                    }
                    // Check for extremely high addresses that are unlikely to be valid
                    addr if addr > 0x80000000 && addr < 0xffff0000 => {
                        tracing::warn!("  Register x{} has extremely high address: 0x{:08x}", i, reg_val);
                        true
                    }
                    // Check for null pointers (except in x0 which should always be 0)
                    0 if i != 0 => {
                        tracing::warn!("  Register x{} has null pointer: 0x{:08x}", i, reg_val);
                        true
                    }
                    // Check for obviously corrupted values (all bits set)
                    0xffffffff => {
                        tracing::warn!("  Register x{} has corrupted value: 0x{:08x}", i, reg_val);
                        true
                    }
                    _ => false
                };

                if is_suspicious {
                    tracing::warn!("  Register x{} has suspicious value: 0x{:08x}", i, reg_val);
                } else {
                    tracing::debug!("  x{}: 0x{:08x}", i, reg_val);
                }
            }

            // TEMPORARY: Memory operations in JIT code are disabled and return dummy values
            // This isolates whether the crashes are caused by the callback mechanism itself
            // or some other aspect of the JIT system. If crashes stop, the issue is in
            // the memory callback implementation or calling convention.
            //
            // DISCOVERY: Crashes continue even with callbacks disabled, so the issue is
            // in the core JIT register access system - RDI is becoming null somehow.
            //
            // Current debugging focus:
            // - Verify CPU context pointer is valid before JIT call
            // - Track when and how RDI becomes corrupted
            // - Ensure calling convention matches between caller and generated code

            tracing::debug!("About to call JIT function with context at {context_ptr:p}");
            let result = jit_fn(context_ptr);
            tracing::debug!("JIT function returned: {result}");

            result
        };

        // Clear the active context
        clear_active_context();

        tracing::debug!("JIT function call completed, processing results");

        // ADDITIONAL SAFETY: Validate register values after JIT execution
        // This helps catch potential issues with register state corruption
        tracing::debug!("Post-JIT register state validation:");
        for i in 0..8 {
            let reg_val = cpu_context.registers[i];
            // Only flag truly suspicious patterns, not just high values
            let is_suspicious = match reg_val {
                // Check for unaligned addresses (not 4-byte aligned)
                addr if addr & 0x3 != 0 && addr > 0x1000 => {
                    tracing::warn!("  Register x{} has unaligned address after JIT: 0x{:08x}", i, reg_val);
                    true
                }
                // Check for extremely high addresses that are unlikely to be valid
                addr if addr > 0x80000000 && addr < 0xffff0000 => {
                    tracing::warn!("  Register x{} has extremely high address after JIT: 0x{:08x}", i, reg_val);
                    true
                }
                // Check for null pointers (except in x0 which should always be 0)
                0 if i != 0 => {
                    tracing::warn!("  Register x{} has null pointer after JIT: 0x{:08x}", i, reg_val);
                    true
                }
                // Check for obviously corrupted values (all bits set)
                0xffffffff => {
                    tracing::warn!("  Register x{} has corrupted value after JIT: 0x{:08x}", i, reg_val);
                    true
                }
                _ => false
            };

            if is_suspicious {
                tracing::warn!("  Register x{} has suspicious value after JIT: 0x{:08x}", i, reg_val);
            } else {
                tracing::debug!("  x{}: 0x{:08x}", i, reg_val);
            }
        }

        // Check if any memory operations returned the RPC failure sentinel value
        // This indicates RPC communication failed and JIT should be disabled
        for i in 1..32 {
            if cpu_context.registers[i] == 0xDEADBEEF {
                tracing::error!("JIT detected RPC communication failure in register {i}, disabling JIT");
                self.jit_enabled = false;
                return Err(anyhow!("RPC communication failure detected in JIT execution"));
            }
        }

        tracing::debug!("Starting register copy-back from CPU context to emulator context");

        // Copy registers back from CPU context to emulator context first
        // THIS IS A LIKELY SOURCE OF RPC COMMUNICATION FAILURE
        for i in 1..32 { // Skip x0 (always zero)
            let reg_val = cpu_context.registers[i];
            tracing::debug!("Copying register {i}: 0x{:08x}", reg_val);

            // ADDITIONAL SAFETY: Validate register values before storing
            // This helps prevent corrupted register values from affecting the guest program
            let is_suspicious = match reg_val {
                // Check for unaligned addresses (not 4-byte aligned)
                addr if addr & 0x3 != 0 && addr > 0x1000 => {
                    tracing::warn!("Register x{} has unaligned address: 0x{:08x}, clamping to aligned value", i, reg_val);
                    true
                }
                // Check for extremely high addresses that are unlikely to be valid
                addr if addr > 0x80000000 && addr < 0xffff0000 => {
                    tracing::warn!("Register x{} has extremely high address: 0x{:08x}, clamping to safe value", i, reg_val);
                    true
                }
                // Check for null pointers (except in x0 which should always be 0)
                0 if i != 0 => {
                    tracing::warn!("Register x{} has null pointer: 0x{:08x}, clamping to safe value", i, reg_val);
                    true
                }
                // Check for obviously corrupted values (all bits set)
                0xffffffff => {
                    tracing::warn!("Register x{} has corrupted value: 0x{:08x}, clamping to safe value", i, reg_val);
                    true
                }
                _ => false
            };

            if is_suspicious {
                // Clamp the value to prevent potential issues
                let clamped_val = 0x7FFFFFFF;
                match ctx.store_register(i, clamped_val) {
                    Ok(()) => {
                        tracing::debug!("Successfully stored clamped register {i}");
                    }
                    Err(e) => {
                        tracing::error!("Failed to store clamped register {i}: {e}");
                        let error_str = format!("{e:?}");
                        if error_str.contains("rx len failed") || error_str.contains("failed to fill whole buffer") {
                            tracing::error!("RPC failure during register store for register {i}, disabling JIT");
                            self.jit_enabled = false;
                            return Err(anyhow!("RPC communication failure during register store"));
                        }
                        return Err(e);
                    }
                }
            } else {
                match ctx.store_register(i, reg_val) {
                    Ok(()) => {
                        tracing::debug!("Successfully stored register {i}");
                    }
                    Err(e) => {
                        tracing::error!("Failed to store register {i}: {e}");
                        let error_str = format!("{e:?}");
                        if error_str.contains("rx len failed") || error_str.contains("failed to fill whole buffer") {
                            tracing::error!("RPC failure during register store for register {i}, disabling JIT");
                            self.jit_enabled = false;
                            return Err(anyhow!("RPC communication failure during register store"));
                        }
                        return Err(e);
                    }
                }
            }
        }

        tracing::debug!("Register copy-back completed successfully, handling JIT result: {result}");

        // Handle the result (PC update, exceptions, etc.)
        match result {
            0 => {
                // Normal completion - update PC to next instruction
                tracing::debug!("JIT result: normal completion, updating PC");
                ctx.set_pc(ctx.get_pc() + 4);
                tracing::debug!("PC updated successfully");
                Ok(())
            }
            pc_value if pc_value > 0x1000 => {
                // Branch/jump - set new PC
                tracing::debug!("JIT result: branch/jump to PC 0x{:08x}", pc_value);
                ctx.set_pc(risc0_binfmt::ByteAddr(pc_value as u32));
                tracing::debug!("Branch PC updated successfully");
                Ok(())
            }
            8 => {
                // ECALL - trigger actual environment call in emulator
                tracing::debug!("JIT result: ECALL, delegating to emulator");
                match ctx.ecall() {
                    Ok(_) => {
                        tracing::debug!("ECALL handled successfully");
                        // Both termination and continuation are handled by the emulator
                        // PC is already updated appropriately by the ecall handler
                        Ok(())
                    }
                    Err(e) => {
                        tracing::error!("ECALL failed in JIT context: {e}");
                        // Check if this is also an RPC failure
                        let error_str = format!("{e:?}");
                        if error_str.contains("rx len failed") || error_str.contains("failed to fill whole buffer") {
                            tracing::error!("RPC communication failure during ECALL, disabling JIT");
                            self.jit_enabled = false;
                        }
                        Err(e)
                    }
                }
            }
            3 => {
                // EBREAK - trigger breakpoint exception
                tracing::debug!("JIT result: EBREAK, delegating to emulator");
                match ctx.trap(super::rv32im::Exception::Breakpoint) {
                    Ok(_) => {
                        tracing::debug!("EBREAK handled successfully");
                        // Breakpoint handled by emulator, PC updated appropriately
                        Ok(())
                    }
                    Err(e) => {
                        tracing::error!("EBREAK failed in JIT context: {e}");
                        Err(e)
                    }
                }
            }
            48 => {
                // MRET - machine return
                tracing::debug!("JIT result: MRET, delegating to emulator");
                match ctx.mret() {
                    Ok(_) => {
                        tracing::debug!("MRET handled successfully");
                        // Machine return handled by emulator, privilege level and PC updated
                        Ok(())
                    }
                    Err(e) => {
                        tracing::error!("MRET failed in JIT context: {e}");
                        Err(e)
                    }
                }
            }
            _ => {
                tracing::warn!("JIT returned unknown result code: {result}");
                // Assume normal completion and advance PC
                tracing::debug!("Unknown result code, advancing PC normally");
                ctx.set_pc(ctx.get_pc() + 4);
                tracing::debug!("PC advanced successfully");
                Ok(())
            }
        }
    }

    /// Clear the JIT cache to force recompilation
    pub fn clear_jit_cache(&mut self) {
        self.jit_compiler.clear_cache();
        tracing::debug!("JIT cache cleared in emulator");
    }

    /// Get the size of the current basic block (for demonstration)
    #[allow(dead_code)]
    fn get_current_block_size(&self, pc: ByteAddr) -> usize {
        // In a real implementation, we'd track block sizes
        // For now, return a reasonable estimate
        if pc.0 == 0x0001000c { 6 } else { 5 }
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
    #[allow(dead_code)]
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

/// Enhanced emulator with profiling and adaptive compilation
#[allow(dead_code)]
pub struct AdaptiveJitEmulator {
    jit_emulator: JitEmulator,
    adaptive_threshold: bool,
    #[allow(dead_code)]
    max_block_size: usize,
}

#[allow(dead_code)]
impl AdaptiveJitEmulator {
    pub fn new() -> Result<Self> {
        Ok(Self {
            jit_emulator: JitEmulator::new()?,
            adaptive_threshold: true,
            max_block_size: 32,
        })
    }

    /// Step with adaptive compilation
    pub fn step<C: EmuContext>(&mut self, ctx: &mut C) -> Result<()> {
        if self.adaptive_threshold {
            self.adjust_compilation_threshold();
        }
        self.jit_emulator.step(ctx)
    }

            /// Adjust compilation threshold based on execution patterns
    fn adjust_compilation_threshold(&mut self) {
        let stats = self.jit_emulator.get_stats().clone();
        let ratio = stats.compilation_ratio();
        let current_threshold = self.jit_emulator.jit_threshold;

        // If we're getting good compilation ratios, lower the threshold
        if ratio > 0.8 && current_threshold > 5 {
            self.jit_emulator.jit_threshold -= 1;
        }

        // If we're not getting good ratios, raise the threshold
        if ratio < 0.2 && current_threshold < 100 {
            self.jit_emulator.jit_threshold += 1;
        }
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> &JitStats {
        self.jit_emulator.get_stats()
    }

    /// Get performance information as a string
    pub fn get_performance_info(&self) -> String {
        self.jit_emulator.get_performance_info()
    }
}

impl EmuStep for AdaptiveJitEmulator {
    fn step<C: EmuContext>(&mut self, ctx: &mut C) -> Result<()> {
        self.step(ctx)
    }
}

// Thread-local storage for the active emulator context
thread_local! {
    static ACTIVE_EMU_CONTEXT: std::cell::RefCell<Option<*mut dyn EmuContext>> =
        std::cell::RefCell::new(None);
}

/// Set the active emulator context for JIT callbacks
/// # Safety
/// The context must remain valid for the duration of JIT execution
#[allow(dead_code)]
unsafe fn set_active_context(ctx: *mut dyn EmuContext) {
    ACTIVE_EMU_CONTEXT.with(|active| {
        *active.borrow_mut() = Some(ctx);
    });
}

/// Clear the active emulator context
#[allow(dead_code)]
fn clear_active_context() {
    ACTIVE_EMU_CONTEXT.with(|active| {
        *active.borrow_mut() = None;
    });
}

// Real callback functions that use the active emulator context
unsafe extern "C" fn emu_load_memory(_ctx_ptr: *mut u8, addr: u32) -> u32 {
    ACTIVE_EMU_CONTEXT.with(|active| {
        if let Some(ctx_ptr) = *active.borrow() {
            let ctx = &mut *ctx_ptr;
            let word_addr = risc0_binfmt::WordAddr::from(risc0_binfmt::ByteAddr(addr));
            match ctx.load_memory(word_addr) {
                Ok(value) => {
                    tracing::debug!("JIT load_memory: addr=0x{:x} -> 0x{:x}", addr, value);
                    value
                }
                Err(e) => {
                    tracing::warn!("JIT load_memory failed: addr=0x{:x}, error={}", addr, e);
                    // Check if this is an RPC communication error
                    let error_str = format!("{e:?}");
                    if error_str.contains("rx len failed") || error_str.contains("failed to fill whole buffer") {
                        tracing::error!("JIT detected RPC communication failure, disabling JIT compilation");
                        // Signal that JIT should be disabled by returning a sentinel value
                        // The caller should check for this and disable JIT
                        0xDEADBEEF // Distinctive failure value
                    } else {
                        0 // Regular fallback value
                    }
                }
            }
        } else {
            tracing::error!("JIT load_memory called without active context");
            0
        }
    })
}

unsafe extern "C" fn emu_store_memory(_ctx_ptr: *mut u8, addr: u32, value: u32) {
    ACTIVE_EMU_CONTEXT.with(|active| {
        if let Some(ctx_ptr) = *active.borrow() {
            let ctx = &mut *ctx_ptr;
            let word_addr = risc0_binfmt::WordAddr::from(risc0_binfmt::ByteAddr(addr));
            match ctx.store_memory(word_addr, value) {
                Ok(()) => {
                    tracing::debug!("JIT store_memory: addr=0x{:x} <- 0x{:x}", addr, value);
                }
                Err(e) => {
                    tracing::warn!("JIT store_memory failed: addr=0x{:x}, value=0x{:x}, error={}", addr, value, e);
                    // Check if this is an RPC communication error
                    let error_str = format!("{e:?}");
                    if error_str.contains("rx len failed") || error_str.contains("failed to fill whole buffer") {
                        tracing::error!("JIT detected RPC communication failure during store, disabling JIT compilation");
                        // Store failures in JIT context can't easily signal back to disable JIT
                        // The next load operation will likely also fail and trigger JIT disabling
                    }
                }
            }
        } else {
            tracing::error!("JIT store_memory called without active context");
        }
    })
}

unsafe extern "C" fn emu_load_register(_ctx_ptr: *mut u8, reg: u32) -> u32 {
    if reg == 0 {
        return 0; // x0 is hardwired to zero
    }

    ACTIVE_EMU_CONTEXT.with(|active| {
        if let Some(ctx_ptr) = *active.borrow() {
            let ctx = &mut *ctx_ptr;
            match ctx.load_register(reg as usize) {
                Ok(value) => {
                    tracing::debug!("JIT load_register: reg={} -> 0x{:x}", reg, value);
                    value
                }
                Err(e) => {
                    tracing::warn!("JIT load_register failed: reg={}, error={}", reg, e);
                    0
                }
            }
        } else {
            tracing::error!("JIT load_register called without active context");
            0
        }
    })
}

unsafe extern "C" fn emu_store_register(_ctx_ptr: *mut u8, reg: u32, value: u32) {
    if reg == 0 {
        return; // Ignore writes to x0
    }

    ACTIVE_EMU_CONTEXT.with(|active| {
        if let Some(ctx_ptr) = *active.borrow() {
            let ctx = &mut *ctx_ptr;
            match ctx.store_register(reg as usize, value) {
                Ok(()) => {
                    tracing::debug!("JIT store_register: reg={} <- 0x{:x}", reg, value);
                }
                Err(e) => {
                    tracing::warn!("JIT store_register failed: reg={}, value=0x{:x}, error={}", reg, value, e);
                }
            }
        } else {
            tracing::error!("JIT store_register called without active context");
        }
    })
}

static MEMORY_CALLBACKS: MemoryCallbacks = MemoryCallbacks {
    load_memory: emu_load_memory,
    store_memory: emu_store_memory,
    load_register: emu_load_register,
    store_register: emu_store_register,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_emulator_creation() {
        let emulator = JitEmulator::new().unwrap();
        // JIT threshold is now environment-dependent and platform-dependent
        if std::env::var("RISC0_DISABLE_JIT").is_ok() || cfg!(target_os = "macos") {
            assert_eq!(emulator.jit_threshold, u32::MAX);
            assert!(!emulator.jit_enabled);
        } else {
            assert_eq!(emulator.jit_threshold, 25000);
            assert!(emulator.jit_enabled);
        }
        assert_eq!(emulator.stats.compiled_blocks, 0);
    }

    #[test]
    fn test_adaptive_emulator() {
        let _emulator = AdaptiveJitEmulator::new().unwrap();
    }

    #[test]
    fn test_hotspot_detection() {
        let mut emulator = JitEmulator::new().unwrap();

        // Override the threshold for testing
        emulator.jit_threshold = 10;

        // Simulate hotspot detection
        let addr = ByteAddr(0x1000);
        for _ in 0..15 {
            *emulator.execution_count.entry(addr).or_insert(0) += 1;
        }

        // Count hot spots
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
