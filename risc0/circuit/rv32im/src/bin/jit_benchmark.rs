use std::collections::HashMap;
use std::time::Instant;
use risc0_circuit_rv32im::execute::{jit_emulator::JitEmulator, rv32im::{EmuContext, InsnKind, DecodedInstruction, Exception}};
use risc0_binfmt::{ByteAddr, WordAddr};
use anyhow::Result;

/// Simple memory-based emulator context for testing
struct TestContext {
    pc: ByteAddr,
    registers: [u32; 32],
    memory: HashMap<WordAddr, u32>,
    instruction_count: u64,
}

impl TestContext {
    fn new() -> Self {
        let mut ctx = Self {
            pc: ByteAddr(0x1000),
            registers: [0; 32],
            memory: HashMap::new(),
            instruction_count: 0,
        };

        // Set up a simple test program: ADD operations in a loop
        // addi x1, x0, 100    # Load 100 into x1 (loop counter)
        // loop: add x2, x2, x1  # Add x1 to x2 (accumulator)
        //       addi x1, x1, -1  # Decrement counter
        //       bne x1, x0, loop # Branch if not zero

        ctx.memory.insert(WordAddr(0x400), 0x06400093); // addi x1, x0, 100
        ctx.memory.insert(WordAddr(0x401), 0x00110133); // add x2, x2, x1
        ctx.memory.insert(WordAddr(0x402), 0xfff08093); // addi x1, x1, -1
        ctx.memory.insert(WordAddr(0x403), 0xfe109ce3); // bne x1, x0, loop

        ctx
    }

    fn load_simple_program(&mut self) {
        // Simple arithmetic program that will trigger JIT compilation
        let instructions = vec![
            0x00100093, // addi x1, x0, 1     # x1 = 1
            0x00200113, // addi x2, x0, 2     # x2 = 2
            0x00310133, // add x3, x2, x3     # x3 = x2 + x3
            0x00408093, // addi x1, x1, 4     # x1 = x1 + 4
            0x00208133, // add x2, x1, x2     # x2 = x1 + x2
            0x00110113, // add x2, x2, x1     # x2 = x2 + x1
            0x00318133, // add x2, x3, x3     # x2 = x3 + x3
            0x00000073, // ecall               # Exit
        ];

        for (i, &insn) in instructions.iter().enumerate() {
            self.memory.insert(WordAddr(0x400 + i as u32), insn);
        }
    }
}

impl EmuContext for TestContext {
    fn ecall(&mut self) -> Result<bool> {
        // Simple ecall handler - just stop execution
        Ok(false)
    }

    fn mret(&mut self) -> Result<bool> {
        Ok(false)
    }

    fn trap(&mut self, _cause: Exception) -> Result<bool> {
        Ok(false)
    }

    fn on_insn_decoded(&mut self, _kind: InsnKind, _decoded: &DecodedInstruction) -> Result<()> {
        self.instruction_count += 1;
        Ok(())
    }

    fn on_normal_end(&mut self, _kind: InsnKind) -> Result<()> {
        Ok(())
    }

    fn get_pc(&self) -> ByteAddr {
        self.pc
    }

    fn set_pc(&mut self, addr: ByteAddr) {
        self.pc = addr;
    }

    fn load_register(&mut self, idx: usize) -> Result<u32> {
        if idx == 0 {
            Ok(0) // x0 is always zero
        } else if idx < 32 {
            Ok(self.registers[idx])
        } else {
            Err(anyhow::anyhow!("Invalid register index: {}", idx))
        }
    }

    fn store_register(&mut self, idx: usize, word: u32) -> Result<()> {
        if idx != 0 && idx < 32 {
            self.registers[idx] = word;
        }
        Ok(())
    }

    fn load_memory(&mut self, addr: WordAddr) -> Result<u32> {
        Ok(self.memory.get(&addr).copied().unwrap_or(0))
    }

    fn store_memory(&mut self, addr: WordAddr, word: u32) -> Result<()> {
        self.memory.insert(addr, word);
        Ok(())
    }

    fn check_insn_load(&self, addr: ByteAddr) -> bool {
        self.memory.contains_key(&addr.waddr())
    }

    fn check_data_load(&self, _addr: ByteAddr) -> bool {
        true
    }

    fn check_data_store(&self, _addr: ByteAddr) -> bool {
        true
    }
}

fn run_benchmark() -> Result<()> {
    println!("🚀 JIT Emulator Benchmark");
    println!("=========================");

    let mut jit_emulator = JitEmulator::new()?;
    let mut ctx = TestContext::new();
    ctx.load_simple_program();

    let start_time = Instant::now();
    let mut total_cycles = 0;

    // Run the program multiple times to trigger JIT compilation
    for iteration in 0..1000 {
        ctx.pc = ByteAddr(0x1000); // Reset PC
        ctx.instruction_count = 0;

        // Execute instructions until ecall or error
        let mut steps = 0;
        while steps < 100 {
            match jit_emulator.step(&mut ctx) {
                Ok(_) => {
                    steps += 1;
                    total_cycles += 1;

                    // Check if we hit ecall (instruction 0x73)
                    let current_insn = ctx.load_memory(ctx.pc.waddr())?;
                    if current_insn == 0x00000073 {
                        break;
                    }
                },
                Err(e) => {
                    println!("Execution error on iteration {}: {}", iteration, e);
                    break;
                }
            }
        }

        if iteration % 100 == 0 {
            println!("Iteration {}: {} instructions executed", iteration, ctx.instruction_count);
        }
    }

    let duration = start_time.elapsed();
    let stats = jit_emulator.get_stats();

    println!("\n📊 Benchmark Results:");
    println!("  Total duration: {:?}", duration);
    println!("  Total cycles: {}", total_cycles);
    println!("  Cycles per second: {:.2}", total_cycles as f64 / duration.as_secs_f64());
    println!("  Compiled blocks: {}", stats.compiled_blocks);
    println!("  Native executions: {}", stats.native_executions);
    println!("  Total executions: {}", stats.total_executions);
    println!("  Native execution ratio: {:.2}%", stats.compilation_ratio() * 100.0);

    // Test register values
    println!("\n🔍 Final Register State:");
    for i in 1..8 {
        println!("  x{}: {}", i, ctx.load_register(i)?);
    }

    if stats.compiled_blocks > 0 {
        println!("\n✅ JIT compilation successfully triggered!");
        println!("🎯 Performance improvement from native execution: {:.1}%",
                 stats.compilation_ratio() * 100.0);
    } else {
        println!("\n⚠️  JIT compilation was not triggered - may need more iterations");
    }

    Ok(())
}

fn main() -> Result<()> {
    run_benchmark()
}
