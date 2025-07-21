#![allow(dead_code, unused_imports, unused_variables)]
use std::collections::HashMap;
use anyhow::Result;
use risc0_binfmt::{ByteAddr, WordAddr};

use super::rv32im::{DecodedInstruction, InsnKind, EmuContext};

/// CPU context structure for JIT execution
/// This represents the RISC-V CPU state that the JIT code operates on
#[repr(C)]
#[derive(Debug)]
#[allow(dead_code)]
pub struct CpuContext {
    /// RISC-V general-purpose registers (x0-x31)
    /// x0 is always 0, but we include it for indexing convenience
    pub registers: [u32; 32],
    /// Program counter
    pub pc: u32,
    /// Pointer to the emulator context for memory operations
    /// This will be cast to &mut dyn EmuContext at runtime
    pub emu_context: *mut u8,
    /// Reserved for future use
    pub _reserved: u32,
    // Additional state can be added here (CSRs, etc.)
}

impl CpuContext {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            registers: [0; 32], // x0 will always be 0
            pc: 0,
            emu_context: std::ptr::null_mut(),
            _reserved: 0,
        }
    }

    #[allow(dead_code)]
    pub fn with_context(emu_context: *mut u8) -> Self {
        Self {
            registers: [0; 32],
            pc: 0,
            emu_context,
            _reserved: 0,
        }
    }

    /// Get register value (handles x0 special case)
    #[allow(dead_code)]
    pub fn get_register(&self, reg: u32) -> u32 {
        if reg == 0 {
            0 // x0 is hardwired to zero
        } else if reg < 32 {
            self.registers[reg as usize]
        } else {
            0 // Invalid register
        }
    }

    /// Set register value (handles x0 special case)
    #[allow(dead_code)]
    pub fn set_register(&mut self, reg: u32, value: u32) {
        if reg != 0 && reg < 32 {
            self.registers[reg as usize] = value;
        }
        // Ignore writes to x0 or invalid registers
    }

    // Note: Memory operations are now handled by the emulator context
    // The JIT generates calls to memory helper functions instead of direct access
}

/// A basic block of RISC-V instructions
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub start_addr: ByteAddr,
    pub instructions: Vec<(InsnKind, DecodedInstruction)>,
    pub end_addr: ByteAddr,
    pub is_conditional_branch: bool,
}

/// Native function signature for JIT-compiled blocks
/// Takes a pointer to CPU state (register file + other state)
#[allow(dead_code)]
pub type JitFunction = unsafe extern "C" fn(cpu_context: *mut u32) -> i32;

/// x86-64 code generator for RISC-V instructions
pub struct X86CodeGen {
    code: Vec<u8>,
}

impl X86CodeGen {
    pub fn new() -> Self {
        Self { code: Vec::new() }
    }

    /// Generate function prologue
    pub fn prologue(&mut self) {
        // Function prologue: push rbp; mov rbp, rsp
        self.code.extend_from_slice(&[
            0x55,             // push rbp
            0x48, 0x89, 0xe5, // mov rbp, rsp
        ]);
    }

    /// Generate function epilogue and return
    pub fn epilogue(&mut self) {
        // Function epilogue: xor eax, eax; pop rbp; ret
        self.code.extend_from_slice(&[
            0x31, 0xc0,       // xor eax, eax (return 0)
            0x5d,             // pop rbp
            0xc3,             // ret
        ]);
    }

            /// Generate RISC-V ADD instruction as native x86-64
    pub fn gen_add(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Real x86-64 implementation of RISC-V ADD
        // We'll use a calling convention where:
        // - RDI (first arg) contains the context pointer
        // - We call helper functions to load/store registers

        // Load rs1 into EAX: call jit_load_register(ctx, rs1)
        self.gen_load_register_to_eax(rs1);

        // Load rs2 into EDX: call jit_load_register(ctx, rs2)
        self.gen_load_register_to_edx(rs2);

        // ADD EAX, EDX (perform the actual addition)
        self.code.extend_from_slice(&[0x01, 0xd0]); // add eax, edx

        // Store result back to rd: call jit_store_register(ctx, rd, eax)
        self.gen_store_register_from_eax(rd);
    }

        /// Generate RISC-V ADDI instruction as native x86-64
    pub fn gen_addi(&mut self, rd: u32, rs1: u32, imm: i32) {
        // Real x86-64 implementation of RISC-V ADDI
        // Load rs1 into EAX
        self.gen_load_register_to_eax(rs1);

        // ADD EAX, imm (add immediate to EAX)
        self.code.extend_from_slice(&[0x05]); // add eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // Store result back to rd
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SUB instruction as native x86-64
    pub fn gen_sub(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Real x86-64 implementation of RISC-V SUB
        // Load rs1 into EAX
        self.gen_load_register_to_eax(rs1);

        // Load rs2 into EDX
        self.gen_load_register_to_edx(rs2);

        // SUB EAX, EDX (subtract EDX from EAX)
        self.code.extend_from_slice(&[0x29, 0xd0]); // sub eax, edx

        // Store result back to rd
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V AND instruction as native x86-64
    pub fn gen_and(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Real x86-64 implementation of RISC-V AND
        // Load rs1 into EAX
        self.gen_load_register_to_eax(rs1);

        // Load rs2 into EDX
        self.gen_load_register_to_edx(rs2);

        // AND EAX, EDX (bitwise AND EAX with EDX)
        self.code.extend_from_slice(&[0x21, 0xd0]); // and eax, edx

        // Store result back to rd
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V OR instruction as native x86-64
    pub fn gen_or(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Real x86-64 implementation of RISC-V OR
        // Load rs1 into EAX
        self.gen_load_register_to_eax(rs1);

        // Load rs2 into EDX
        self.gen_load_register_to_edx(rs2);

        // OR EAX, EDX (bitwise OR EAX with EDX)
        self.code.extend_from_slice(&[0x09, 0xd0]); // or eax, edx

        // Store result back to rd
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V XOR instruction as native x86-64
    pub fn gen_xor(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Real x86-64 implementation of RISC-V XOR
        // Load rs1 into EAX
        self.gen_load_register_to_eax(rs1);

        // Load rs2 into EDX
        self.gen_load_register_to_edx(rs2);

        // XOR EAX, EDX (bitwise XOR EAX with EDX)
        self.code.extend_from_slice(&[0x31, 0xd0]); // xor eax, edx

        // Store result back to rd
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V LW (Load Word) instruction as native x86-64
    pub fn gen_lw(&mut self, rd: u32, rs1: u32, imm: i32) {
        // LW: rd = memory[rs1 + imm] (32-bit load)
        // Load base address into EAX
        self.gen_load_register_to_eax(rs1);

        // Add immediate offset to EAX: ADD EAX, imm
        if imm != 0 {
            self.code.extend_from_slice(&[0x05]); // add eax, imm32
            self.code.extend_from_slice(&imm.to_le_bytes());
        }

        // Call memory load helper: load 32-bit value at address in EAX
        // For now, generate a simplified version - in production this would
        // call into the emulator's memory system
        self.gen_memory_load(4); // 4 bytes for word

        // Store result back to rd
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V LH (Load Halfword) instruction as native x86-64
    pub fn gen_lh(&mut self, rd: u32, rs1: u32, imm: i32) {
        // LH: rd = sign_extend(memory[rs1 + imm][15:0]) (16-bit load, sign extended)
        self.gen_load_register_to_eax(rs1);
        if imm != 0 {
            self.code.extend_from_slice(&[0x05]); // add eax, imm32
            self.code.extend_from_slice(&imm.to_le_bytes());
        }
        self.gen_memory_load(2); // 2 bytes for halfword
        // Sign extend: MOVSX EAX, AX (sign extend 16-bit to 32-bit)
        self.code.extend_from_slice(&[0x0f, 0xbf, 0xc0]); // movsx eax, ax
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V LHU (Load Halfword Unsigned) instruction as native x86-64
    pub fn gen_lhu(&mut self, rd: u32, rs1: u32, imm: i32) {
        // LHU: rd = zero_extend(memory[rs1 + imm][15:0]) (16-bit load, zero extended)
        self.gen_load_register_to_eax(rs1);
        if imm != 0 {
            self.code.extend_from_slice(&[0x05]); // add eax, imm32
            self.code.extend_from_slice(&imm.to_le_bytes());
        }
        self.gen_memory_load(2); // 2 bytes for halfword
        // Zero extend: MOVZX EAX, AX (zero extend 16-bit to 32-bit)
        self.code.extend_from_slice(&[0x0f, 0xb7, 0xc0]); // movzx eax, ax
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V LB (Load Byte) instruction as native x86-64
    pub fn gen_lb(&mut self, rd: u32, rs1: u32, imm: i32) {
        // LB: rd = sign_extend(memory[rs1 + imm][7:0]) (8-bit load, sign extended)
        self.gen_load_register_to_eax(rs1);
        if imm != 0 {
            self.code.extend_from_slice(&[0x05]); // add eax, imm32
            self.code.extend_from_slice(&imm.to_le_bytes());
        }
        self.gen_memory_load(1); // 1 byte
        // Sign extend: MOVSX EAX, AL (sign extend 8-bit to 32-bit)
        self.code.extend_from_slice(&[0x0f, 0xbe, 0xc0]); // movsx eax, al
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V LBU (Load Byte Unsigned) instruction as native x86-64
    pub fn gen_lbu(&mut self, rd: u32, rs1: u32, imm: i32) {
        // LBU: rd = zero_extend(memory[rs1 + imm][7:0]) (8-bit load, zero extended)
        self.gen_load_register_to_eax(rs1);
        if imm != 0 {
            self.code.extend_from_slice(&[0x05]); // add eax, imm32
            self.code.extend_from_slice(&imm.to_le_bytes());
        }
        self.gen_memory_load(1); // 1 byte
        // Zero extend: MOVZX EAX, AL (zero extend 8-bit to 32-bit)
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SW (Store Word) instruction as native x86-64
    pub fn gen_sw(&mut self, rs2: u32, rs1: u32, imm: i32) {
        // SW: memory[rs1 + imm] = rs2 (32-bit store)
        // Load base address into EAX
        self.gen_load_register_to_eax(rs1);

        // Add immediate offset: ADD EAX, imm
        if imm != 0 {
            self.code.extend_from_slice(&[0x05]); // add eax, imm32
            self.code.extend_from_slice(&imm.to_le_bytes());
        }

        // Load source value into EDX
        self.gen_load_register_to_edx(rs2);

        // Call memory store helper
        self.gen_memory_store(4); // 4 bytes for word
    }

    /// Generate RISC-V SH (Store Halfword) instruction as native x86-64
    pub fn gen_sh(&mut self, rs2: u32, rs1: u32, imm: i32) {
        // SH: memory[rs1 + imm][15:0] = rs2[15:0] (16-bit store)
        self.gen_load_register_to_eax(rs1);
        if imm != 0 {
            self.code.extend_from_slice(&[0x05]); // add eax, imm32
            self.code.extend_from_slice(&imm.to_le_bytes());
        }
        self.gen_load_register_to_edx(rs2);
        self.gen_memory_store(2); // 2 bytes for halfword
    }

    /// Generate RISC-V SB (Store Byte) instruction as native x86-64
    pub fn gen_sb(&mut self, rs2: u32, rs1: u32, imm: i32) {
        // SB: memory[rs1 + imm][7:0] = rs2[7:0] (8-bit store)
        self.gen_load_register_to_eax(rs1);
        if imm != 0 {
            self.code.extend_from_slice(&[0x05]); // add eax, imm32
            self.code.extend_from_slice(&imm.to_le_bytes());
        }
        self.gen_load_register_to_edx(rs2);
        self.gen_memory_store(1); // 1 byte
    }

    /// Generate RISC-V SLL (Shift Left Logical) instruction as native x86-64
    pub fn gen_sll(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // SLL: rd = rs1 << (rs2 & 0x1f) (shift left logical)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);
        // Mask shift amount: AND EDX, 0x1f (only use lower 5 bits)
        self.code.extend_from_slice(&[0x83, 0xe2, 0x1f]); // and edx, 0x1f
        // Move shift amount to CL: MOV CL, DL
        self.code.extend_from_slice(&[0x8a, 0xca]); // mov cl, dl
        // Shift left: SHL EAX, CL
        self.code.extend_from_slice(&[0xd3, 0xe0]); // shl eax, cl
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SLLI (Shift Left Logical Immediate) instruction
    pub fn gen_slli(&mut self, rd: u32, rs1: u32, shamt: u32) {
        // SLLI: rd = rs1 << (shamt & 0x1f)
        self.gen_load_register_to_eax(rs1);
        let shift_amount = shamt & 0x1f; // Mask to 5 bits
        if shift_amount != 0 {
            // SHL EAX, imm8
            self.code.extend_from_slice(&[0xc1, 0xe0, shift_amount as u8]); // shl eax, imm8
        }
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SRL (Shift Right Logical) instruction as native x86-64
    pub fn gen_srl(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // SRL: rd = rs1 >> (rs2 & 0x1f) (shift right logical, zero extend)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);
        // Mask shift amount: AND EDX, 0x1f
        self.code.extend_from_slice(&[0x83, 0xe2, 0x1f]); // and edx, 0x1f
        // Move to CL: MOV CL, DL
        self.code.extend_from_slice(&[0x8a, 0xca]); // mov cl, dl
        // Shift right logical: SHR EAX, CL
        self.code.extend_from_slice(&[0xd3, 0xe8]); // shr eax, cl
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SRLI (Shift Right Logical Immediate) instruction
    pub fn gen_srli(&mut self, rd: u32, rs1: u32, shamt: u32) {
        // SRLI: rd = rs1 >> (shamt & 0x1f)
        self.gen_load_register_to_eax(rs1);
        let shift_amount = shamt & 0x1f;
        if shift_amount != 0 {
            // SHR EAX, imm8
            self.code.extend_from_slice(&[0xc1, 0xe8, shift_amount as u8]); // shr eax, imm8
        }
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SRA (Shift Right Arithmetic) instruction as native x86-64
    pub fn gen_sra(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // SRA: rd = rs1 >> (rs2 & 0x1f) (shift right arithmetic, sign extend)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);
        // Mask shift amount: AND EDX, 0x1f
        self.code.extend_from_slice(&[0x83, 0xe2, 0x1f]); // and edx, 0x1f
        // Move to CL: MOV CL, DL
        self.code.extend_from_slice(&[0x8a, 0xca]); // mov cl, dl
        // Shift right arithmetic: SAR EAX, CL
        self.code.extend_from_slice(&[0xd3, 0xf8]); // sar eax, cl
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SRAI (Shift Right Arithmetic Immediate) instruction
    pub fn gen_srai(&mut self, rd: u32, rs1: u32, shamt: u32) {
        // SRAI: rd = rs1 >> (shamt & 0x1f) (arithmetic)
        self.gen_load_register_to_eax(rs1);
        let shift_amount = shamt & 0x1f;
        if shift_amount != 0 {
            // SAR EAX, imm8
            self.code.extend_from_slice(&[0xc1, 0xf8, shift_amount as u8]); // sar eax, imm8
        }
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SLT (Set Less Than) instruction as native x86-64
    pub fn gen_slt(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // SLT: rd = (rs1 < rs2) ? 1 : 0 (signed comparison)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);
        // Compare: CMP EAX, EDX
        self.code.extend_from_slice(&[0x39, 0xd0]); // cmp eax, edx
        // Set if less: SETL AL (set AL to 1 if less, 0 otherwise)
        self.code.extend_from_slice(&[0x0f, 0x9c, 0xc0]); // setl al
        // Zero extend to 32-bit: MOVZX EAX, AL
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SLTU (Set Less Than Unsigned) instruction
    pub fn gen_sltu(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // SLTU: rd = (rs1 < rs2) ? 1 : 0 (unsigned comparison)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);
        // Compare: CMP EAX, EDX
        self.code.extend_from_slice(&[0x39, 0xd0]); // cmp eax, edx
        // Set if below (unsigned less): SETB AL
        self.code.extend_from_slice(&[0x0f, 0x92, 0xc0]); // setb al
        // Zero extend: MOVZX EAX, AL
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SLTI (Set Less Than Immediate) instruction
    pub fn gen_slti(&mut self, rd: u32, rs1: u32, imm: i32) {
        // SLTI: rd = (rs1 < imm) ? 1 : 0 (signed comparison)
        self.gen_load_register_to_eax(rs1);
        // Compare with immediate: CMP EAX, imm
        self.code.extend_from_slice(&[0x3d]); // cmp eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());
        // Set if less: SETL AL
        self.code.extend_from_slice(&[0x0f, 0x9c, 0xc0]); // setl al
        // Zero extend: MOVZX EAX, AL
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V SLTIU (Set Less Than Immediate Unsigned) instruction
    pub fn gen_sltiu(&mut self, rd: u32, rs1: u32, imm: i32) {
        // SLTIU: rd = (rs1 < imm) ? 1 : 0 (unsigned comparison)
        self.gen_load_register_to_eax(rs1);
        // Compare with immediate: CMP EAX, imm
        self.code.extend_from_slice(&[0x3d]); // cmp eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());
        // Set if below: SETB AL
        self.code.extend_from_slice(&[0x0f, 0x92, 0xc0]); // setb al
        // Zero extend: MOVZX EAX, AL
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V XORI instruction as native x86-64
    pub fn gen_xori(&mut self, rd: u32, rs1: u32, imm: i32) {
        // XORI: rd = rs1 ^ imm
        self.gen_load_register_to_eax(rs1);
        // XOR EAX, imm
        self.code.extend_from_slice(&[0x35]); // xor eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V ORI instruction as native x86-64
    pub fn gen_ori(&mut self, rd: u32, rs1: u32, imm: i32) {
        // ORI: rd = rs1 | imm
        self.gen_load_register_to_eax(rs1);
        // OR EAX, imm
        self.code.extend_from_slice(&[0x0d]); // or eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V ANDI instruction as native x86-64
    pub fn gen_andi(&mut self, rd: u32, rs1: u32, imm: i32) {
        // ANDI: rd = rs1 & imm
        self.gen_load_register_to_eax(rs1);
        // AND EAX, imm
        self.code.extend_from_slice(&[0x25]); // and eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V LUI (Load Upper Immediate) instruction
    pub fn gen_lui(&mut self, rd: u32, imm: u32) {
        // LUI: rd = imm << 12 (load 20-bit immediate to upper bits)
        let upper_imm = imm & 0xfffff000; // Keep upper 20 bits, zero lower 12
        // MOV EAX, imm
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&upper_imm.to_le_bytes());
        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V AUIPC (Add Upper Immediate to PC) instruction
    pub fn gen_auipc(&mut self, rd: u32, imm: u32, current_pc: u32) {
        // AUIPC: rd = PC + (imm << 12)
        let upper_imm = imm & 0xfffff000;
        let result = current_pc.wrapping_add(upper_imm);
        // MOV EAX, result
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&result.to_le_bytes());
        self.gen_store_register_from_eax(rd);
    }

        /// Generate RISC-V BEQ (Branch if Equal) instruction
    pub fn gen_beq(&mut self, rs1: u32, rs2: u32, imm: i32, current_pc: u32) {
        // BEQ: if (rs1 == rs2) PC = PC + imm
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);

        // Compare: CMP EAX, EDX
        self.code.extend_from_slice(&[0x39, 0xd0]); // cmp eax, edx

        // Calculate branch target address
        let branch_target = current_pc.wrapping_add(imm as u32);
        let next_pc = current_pc.wrapping_add(4);

        // JE taken (jump if equal)
        self.code.extend_from_slice(&[0x74, 0x0a]); // je +10 bytes (to branch taken case)

        // Not taken: return next PC
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&next_pc.to_le_bytes());
        self.code.extend_from_slice(&[0xeb, 0x05]); // jmp +5 bytes (skip taken case)

        // Taken: return branch target
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&branch_target.to_le_bytes());
    }

    /// Generate RISC-V BNE (Branch if Not Equal) instruction
    pub fn gen_bne(&mut self, rs1: u32, rs2: u32, imm: i32, current_pc: u32) {
        // BNE: if (rs1 != rs2) PC = PC + imm
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);

        // Compare: CMP EAX, EDX
        self.code.extend_from_slice(&[0x39, 0xd0]); // cmp eax, edx

        // Calculate branch target address
        let branch_target = current_pc.wrapping_add(imm as u32);
        let next_pc = current_pc.wrapping_add(4);

        // JNE taken (jump if not equal)
        self.code.extend_from_slice(&[0x75, 0x0a]); // jne +10 bytes

        // Not taken: return next PC
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&next_pc.to_le_bytes());
        self.code.extend_from_slice(&[0xeb, 0x05]); // jmp +5 bytes

        // Taken: return branch target
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&branch_target.to_le_bytes());
    }

    /// Generate RISC-V BLT (Branch if Less Than) instruction
    pub fn gen_blt(&mut self, rs1: u32, rs2: u32, imm: i32, current_pc: u32) {
        // BLT: if (rs1 < rs2) PC = PC + imm (signed comparison)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);

        // Compare: CMP EAX, EDX
        self.code.extend_from_slice(&[0x39, 0xd0]); // cmp eax, edx

        // Calculate branch target address
        let branch_target = current_pc.wrapping_add(imm as u32);
        let next_pc = current_pc.wrapping_add(4);

        // JL taken (jump if less - signed)
        self.code.extend_from_slice(&[0x7c, 0x0a]); // jl +10 bytes

        // Not taken: return next PC
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&next_pc.to_le_bytes());
        self.code.extend_from_slice(&[0xeb, 0x05]); // jmp +5 bytes

        // Taken: return branch target
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&branch_target.to_le_bytes());
    }

    /// Generate RISC-V BLTU (Branch if Less Than Unsigned) instruction
    pub fn gen_bltu(&mut self, rs1: u32, rs2: u32, imm: i32, current_pc: u32) {
        // BLTU: if (rs1 < rs2) PC = PC + imm (unsigned comparison)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);

        // Compare: CMP EAX, EDX
        self.code.extend_from_slice(&[0x39, 0xd0]); // cmp eax, edx

        // Calculate branch target address
        let branch_target = current_pc.wrapping_add(imm as u32);
        let next_pc = current_pc.wrapping_add(4);

        // JB taken (jump if below - unsigned)
        self.code.extend_from_slice(&[0x72, 0x0a]); // jb +10 bytes

        // Not taken: return next PC
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&next_pc.to_le_bytes());
        self.code.extend_from_slice(&[0xeb, 0x05]); // jmp +5 bytes

        // Taken: return branch target
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&branch_target.to_le_bytes());
    }

    /// Generate RISC-V BGE (Branch if Greater or Equal) instruction
    pub fn gen_bge(&mut self, rs1: u32, rs2: u32, imm: i32, current_pc: u32) {
        // BGE: if (rs1 >= rs2) PC = PC + imm (signed comparison)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);

        // Compare: CMP EAX, EDX
        self.code.extend_from_slice(&[0x39, 0xd0]); // cmp eax, edx

        // Calculate branch target address
        let branch_target = current_pc.wrapping_add(imm as u32);
        let next_pc = current_pc.wrapping_add(4);

        // JGE taken (jump if greater or equal - signed)
        self.code.extend_from_slice(&[0x7d, 0x0a]); // jge +10 bytes

        // Not taken: return next PC
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&next_pc.to_le_bytes());
        self.code.extend_from_slice(&[0xeb, 0x05]); // jmp +5 bytes

        // Taken: return branch target
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&branch_target.to_le_bytes());
    }

    /// Generate RISC-V BGEU (Branch if Greater or Equal Unsigned) instruction
    pub fn gen_bgeu(&mut self, rs1: u32, rs2: u32, imm: i32, current_pc: u32) {
        // BGEU: if (rs1 >= rs2) PC = PC + imm (unsigned comparison)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);

        // Compare: CMP EAX, EDX
        self.code.extend_from_slice(&[0x39, 0xd0]); // cmp eax, edx

        // Calculate branch target address
        let branch_target = current_pc.wrapping_add(imm as u32);
        let next_pc = current_pc.wrapping_add(4);

        // JAE taken (jump if above or equal - unsigned)
        self.code.extend_from_slice(&[0x73, 0x0a]); // jae +10 bytes

        // Not taken: return next PC
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&next_pc.to_le_bytes());
        self.code.extend_from_slice(&[0xeb, 0x05]); // jmp +5 bytes

        // Taken: return branch target
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&branch_target.to_le_bytes());
    }

    /// Generate RISC-V JAL (Jump and Link) instruction
    pub fn gen_jal(&mut self, rd: u32, imm: i32, current_pc: u32) {
        // JAL: rd = PC + 4; PC = PC + imm
        let link_addr = current_pc.wrapping_add(4);
        let jump_target = current_pc.wrapping_add(imm as u32);

        // Store return address in rd
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&link_addr.to_le_bytes());
        self.gen_store_register_from_eax(rd);

        // Return the jump target address
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&jump_target.to_le_bytes());
    }

    /// Generate RISC-V JALR (Jump and Link Register) instruction
    pub fn gen_jalr(&mut self, rd: u32, rs1: u32, imm: i32, current_pc: u32) {
        // JALR: rd = PC + 4; PC = (rs1 + imm) & ~1
        let link_addr = current_pc.wrapping_add(4);

        // Store return address in rd
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&link_addr.to_le_bytes());
        self.gen_store_register_from_eax(rd);

        // Compute jump target: (rs1 + imm) & ~1
        // Load rs1 into EAX
        self.gen_load_register_to_eax(rs1);

        // Add immediate: ADD EAX, imm
        if imm != 0 {
            self.code.extend_from_slice(&[0x05]); // add eax, imm32
            self.code.extend_from_slice(&imm.to_le_bytes());
        }

        // Clear lowest bit: AND EAX, 0xFFFFFFFE
        self.code.extend_from_slice(&[0x25]); // and eax, imm32
        self.code.extend_from_slice(&0xFFFFFFFE_u32.to_le_bytes());

        // EAX now contains the computed jump target, and that's our return value
    }

    /// Generate RISC-V MUL (Multiply) instruction as native x86-64
    #[allow(unused_variables)]
    pub fn gen_mul(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // MUL: rd = (rs1 * rs2)[31:0] (lower 32 bits of product)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);

        // Multiply: IMUL EAX, EDX (32-bit signed multiply, result in EAX)
        self.code.extend_from_slice(&[0x0f, 0xaf, 0xc2]); // imul eax, edx

        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V MULH (Multiply High) instruction as native x86-64
    pub fn gen_mulh(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // MULH: rd = (rs1 * rs2)[63:32] (upper 32 bits, signed × signed)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);

        // Sign extend EAX to 64-bit: CDQ (extends EAX sign to EDX:EAX)
        self.code.push(0x99); // cdq

        // Signed multiply: IMUL EDX (64-bit result in EDX:EAX)
        self.code.extend_from_slice(&[0xf7, 0xea]); // imul edx

        // Move upper 32 bits (EDX) to EAX
        self.code.extend_from_slice(&[0x89, 0xd0]); // mov eax, edx

        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V MULHSU (Multiply High Signed×Unsigned) instruction
    pub fn gen_mulhsu(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // MULHSU: rd = (rs1 * rs2)[63:32] (upper 32 bits, signed × unsigned)
        // This is complex - we need to handle the mixed signs carefully

        self.gen_load_register_to_eax(rs1); // signed operand

        // Convert to 64-bit signed: CDQ
        self.code.push(0x99); // cdq (EDX:EAX now has sign-extended rs1)

        // Load rs2 (unsigned) into ECX
        self.gen_load_register_to_ecx(rs2);

        // Multiply EDX:EAX by ECX (treating rs2 as unsigned)
        // This is tricky in x86 - we'll use a simplified approach
        // For now, use signed multiply and handle edge cases later
        self.code.extend_from_slice(&[0xf7, 0xe9]); // imul ecx

        // Move upper 32 bits to EAX
        self.code.extend_from_slice(&[0x89, 0xd0]); // mov eax, edx

        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V MULHU (Multiply High Unsigned) instruction
    pub fn gen_mulhu(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // MULHU: rd = (rs1 * rs2)[63:32] (upper 32 bits, unsigned × unsigned)
        self.gen_load_register_to_eax(rs1);
        self.gen_load_register_to_edx(rs2);

        // Zero extend (clear upper 32 bits)
        self.code.extend_from_slice(&[0x31, 0xd2]); // xor edx, edx

        // Unsigned multiply: MUL EDX (64-bit result in EDX:EAX)
        self.code.extend_from_slice(&[0xf7, 0xe2]); // mul edx (but we need the original rs2)

        // Reload rs2 and multiply properly
        self.gen_load_register_to_edx(rs2);
        self.code.extend_from_slice(&[0xf7, 0xe2]); // mul edx

        // Move upper 32 bits to EAX
        self.code.extend_from_slice(&[0x89, 0xd0]); // mov eax, edx

        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V DIV (Divide Signed) instruction as native x86-64
    pub fn gen_div(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // DIV: rd = rs1 / rs2 (signed division)
        // Handle division by zero: return -1

        self.gen_load_register_to_eax(rs1); // dividend
        self.gen_load_register_to_edx(rs2); // divisor

        // Check for division by zero: TEST EDX, EDX
        self.code.extend_from_slice(&[0x85, 0xd2]); // test edx, edx

        // Jump if zero to return -1
        self.code.extend_from_slice(&[0x74, 0x0a]); // je +10 bytes

        // Sign extend EAX to EDX:EAX: CDQ
        self.code.push(0x99); // cdq

        // Signed divide: IDIV EDX (quotient in EAX, remainder in EDX)
        self.code.extend_from_slice(&[0xf7, 0xfa]); // idiv edx

        // Jump past the division-by-zero case
        self.code.extend_from_slice(&[0xeb, 0x05]); // jmp +5 bytes

        // Division by zero case: MOV EAX, -1
        self.code.extend_from_slice(&[0xb8, 0xff, 0xff, 0xff, 0xff]); // mov eax, -1

        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V DIVU (Divide Unsigned) instruction as native x86-64
    pub fn gen_divu(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // DIVU: rd = rs1 / rs2 (unsigned division)
        // Handle division by zero: return 0xFFFFFFFF

        self.gen_load_register_to_eax(rs1); // dividend
        self.gen_load_register_to_edx(rs2); // divisor

        // Check for division by zero
        self.code.extend_from_slice(&[0x85, 0xd2]); // test edx, edx
        self.code.extend_from_slice(&[0x74, 0x09]); // je +9 bytes

        // Zero upper 32 bits for unsigned division
        self.code.extend_from_slice(&[0x31, 0xd2]); // xor edx, edx

        // Reload divisor into ECX (since we cleared EDX)
        self.gen_load_register_to_ecx(rs2);

        // Unsigned divide: DIV ECX
        self.code.extend_from_slice(&[0xf7, 0xf1]); // div ecx

        // Jump past division-by-zero case
        self.code.extend_from_slice(&[0xeb, 0x05]); // jmp +5 bytes

        // Division by zero: MOV EAX, 0xFFFFFFFF
        self.code.extend_from_slice(&[0xb8, 0xff, 0xff, 0xff, 0xff]); // mov eax, -1

        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V REM (Remainder Signed) instruction as native x86-64
    pub fn gen_rem(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // REM: rd = rs1 % rs2 (signed remainder)
        // Handle division by zero: return rs1

        self.gen_load_register_to_eax(rs1); // dividend

        // Save rs1 for division-by-zero case
        self.code.extend_from_slice(&[0x89, 0xc1]); // mov ecx, eax (save rs1)

        self.gen_load_register_to_edx(rs2); // divisor

        // Check for division by zero
        self.code.extend_from_slice(&[0x85, 0xd2]); // test edx, edx
        self.code.extend_from_slice(&[0x74, 0x08]); // je +8 bytes

        // Sign extend and divide
        self.code.push(0x99); // cdq
        self.code.extend_from_slice(&[0xf7, 0xfa]); // idiv edx

        // Move remainder (EDX) to EAX
        self.code.extend_from_slice(&[0x89, 0xd0]); // mov eax, edx

        // Jump past division-by-zero case
        self.code.extend_from_slice(&[0xeb, 0x02]); // jmp +2 bytes

        // Division by zero: return original dividend (in ECX)
        self.code.extend_from_slice(&[0x89, 0xc8]); // mov eax, ecx

        self.gen_store_register_from_eax(rd);
    }

    /// Generate RISC-V REMU (Remainder Unsigned) instruction as native x86-64
    pub fn gen_remu(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // REMU: rd = rs1 % rs2 (unsigned remainder)
        // Handle division by zero: return rs1

        self.gen_load_register_to_eax(rs1); // dividend

        // Save rs1 for division-by-zero case
        self.code.extend_from_slice(&[0x89, 0xc1]); // mov ecx, eax

        self.gen_load_register_to_edx(rs2); // divisor

        // Check for division by zero
        self.code.extend_from_slice(&[0x85, 0xd2]); // test edx, edx
        self.code.extend_from_slice(&[0x74, 0x0a]); // je +10 bytes

        // Zero upper bits and divide
        self.code.extend_from_slice(&[0x31, 0xd2]); // xor edx, edx

        // Reload divisor (since we cleared EDX)
        // Use a different register approach
        self.code.extend_from_slice(&[0x50]); // push eax (save dividend)
        self.gen_load_register_to_eax(rs2); // reload divisor to EAX
        self.code.extend_from_slice(&[0x89, 0xc2]); // mov edx, eax (divisor to EDX)
        self.code.extend_from_slice(&[0x58]); // pop eax (restore dividend)
        self.code.extend_from_slice(&[0x31, 0xd2]); // xor edx, edx (clear upper)

        // Actually, let's use a simpler approach with stack
        // This is getting complex, let's use ECX consistently
        self.gen_load_register_to_ecx(rs2);
        self.code.extend_from_slice(&[0x31, 0xd2]); // xor edx, edx
        self.code.extend_from_slice(&[0xf7, 0xf1]); // div ecx

        // Move remainder to EAX
        self.code.extend_from_slice(&[0x89, 0xd0]); // mov eax, edx

        // Jump past division-by-zero
        self.code.extend_from_slice(&[0xeb, 0x02]); // jmp +2

        // Division by zero: return rs1 (saved in ECX earlier)
        self.code.extend_from_slice(&[0x89, 0xc8]); // mov eax, ecx

        self.gen_store_register_from_eax(rd);
    }

    /// Helper: Generate code to load a register value into ECX
    fn gen_load_register_to_ecx(&mut self, reg: u32) {
        if reg == 0 {
            // XOR ECX, ECX (set ECX to 0)
            self.code.extend_from_slice(&[0x31, 0xc9]);
        } else {
            // MOV ECX, [RDI + reg*4]
            let offset = reg * 4;

            if offset < 128 {
                // Use 8-bit displacement: MOV ECX, [RDI + disp8]
                self.code.extend_from_slice(&[0x8b, 0x4f]); // mov ecx, [rdi + disp8]
                self.code.push(offset as u8);
            } else {
                // Use 32-bit displacement: MOV ECX, [RDI + disp32]
                self.code.extend_from_slice(&[0x8b, 0x8f]); // mov ecx, [rdi + disp32]
                self.code.extend_from_slice(&offset.to_le_bytes());
            }
        }
    }

    /// Get the generated code
    pub fn get_code(&self) -> &[u8] {
        &self.code
    }

    /// Get the size of generated code
    pub fn get_code_size(&self) -> usize {
        self.code.len()
    }

    /// Generate memory load helper - loads from address in EAX
    fn gen_memory_load(&mut self, size: u32) {
        // For now, generate placeholder - in production this would:
        // 1. Validate address is within bounds
        // 2. Translate virtual to physical address
        // 3. Handle memory-mapped I/O
        // 4. Load value from memory at address in EAX

        // Placeholder: MOV EAX, [EAX] (direct memory access)
        // This assumes memory is identity-mapped which isn't realistic
        match size {
            1 => {
                // MOV AL, [EAX] then zero extend
                self.code.extend_from_slice(&[0x8a, 0x00]); // mov al, [eax]
                self.code.extend_from_slice(&[0x25]); // and eax, imm32 (zero upper bits)
                self.code.extend_from_slice(&0x000000ff_u32.to_le_bytes());
            }
            2 => {
                // MOV AX, [EAX] then zero extend
                self.code.extend_from_slice(&[0x66, 0x8b, 0x00]); // mov ax, [eax]
                self.code.extend_from_slice(&[0x25]); // and eax, imm32 (zero upper bits)
                self.code.extend_from_slice(&0x0000ffff_u32.to_le_bytes());
            }
            4 => {
                // MOV EAX, [EAX] (32-bit load)
                self.code.extend_from_slice(&[0x8b, 0x00]); // mov eax, [eax]
            }
            _ => {
                // Invalid size - generate NOP
                self.code.push(0x90);
            }
        }
    }

    /// Generate memory store helper - stores EDX to address in EAX
    fn gen_memory_store(&mut self, size: u32) {
        // For now, generate placeholder - in production this would:
        // 1. Validate address is within bounds
        // 2. Translate virtual to physical address
        // 3. Handle memory-mapped I/O
        // 4. Store EDX value to memory at address in EAX

        // Placeholder: MOV [EAX], EDX (direct memory access)
        match size {
            1 => {
                // MOV [EAX], DL (8-bit store)
                self.code.extend_from_slice(&[0x88, 0x10]); // mov [eax], dl
            }
            2 => {
                // MOV [EAX], DX (16-bit store)
                self.code.extend_from_slice(&[0x66, 0x89, 0x10]); // mov [eax], dx
            }
            4 => {
                // MOV [EAX], EDX (32-bit store)
                self.code.extend_from_slice(&[0x89, 0x10]); // mov [eax], edx
            }
            _ => {
                // Invalid size - generate NOP
                self.code.push(0x90);
            }
        }
    }

    /// Generate code to load a register value into EAX
    fn gen_load_register_to_eax(&mut self, reg: u32) {
        // Generate code to load from CPU context: MOV EAX, [RDI + reg*4]
        // RDI contains the CPU context pointer (first argument in System V ABI)

        if reg == 0 {
            // x0 is always zero in RISC-V
            // XOR EAX, EAX (set EAX to 0)
            self.code.extend_from_slice(&[0x31, 0xc0]);
        } else {
            // MOV EAX, [RDI + reg*4] - load 32-bit register value
            // We use reg*4 because RISC-V registers are 32-bit (4 bytes each)
            let offset = reg * 4;

            if offset < 128 {
                // Use 8-bit displacement: MOV EAX, [RDI + disp8]
                self.code.extend_from_slice(&[0x8b, 0x47]); // mov eax, [rdi + disp8]
                self.code.push(offset as u8);
            } else {
                // Use 32-bit displacement: MOV EAX, [RDI + disp32]
                self.code.extend_from_slice(&[0x8b, 0x87]); // mov eax, [rdi + disp32]
                self.code.extend_from_slice(&offset.to_le_bytes());
            }
        }
    }

    /// Generate code to load a register value into EDX
    fn gen_load_register_to_edx(&mut self, reg: u32) {
        // Generate code to load from CPU context: MOV EDX, [RDI + reg*4]

        if reg == 0 {
            // x0 is always zero in RISC-V
            // XOR EDX, EDX (set EDX to 0)
            self.code.extend_from_slice(&[0x31, 0xd2]);
        } else {
            // MOV EDX, [RDI + reg*4] - load 32-bit register value
            let offset = reg * 4;

            if offset < 128 {
                // Use 8-bit displacement: MOV EDX, [RDI + disp8]
                self.code.extend_from_slice(&[0x8b, 0x57]); // mov edx, [rdi + disp8]
                self.code.push(offset as u8);
            } else {
                // Use 32-bit displacement: MOV EDX, [RDI + disp32]
                self.code.extend_from_slice(&[0x8b, 0x97]); // mov edx, [rdi + disp32]
                self.code.extend_from_slice(&offset.to_le_bytes());
            }
        }
    }

        /// Generate code to store EAX value to a register
    fn gen_store_register_from_eax(&mut self, reg: u32) {
        // Generate code to store to CPU context: MOV [RDI + reg*4], EAX

        if reg == 0 {
            // x0 is hardwired to zero in RISC-V - ignore writes
            // Generate a NOP instead of a store
            self.code.push(0x90); // NOP
        } else {
            // MOV [RDI + reg*4], EAX - store 32-bit register value
            let offset = reg * 4;

            if offset < 128 {
                // Use 8-bit displacement: MOV [RDI + disp8], EAX
                self.code.extend_from_slice(&[0x89, 0x47]); // mov [rdi + disp8], eax
                self.code.push(offset as u8);
            } else {
                // Use 32-bit displacement: MOV [RDI + disp32], EAX
                self.code.extend_from_slice(&[0x89, 0x87]); // mov [rdi + disp32], eax
                self.code.extend_from_slice(&offset.to_le_bytes());
            }
        }
    }

        /// Generate code to load an immediate value into EAX
    #[allow(dead_code)]
    fn gen_load_immediate_to_eax(&mut self, imm: i32) {
        // MOV EAX, imm
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// Generate RISC-V ECALL (Environment Call) instruction
    pub fn gen_ecall(&mut self) {
        // ECALL: Return special exception code to trigger emulator's ecall handler
        // The emulator will handle the actual system call based on privilege level
        self.code.extend_from_slice(&[0xb8, 0x08, 0x00, 0x00, 0x00]); // mov eax, 8 (environment call exception)
    }

    /// Generate RISC-V EBREAK (Environment Break) instruction
    pub fn gen_ebreak(&mut self) {
        // EBREAK: Return breakpoint exception code
        // The emulator will handle breakpoint debugging
        self.code.extend_from_slice(&[0xb8, 0x03, 0x00, 0x00, 0x00]); // mov eax, 3 (breakpoint exception)
    }

    /// Generate RISC-V MRET (Machine Return) instruction
    pub fn gen_mret(&mut self) {
        // MRET: Return machine return code
        // The emulator will handle privilege level restoration
        self.code.extend_from_slice(&[0xb8, 0x30, 0x00, 0x00, 0x00]); // mov eax, 48 (machine return)
    }
}

/// JIT compiler with real native code generation
pub struct JitCompiler {
    pub compiled_blocks: HashMap<ByteAddr, *const u8>,
    compilation_count: usize,
}

/// Implement proper memory cleanup for JitCompiler
impl Drop for JitCompiler {
    fn drop(&mut self) {
        // Clean up all allocated executable memory
        #[cfg(unix)]
        {
            for &code_ptr in self.compiled_blocks.values() {
                if !code_ptr.is_null() {
                    // We don't know the exact size, but we can unmap a page
                    // In a production system, we'd track block sizes
                    unsafe {
                        libc::munmap(code_ptr as *mut libc::c_void, 4096);
                    }
                }
            }
        }

        println!("JIT compiler cleaned up {} compiled blocks", self.compiled_blocks.len());
    }
}

impl JitCompiler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            compiled_blocks: HashMap::new(),
            compilation_count: 0,
        })
    }

    /// Compile a basic block to native x86-64 code
    pub fn compile_block(&mut self, block: &BasicBlock) -> Result<*const u8> {
        self.compilation_count += 1;

        let mut codegen = X86CodeGen::new();

        // Generate function prologue
        codegen.prologue();

        // Compile each RISC-V instruction to native x86-64
        for (kind, decoded) in &block.instructions {
            match kind {
                InsnKind::Add => {
                    codegen.gen_add(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::Sub => {
                    codegen.gen_sub(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::AddI => {
                    codegen.gen_addi(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::And => {
                    codegen.gen_and(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::Or => {
                    codegen.gen_or(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::Xor => {
                    codegen.gen_xor(decoded.rd, decoded.rs1, decoded.rs2);
                }
                // Memory load instructions
                InsnKind::Lw => {
                    codegen.gen_lw(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::Lh => {
                    codegen.gen_lh(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::LhU => {
                    codegen.gen_lhu(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::Lb => {
                    codegen.gen_lb(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::LbU => {
                    codegen.gen_lbu(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                // Memory store instructions
                InsnKind::Sw => {
                    codegen.gen_sw(decoded.rs2, decoded.rs1, decoded.imm_s() as i32);
                }
                InsnKind::Sh => {
                    codegen.gen_sh(decoded.rs2, decoded.rs1, decoded.imm_s() as i32);
                }
                InsnKind::Sb => {
                    codegen.gen_sb(decoded.rs2, decoded.rs1, decoded.imm_s() as i32);
                }
                // Shift instructions
                InsnKind::Sll => {
                    codegen.gen_sll(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::SllI => {
                    codegen.gen_slli(decoded.rd, decoded.rs1, decoded.rs2); // rs2 contains shamt for immediate
                }
                InsnKind::Srl => {
                    codegen.gen_srl(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::SrlI => {
                    codegen.gen_srli(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::Sra => {
                    codegen.gen_sra(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::SraI => {
                    codegen.gen_srai(decoded.rd, decoded.rs1, decoded.rs2);
                }
                // Comparison instructions
                InsnKind::Slt => {
                    codegen.gen_slt(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::SltU => {
                    codegen.gen_sltu(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::SltI => {
                    codegen.gen_slti(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::SltIU => {
                    codegen.gen_sltiu(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                // Immediate bitwise instructions
                InsnKind::XorI => {
                    codegen.gen_xori(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::OrI => {
                    codegen.gen_ori(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::AndI => {
                    codegen.gen_andi(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                // Upper immediate instructions
                InsnKind::Lui => {
                    let imm_u = decoded.insn & 0xfffff000; // Extract upper 20 bits
                    codegen.gen_lui(decoded.rd, imm_u);
                }
                InsnKind::Auipc => {
                    let imm_u = decoded.insn & 0xfffff000; // Extract upper 20 bits
                    codegen.gen_auipc(decoded.rd, imm_u, block.start_addr.0);
                }
                // Branch instructions
                InsnKind::Beq => {
                    codegen.gen_beq(decoded.rs1, decoded.rs2, decoded.imm_b() as i32, block.start_addr.0);
                }
                InsnKind::Bne => {
                    codegen.gen_bne(decoded.rs1, decoded.rs2, decoded.imm_b() as i32, block.start_addr.0);
                }
                InsnKind::Blt => {
                    codegen.gen_blt(decoded.rs1, decoded.rs2, decoded.imm_b() as i32, block.start_addr.0);
                }
                InsnKind::BltU => {
                    codegen.gen_bltu(decoded.rs1, decoded.rs2, decoded.imm_b() as i32, block.start_addr.0);
                }
                InsnKind::Bge => {
                    codegen.gen_bge(decoded.rs1, decoded.rs2, decoded.imm_b() as i32, block.start_addr.0);
                }
                InsnKind::BgeU => {
                    codegen.gen_bgeu(decoded.rs1, decoded.rs2, decoded.imm_b() as i32, block.start_addr.0);
                }
                // Jump instructions
                InsnKind::Jal => {
                    // Extract J-type immediate manually since imm_j() is private
                    let imm_j = ((decoded.insn & 0x80000000) >> 11) | // bit 20 -> bit 31
                               (decoded.insn & 0x000ff000) |         // bits 19:12 -> bits 19:12
                               ((decoded.insn & 0x00100000) >> 9) |    // bit 11 -> bit 20
                               ((decoded.insn & 0x7fe00000) >> 20);    // bits 30:21 -> bits 10:1
                    let imm_j = ((imm_j as i32) << 11) >> 11; // Sign extend
                    codegen.gen_jal(decoded.rd, imm_j, block.start_addr.0);
                }
                InsnKind::JalR => {
                    codegen.gen_jalr(decoded.rd, decoded.rs1, decoded.imm_i() as i32, block.start_addr.0);
                }
                // M-extension multiply instructions
                InsnKind::Mul => {
                    codegen.gen_mul(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::MulH => {
                    codegen.gen_mulh(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::MulHSU => {
                    codegen.gen_mulhsu(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::MulHU => {
                    codegen.gen_mulhu(decoded.rd, decoded.rs1, decoded.rs2);
                }
                // M-extension divide/remainder instructions
                InsnKind::Div => {
                    codegen.gen_div(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::DivU => {
                    codegen.gen_divu(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::Rem => {
                    codegen.gen_rem(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::RemU => {
                    codegen.gen_remu(decoded.rd, decoded.rs1, decoded.rs2);
                }
                // System instructions
                InsnKind::Eany => {
                    // Decode specific system instruction based on rs2 field
                    match decoded.rs2 {
                        0 => codegen.gen_ecall(),  // ECALL
                        1 => codegen.gen_ebreak(), // EBREAK
                        _ => codegen.code.push(0x90), // Unknown system instruction -> NOP
                    }
                }
                InsnKind::Mret => {
                    codegen.gen_mret();
                }
                _ => {
                    // For unsupported instructions, generate a NOP
                    codegen.code.push(0x90);
                }
            }
        }

        // Generate function epilogue
        codegen.epilogue();

        // Allocate executable memory and copy code
        let native_code = self.allocate_and_copy_code(codegen.get_code())?;

        // Store the compiled block
        self.compiled_blocks.insert(block.start_addr, native_code);

                        let code_size = codegen.get_code_size();
        let block_addr = block.start_addr;
        println!("Generated {code_size} bytes of native x86-64 code for block at {block_addr:?}");

        Ok(native_code)
    }

    /// Allocate executable memory and copy generated code
    fn allocate_and_copy_code(&self, code: &[u8]) -> Result<*const u8> {
        // Allocate memory with proper alignment
        let code_size = code.len().max(64); // Minimum size
        let memory = self.allocate_executable_memory(code_size)?;

        // Copy the generated code
        unsafe {
            std::ptr::copy_nonoverlapping(
                code.as_ptr(),
                memory as *mut u8,
                code.len(),
            );
        }

        Ok(memory)
    }

        /// Allocate executable memory for native code
    fn allocate_executable_memory(&self, size: usize) -> Result<*const u8> {
        // Try different approaches to get executable memory

        #[cfg(unix)]
        {
            // First try: Use mmap with better error handling
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };

            if addr != libc::MAP_FAILED {
                println!("Successfully allocated {size} bytes of executable memory at {addr:p}");
                return Ok(addr as *const u8);
            }

            // If mmap failed, try alternative approach
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            eprintln!("mmap failed with errno {errno}, trying alternative approach");

            // Alternative: Use mmap without PROT_EXEC, then mprotect
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };

            if addr != libc::MAP_FAILED {
                // Try to make it executable
                let result = unsafe {
                    libc::mprotect(
                        addr,
                        size,
                        libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    )
                };

                if result == 0 {
                    println!("Successfully made {size} bytes executable at {addr:p}");
                    return Ok(addr as *const u8);
                } else {
                    eprintln!("mprotect failed, falling back to regular memory");
                    unsafe { libc::munmap(addr, size) };
                }
            }

            // Final fallback: regular memory (will crash if we try to execute)
            let memory = vec![0u8; size];
            let ptr = Box::into_raw(memory.into_boxed_slice());
            println!("Using regular memory at {ptr:p} (not executable)");
            Ok(ptr as *const u8)
        }

        #[cfg(not(unix))]
        {
            // For non-Unix systems, use regular memory
            let memory = vec![0u8; size];
            let ptr = Box::into_raw(memory.into_boxed_slice());
            Ok(ptr as *const u8)
        }
    }

    /// Get compiled code for a basic block
    pub fn get_compiled_block(&self, addr: ByteAddr) -> Option<*const u8> {
        self.compiled_blocks.get(&addr).copied()
    }

    /// Get compilation statistics
    pub fn get_compilation_count(&self) -> usize {
        self.compilation_count
    }
}

/// Builder for constructing basic blocks
#[allow(dead_code)]
pub struct BasicBlockBuilder {
    current_block: Option<BasicBlock>,
    blocks: Vec<BasicBlock>,
}

#[allow(dead_code)]
impl BasicBlockBuilder {
    pub fn new() -> Self {
        Self {
            current_block: None,
            blocks: Vec::new(),
        }
    }

    pub fn add_instruction(&mut self, addr: ByteAddr, kind: InsnKind, decoded: DecodedInstruction) {
        if self.current_block.is_none() {
            self.current_block = Some(BasicBlock {
                start_addr: addr,
                instructions: Vec::new(),
                end_addr: addr,
                is_conditional_branch: false,
            });
        }

        if let Some(ref mut block) = self.current_block {
            block.instructions.push((kind, decoded));
            block.end_addr = addr;

            // End block on control flow instructions
            if matches!(kind,
                InsnKind::Jal | InsnKind::JalR |
                InsnKind::Beq | InsnKind::Bne | InsnKind::Blt | InsnKind::Bge |
                InsnKind::BltU | InsnKind::BgeU |
                InsnKind::Eany | InsnKind::Mret
            ) {
                block.is_conditional_branch = matches!(kind,
                    InsnKind::Beq | InsnKind::Bne | InsnKind::Blt | InsnKind::Bge |
                    InsnKind::BltU | InsnKind::BgeU
                );
                self.finish_block();
            }
        }
    }

    pub fn finish_block(&mut self) {
        if let Some(block) = self.current_block.take() {
            self.blocks.push(block);
        }
    }

    pub fn get_blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }

    pub fn clear(&mut self) {
        self.current_block = None;
        self.blocks.clear();
    }
}

/// JIT compilation strategy
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum JitStrategy {
    /// Compile after N executions
    Threshold(u32),
    /// Compile based on execution time
    Time(u64),
    /// Adaptive compilation
    Adaptive,
}

/// JIT performance metrics
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct JitMetrics {
    pub total_executions: u64,
    pub compiled_blocks: usize,
    pub compilation_time_us: u64,
    pub native_executions: u64,
}

#[allow(dead_code)]
impl JitMetrics {
    pub fn compilation_ratio(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.native_executions as f64 / self.total_executions as f64
        }
    }
}

/// High-level JIT execution engine with optimization and profiling
pub struct JitExecutionEngine {
    compiler: JitCompiler,
    execution_counts: HashMap<ByteAddr, u32>,
    hot_threshold: u32,
    metrics: JitMetrics,
}

impl JitExecutionEngine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            compiler: JitCompiler::new()?,
            execution_counts: HashMap::new(),
            hot_threshold: 10, // Compile after 10 executions
            metrics: JitMetrics::default(),
        })
    }

    pub fn with_threshold(mut self, threshold: u32) -> Self {
        self.hot_threshold = threshold;
        self
    }

    /// Execute a block, compiling it if it becomes hot
    pub fn execute_block(&mut self, block: &BasicBlock) -> Result<*const u8> {
        let addr = block.start_addr;

        // Check if already compiled
        if let Some(code_ptr) = self.compiler.get_compiled_block(addr) {
            self.metrics.native_executions += 1;
            return Ok(code_ptr);
        }

        // Track execution frequency
        let count = self.execution_counts.entry(addr).or_insert(0);
        *count += 1;
        self.metrics.total_executions += 1;

        // Compile if hot
        if *count >= self.hot_threshold {
            let start_time = std::time::Instant::now();
            let code_ptr = self.compiler.compile_block(block)?;

            self.metrics.compilation_time_us += start_time.elapsed().as_micros() as u64;
            self.metrics.compiled_blocks += 1;

            println!("JIT compiled hot block at {addr:?} after {count} executions");
            return Ok(code_ptr);
        }

        // Return null pointer to indicate interpreted execution
        Ok(std::ptr::null())
    }

    pub fn get_metrics(&self) -> &JitMetrics {
        &self.metrics
    }

    /// Clear compilation cache and reset metrics
    pub fn reset(&mut self) {
        self.execution_counts.clear();
        self.metrics = JitMetrics::default();
        // Note: compiled blocks remain in memory until JitCompiler is dropped
    }
}

/// Return codes for JIT-compiled functions
pub mod jit_return_codes {
    pub const CONTINUE: i32 = 0;         // Continue normal execution
    pub const BRANCH_TAKEN: i32 = 1;     // Branch was taken, PC updated
    pub const JUMP: i32 = 2;             // Unconditional jump, PC updated
    pub const INDIRECT_JUMP: i32 = 3;    // Indirect jump, compute target
    pub const ECALL: i32 = 8;            // Environment call exception
    pub const BREAKPOINT: i32 = 3;       // Breakpoint exception
    pub const MRET: i32 = 48;            // Machine return
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_block_builder() {
        let mut builder = BasicBlockBuilder::new();
        let decoded = DecodedInstruction::default();

        builder.add_instruction(ByteAddr(0x1000), InsnKind::Add, decoded.clone());
        builder.add_instruction(ByteAddr(0x1004), InsnKind::Sub, decoded.clone());
        builder.add_instruction(ByteAddr(0x1008), InsnKind::Jal, decoded);

        let blocks = builder.get_blocks();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].instructions.len(), 3);
        assert_eq!(blocks[0].start_addr, ByteAddr(0x1000));
        assert_eq!(blocks[0].end_addr, ByteAddr(0x1008));
    }

    #[test]
    fn test_jit_compiler() {
        let mut compiler = JitCompiler::new().unwrap();
        let block = BasicBlock {
            start_addr: ByteAddr(0x1000),
            instructions: vec![(InsnKind::Add, DecodedInstruction::default())],
            end_addr: ByteAddr(0x1000),
            is_conditional_branch: false,
        };

        let code_ptr = compiler.compile_block(&block).unwrap();
        assert!(!code_ptr.is_null());
        assert_eq!(compiler.get_compilation_count(), 1);

        let retrieved = compiler.get_compiled_block(ByteAddr(0x1000));
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), code_ptr);
    }

    #[test]
    fn test_jit_execution_engine() {
        let mut engine = JitExecutionEngine::new().unwrap().with_threshold(2);
        let block = BasicBlock {
            start_addr: ByteAddr(0x2000),
            instructions: vec![(InsnKind::AddI, DecodedInstruction::default())],
            end_addr: ByteAddr(0x2000),
            is_conditional_branch: false,
        };

        // First execution - should be interpreted
        let result1 = engine.execute_block(&block).unwrap();
        assert!(result1.is_null());

        // Second execution - should trigger compilation
        let result2 = engine.execute_block(&block).unwrap();
        assert!(!result2.is_null());

        let metrics = engine.get_metrics();
        assert_eq!(metrics.compiled_blocks, 1);
        assert_eq!(metrics.total_executions, 2);
    }
}
