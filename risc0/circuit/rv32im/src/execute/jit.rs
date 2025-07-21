#![allow(dead_code, unused_imports, unused_variables)]
use std::collections::HashMap;
use anyhow::Result;
use risc0_binfmt::{ByteAddr, WordAddr};

use super::rv32im::{DecodedInstruction, InsnKind, EmuContext};

/// A basic block of RISC-V instructions
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub start_addr: ByteAddr,
    pub instructions: Vec<(InsnKind, DecodedInstruction)>,
    pub end_addr: ByteAddr,
    pub is_conditional_branch: bool,
}

/// Simple x86-64 code generator for RISC-V instructions
pub struct X86CodeGen {
    code: Vec<u8>,
}

impl X86CodeGen {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
        }
    }

    /// Generate function prologue
    pub fn prologue(&mut self) {
        // Standard x86-64 function prologue
        // RDI contains the emulator context pointer
        self.code.extend_from_slice(&[0x55]); // push rbp
        self.code.extend_from_slice(&[0x48, 0x89, 0xe5]); // mov rbp, rsp
    }

    /// Generate function epilogue
    pub fn epilogue(&mut self) {
        // Standard x86-64 function epilogue
        self.code.extend_from_slice(&[0x5d]); // pop rbp
        self.code.extend_from_slice(&[0xc3]); // ret
    }

    /// Generate ADD instruction
    pub fn gen_add(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX
        self.gen_load_register_to_ecx(rs2);

        // ADD EAX, ECX
        self.code.extend_from_slice(&[0x01, 0xc8]); // add eax, ecx

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate ADDI instruction
    pub fn gen_addi(&mut self, rd: u32, rs1: u32, imm: i32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // ADD EAX, imm
        self.code.extend_from_slice(&[0x05]); // add eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SUB instruction
    pub fn gen_sub(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX
        self.gen_load_register_to_ecx(rs2);

        // SUB EAX, ECX
        self.code.extend_from_slice(&[0x29, 0xc8]); // sub eax, ecx

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate AND instruction
    pub fn gen_and(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX
        self.gen_load_register_to_ecx(rs2);

        // AND EAX, ECX
        self.code.extend_from_slice(&[0x21, 0xc8]); // and eax, ecx

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate OR instruction
    pub fn gen_or(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX
        self.gen_load_register_to_ecx(rs2);

        // OR EAX, ECX
        self.code.extend_from_slice(&[0x09, 0xc8]); // or eax, ecx

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate XOR instruction
    pub fn gen_xor(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX
        self.gen_load_register_to_ecx(rs2);

        // XOR EAX, ECX
        self.code.extend_from_slice(&[0x31, 0xc8]); // xor eax, ecx

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate LW instruction (load word)
    pub fn gen_lw(&mut self, rd: u32, rs1: u32, imm: i32) {
        // Load base address (rs1) into EAX
        self.gen_load_register(rs1);

        // Add immediate offset
        self.code.extend_from_slice(&[0x05]); // add eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // Load word from memory address in EAX
        self.code.extend_from_slice(&[0x8b, 0x00]); // mov eax, [eax]

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SW instruction (store word)
    pub fn gen_sw(&mut self, rs2: u32, rs1: u32, imm: i32) {
        // Load value to store (rs2) into EAX
        self.gen_load_register(rs2);

        // Push EAX to save the value
        self.code.push(0x50); // push eax

        // Load base address (rs1) into EAX
        self.gen_load_register(rs1);

        // Add immediate offset
        self.code.extend_from_slice(&[0x05]); // add eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // Move address to ECX
        self.code.extend_from_slice(&[0x89, 0xc1]); // mov ecx, eax

        // Pop value back to EAX
        self.code.push(0x58); // pop eax

        // Store word to memory address in ECX
        self.code.extend_from_slice(&[0x89, 0x01]); // mov [ecx], eax
    }

    /// Generate SLL instruction (logical left shift)
    pub fn gen_sll(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX (shift amount)
        self.gen_load_register_to_ecx(rs2);

        // SHL EAX, CL
        self.code.extend_from_slice(&[0xd3, 0xe0]); // shl eax, cl

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SLLI instruction (logical left shift immediate)
    pub fn gen_slli(&mut self, rd: u32, rs1: u32, shamt: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // SHL EAX, shamt
        self.code.extend_from_slice(&[0xc1, 0xe0]); // shl eax, shamt
        self.code.push(shamt as u8);

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SRL instruction (logical right shift)
    pub fn gen_srl(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX (shift amount)
        self.gen_load_register_to_ecx(rs2);

        // SHR EAX, CL
        self.code.extend_from_slice(&[0xd3, 0xe8]); // shr eax, cl

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SRLI instruction (logical right shift immediate)
    pub fn gen_srli(&mut self, rd: u32, rs1: u32, shamt: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // SHR EAX, shamt
        self.code.extend_from_slice(&[0xc1, 0xe8]); // shr eax, shamt
        self.code.push(shamt as u8);

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SRA instruction (arithmetic right shift)
    pub fn gen_sra(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX (shift amount)
        self.gen_load_register_to_ecx(rs2);

        // SAR EAX, CL
        self.code.extend_from_slice(&[0xd3, 0xf8]); // sar eax, cl

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SRAI instruction (arithmetic right shift immediate)
    pub fn gen_srai(&mut self, rd: u32, rs1: u32, shamt: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // SAR EAX, shamt
        self.code.extend_from_slice(&[0xc1, 0xf8]); // sar eax, shamt
        self.code.push(shamt as u8);

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SLT instruction (set if less than)
    pub fn gen_slt(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX
        self.gen_load_register_to_ecx(rs2);

        // CMP EAX, ECX
        self.code.extend_from_slice(&[0x39, 0xc8]); // cmp eax, ecx

        // SETL AL (set if less than)
        self.code.extend_from_slice(&[0x0f, 0x9c, 0xc0]); // setl al

        // Zero-extend AL to EAX
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SLTU instruction (set if less than unsigned)
    pub fn gen_sltu(&mut self, rd: u32, rs1: u32, rs2: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX
        self.gen_load_register_to_ecx(rs2);

        // CMP EAX, ECX
        self.code.extend_from_slice(&[0x39, 0xc8]); // cmp eax, ecx

        // SETB AL (set if below/unsigned less than)
        self.code.extend_from_slice(&[0x0f, 0x92, 0xc0]); // setb al

        // Zero-extend AL to EAX
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SLTI instruction (set if less than immediate)
    pub fn gen_slti(&mut self, rd: u32, rs1: u32, imm: i32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // CMP EAX, imm
        self.code.extend_from_slice(&[0x3d]); // cmp eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // SETL AL (set if less than)
        self.code.extend_from_slice(&[0x0f, 0x9c, 0xc0]); // setl al

        // Zero-extend AL to EAX
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate SLTIU instruction (set if less than immediate unsigned)
    pub fn gen_sltiu(&mut self, rd: u32, rs1: u32, imm: i32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // CMP EAX, imm
        self.code.extend_from_slice(&[0x3d]); // cmp eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // SETB AL (set if below/unsigned less than)
        self.code.extend_from_slice(&[0x0f, 0x92, 0xc0]); // setb al

        // Zero-extend AL to EAX
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate XORI instruction
    pub fn gen_xori(&mut self, rd: u32, rs1: u32, imm: i32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // XOR EAX, imm
        self.code.extend_from_slice(&[0x35]); // xor eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate ORI instruction
    pub fn gen_ori(&mut self, rd: u32, rs1: u32, imm: i32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // OR EAX, imm
        self.code.extend_from_slice(&[0x0d]); // or eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate ANDI instruction
    pub fn gen_andi(&mut self, rd: u32, rs1: u32, imm: i32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // AND EAX, imm
        self.code.extend_from_slice(&[0x25]); // and eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate LUI instruction (load upper immediate)
    pub fn gen_lui(&mut self, rd: u32, imm: u32) {
        // Load immediate into EAX
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate AUIPC instruction (add upper immediate to PC)
    pub fn gen_auipc(&mut self, rd: u32, imm: u32, current_pc: u32) {
        // Load PC + immediate into EAX
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        let pc_plus_imm = current_pc + imm;
        self.code.extend_from_slice(&pc_plus_imm.to_le_bytes());

        // Store result to rd
        self.gen_store_register(rd);
    }

    /// Generate BEQ instruction (branch if equal)
    pub fn gen_beq(&mut self, rs1: u32, rs2: u32, imm: i32, current_pc: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX
        self.gen_load_register_to_ecx(rs2);

        // CMP EAX, ECX
        self.code.extend_from_slice(&[0x39, 0xc8]); // cmp eax, ecx

        // JNE to skip branch (if not equal, don't branch)
        let skip_offset = 8; // Size of the jump instruction
        self.code.extend_from_slice(&[0x75, skip_offset as u8]); // jne +8

        // If equal, return branch target
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        let target = (current_pc as i32 + imm) as u32;
        self.code.extend_from_slice(&target.to_le_bytes());

        // Return the target PC
        self.code.extend_from_slice(&[0xc3]); // ret
    }

    /// Generate BNE instruction (branch if not equal)
    pub fn gen_bne(&mut self, rs1: u32, rs2: u32, imm: i32, current_pc: u32) {
        // Load rs1 into EAX
        self.gen_load_register(rs1);

        // Load rs2 into ECX
        self.gen_load_register_to_ecx(rs2);

        // CMP EAX, ECX
        self.code.extend_from_slice(&[0x39, 0xc8]); // cmp eax, ecx

        // JE to skip branch (if equal, don't branch)
        let skip_offset = 8; // Size of the jump instruction
        self.code.extend_from_slice(&[0x74, skip_offset as u8]); // je +8

        // If not equal, return branch target
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        let target = (current_pc as i32 + imm) as u32;
        self.code.extend_from_slice(&target.to_le_bytes());

        // Return the target PC
        self.code.extend_from_slice(&[0xc3]); // ret
    }

    /// Generate JAL instruction (jump and link)
    pub fn gen_jal(&mut self, rd: u32, imm: i32, current_pc: u32) {
        // Save return address to rd (if rd != 0)
        if rd != 0 {
            self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
            let return_addr = current_pc + 4;
            self.code.extend_from_slice(&return_addr.to_le_bytes());
            self.gen_store_register(rd);
        }

        // Return branch target
        self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
        let target = (current_pc as i32 + imm) as u32;
        self.code.extend_from_slice(&target.to_le_bytes());

        // Return the target PC
        self.code.extend_from_slice(&[0xc3]); // ret
    }

    /// Generate JALR instruction (jump and link register)
    pub fn gen_jalr(&mut self, rd: u32, rs1: u32, imm: i32, current_pc: u32) {
        // Save return address to rd (if rd != 0)
        if rd != 0 {
            self.code.extend_from_slice(&[0xb8]); // mov eax, imm32
            let return_addr = current_pc + 4;
            self.code.extend_from_slice(&return_addr.to_le_bytes());
            self.gen_store_register(rd);
        }

        // Load base address (rs1) into EAX
        self.gen_load_register(rs1);

        // Add immediate offset
        self.code.extend_from_slice(&[0x05]); // add eax, imm32
        self.code.extend_from_slice(&imm.to_le_bytes());

        // Clear least significant bit (JALR requirement)
        self.code.extend_from_slice(&[0x83, 0xe0, 0xfe]); // and eax, -2

        // Return the target PC
        self.code.extend_from_slice(&[0xc3]); // ret
    }

    /// Generate ECALL instruction
    pub fn gen_ecall(&mut self) {
        // Return ECALL code
        self.code.extend_from_slice(&[0xb8, 0x08, 0x00, 0x00, 0x00]); // mov eax, 8
        self.code.extend_from_slice(&[0xc3]); // ret
    }

    /// Generate EBREAK instruction
    pub fn gen_ebreak(&mut self) {
        // Return EBREAK code
        self.code.extend_from_slice(&[0xb8, 0x03, 0x00, 0x00, 0x00]); // mov eax, 3
        self.code.extend_from_slice(&[0xc3]); // ret
    }

    /// Generate MRET instruction
    pub fn gen_mret(&mut self) {
        // Return MRET code
        self.code.extend_from_slice(&[0xb8, 0x30, 0x00, 0x00, 0x00]); // mov eax, 48
        self.code.extend_from_slice(&[0xc3]); // ret
    }

    /// Load register value into EAX
    fn gen_load_register(&mut self, reg: u32) {
        if reg == 0 {
            // x0 is always zero
            self.code.extend_from_slice(&[0x31, 0xc0]); // xor eax, eax
        } else {
            // Load from emulator context: MOV EAX, [RDI + reg*4]
            let offset = reg * 4;
            if offset < 128 {
                self.code.extend_from_slice(&[0x8b, 0x47]); // mov eax, [rdi + disp8]
                self.code.push(offset as u8);
            } else {
                self.code.extend_from_slice(&[0x8b, 0x87]); // mov eax, [rdi + disp32]
                self.code.extend_from_slice(&offset.to_le_bytes());
            }
        }
    }

    /// Load register value into ECX
    fn gen_load_register_to_ecx(&mut self, reg: u32) {
        if reg == 0 {
            // x0 is always zero
            self.code.extend_from_slice(&[0x31, 0xc9]); // xor ecx, ecx
        } else {
            // Load from emulator context: MOV ECX, [RDI + reg*4]
            let offset = reg * 4;
            if offset < 128 {
                self.code.extend_from_slice(&[0x8b, 0x4f]); // mov ecx, [rdi + disp8]
                self.code.push(offset as u8);
            } else {
                self.code.extend_from_slice(&[0x8b, 0x8f]); // mov ecx, [rdi + disp32]
                self.code.extend_from_slice(&offset.to_le_bytes());
            }
        }
    }

    /// Store EAX value to register
    fn gen_store_register(&mut self, reg: u32) {
        if reg == 0 {
            // x0 is hardwired to zero - ignore writes
            self.code.push(0x90); // nop
        } else {
            // Store to emulator context: MOV [RDI + reg*4], EAX
            let offset = reg * 4;
            if offset < 128 {
                self.code.extend_from_slice(&[0x89, 0x47]); // mov [rdi + disp8], eax
                self.code.push(offset as u8);
            } else {
                self.code.extend_from_slice(&[0x89, 0x87]); // mov [rdi + disp32], eax
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
}

/// Simple JIT compiler
pub struct JitCompiler {
    pub compiled_blocks: HashMap<ByteAddr, *const u8>,
    compilation_count: usize,
}

impl Drop for JitCompiler {
    fn drop(&mut self) {
        // Clean up allocated executable memory
        #[cfg(unix)]
        {
            for &code_ptr in self.compiled_blocks.values() {
                if !code_ptr.is_null() {
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
                InsnKind::Lw => {
                    codegen.gen_lw(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::Sw => {
                    codegen.gen_sw(decoded.rs2, decoded.rs1, decoded.imm_s() as i32);
                }
                InsnKind::Sll => {
                    codegen.gen_sll(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::SllI => {
                    let shamt = decoded.imm_i() & 0x1f;
                    codegen.gen_slli(decoded.rd, decoded.rs1, shamt);
                }
                InsnKind::Srl => {
                    codegen.gen_srl(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::SrlI => {
                    let shamt = decoded.imm_i() & 0x1f;
                    codegen.gen_srli(decoded.rd, decoded.rs1, shamt);
                }
                InsnKind::Sra => {
                    codegen.gen_sra(decoded.rd, decoded.rs1, decoded.rs2);
                }
                InsnKind::SraI => {
                    let shamt = decoded.imm_i() & 0x1f;
                    codegen.gen_srai(decoded.rd, decoded.rs1, shamt);
                }
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
                InsnKind::XorI => {
                    codegen.gen_xori(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::OrI => {
                    codegen.gen_ori(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::AndI => {
                    codegen.gen_andi(decoded.rd, decoded.rs1, decoded.imm_i() as i32);
                }
                InsnKind::Lui => {
                    // Calculate U-type immediate manually: upper 20 bits
                    let imm_u = decoded.insn & 0xfffff000;
                    codegen.gen_lui(decoded.rd, imm_u);
                }
                InsnKind::Auipc => {
                    // Calculate U-type immediate manually: upper 20 bits
                    let imm_u = decoded.insn & 0xfffff000;
                    codegen.gen_auipc(decoded.rd, imm_u, block.start_addr.0);
                }
                InsnKind::Beq => {
                    codegen.gen_beq(decoded.rs1, decoded.rs2, decoded.imm_b() as i32, block.start_addr.0);
                }
                InsnKind::Bne => {
                    codegen.gen_bne(decoded.rs1, decoded.rs2, decoded.imm_b() as i32, block.start_addr.0);
                }
                InsnKind::Jal => {
                    // Calculate J-type immediate manually
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
                InsnKind::Eany => {
                    codegen.gen_ecall();
                }
                InsnKind::Mret => {
                    codegen.gen_mret();
                }
                _ => {
                    // For unsupported instructions, just continue
                    continue;
                }
            }
        }

        // Generate function epilogue
        codegen.epilogue();

        // Allocate executable memory and copy the code
        let code = codegen.get_code();
        let code_ptr = self.allocate_executable_memory(code.len())?;

        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), code_ptr as *mut u8, code.len());
        }

        // Cache the compiled block
        self.compiled_blocks.insert(block.start_addr, code_ptr);

        Ok(code_ptr)
    }

    /// Allocate executable memory
    fn allocate_executable_memory(&self, size: usize) -> Result<*const u8> {
        #[cfg(unix)]
        {
            use libc::{mmap, MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

            let aligned_size = (size + 4095) & !4095; // Round up to page size

            let addr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    aligned_size,
                    PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };

            if addr == MAP_FAILED {
                return Err(anyhow::anyhow!("Failed to allocate executable memory"));
            }

            Ok(addr as *const u8)
        }

        #[cfg(not(unix))]
        {
            Err(anyhow::anyhow!("Executable memory allocation not supported on this platform"))
        }
    }

    /// Get a compiled block for the given address
    pub fn get_compiled_block(&self, addr: ByteAddr) -> Option<*const u8> {
        self.compiled_blocks.get(&addr).copied()
    }

    /// Get compilation count
    pub fn get_compilation_count(&self) -> usize {
        self.compilation_count
    }

    /// Clear the compilation cache
    pub fn clear_cache(&mut self) {
        self.compiled_blocks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_compiler_creation() {
        let compiler = JitCompiler::new().unwrap();
        assert_eq!(compiler.compiled_blocks.len(), 0);
        assert_eq!(compiler.compilation_count, 0);
    }

    #[test]
    fn test_basic_block() {
        let block = BasicBlock {
            start_addr: ByteAddr(0x1000),
            instructions: vec![],
            end_addr: ByteAddr(0x1000),
            is_conditional_branch: false,
        };
        assert_eq!(block.start_addr.0, 0x1000);
    }

    #[test]
    fn test_x86_codegen() {
        let mut codegen = X86CodeGen::new();
        codegen.prologue();
        codegen.gen_add(1, 2, 3);
        codegen.epilogue();
        assert!(codegen.get_code_size() > 0);
    }
}
