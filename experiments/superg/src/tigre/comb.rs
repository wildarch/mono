use std::arch::global_asm;

use super::{Cell, CellPtr, TigreEngine, CALL_LEN, CALL_OPCODE};

// LIT combinator
global_asm!(
    r#"
    .global comb_LIT
    comb_LIT:
        pop rax
        mov rax, [rax]
        ret
"#
);
extern "C" {
    pub fn comb_LIT() -> usize;
}

// I combinator
global_asm!(
    r#"
    .global comb_I
    comb_I:
        pop rax
        jmp [rax]
"#
);
extern "C" {
    pub fn comb_I() -> usize;
}

// K combinator
global_asm!(
    r#"
    .global comb_K
    comb_K:
        mov rax, [rsp]
        add rsp, 16
        jmp [rax]
"#
);
extern "C" {
    pub fn comb_K() -> usize;
}

// S combinator
global_asm!(
    r#"
    .global comb_S
    comb_S:
        // Load f, g, x pointers as arguments to make_s 
        mov rdi, [rsp]
        mov rsi, [rsp+8]
        mov rdx, [rsp+16]
        call make_s
        // Pop arguments and jump to the updated node 
        add rsp, 24
        jmp rax
    "#
);
extern "C" {
    pub fn comb_S() -> usize;
}
#[no_mangle]
unsafe extern "C" fn make_s(f: *const CellPtr, g: *const CellPtr, x: *const CellPtr) -> CellPtr {
    TigreEngine::with_current(|engine| {
        // Make new cells for (f x), (g x)
        let fx = engine.make_cell(*f, *x);
        let gx = engine.make_cell(*g, *x);
        // Pointers on the stack are to the right pointer of a cell, after the call instruction.
        // To get a pointer to the full cell, we subtract the length of a call instruction.
        let top_cell_ptr = CellPtr((x as *mut u8).offset(-CALL_LEN) as *mut Cell);
        let top_cell = engine.cell_mut(top_cell_ptr);
        debug_assert_eq!(top_cell.call_opcode, CALL_OPCODE);
        top_cell.set_call_addr(fx.0 as usize);
        top_cell.arg = gx.0 as i64;
        top_cell_ptr
    })
}

#[allow(non_snake_case)]
pub extern "C" fn comb_Abort() -> usize {
    panic!("Abort called")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lit_dispatch() {
        let res = unsafe { lit_dispatch_helper() };
        assert_eq!(res, 42);
    }

    global_asm!(
        r#"
        .global lit_dispatch_helper
        lit_dispatch_helper:
        call comb_LIT
        .long 42
        .long 0
    "#
    );
    extern "C" {
        fn lit_dispatch_helper() -> u64;
    }
}
