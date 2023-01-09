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

// Abort combinator
#[allow(non_snake_case)]
pub extern "C" fn comb_Abort() -> usize {
    panic!("Abort called")
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

// Plus combinator
global_asm!(
    r#"
    .global comb_plus
    comb_plus:
        // Load argument pointers as arguments
        mov rdi, [rsp]
        mov rsi, [rsp+8]
        call apply_plus
        // Pop arguments
        add rsp, 16
        // Return the computed value
        ret
"#
);
extern "C" {
    pub fn comb_plus() -> usize;
}
// A pointer to a function that evaluates an argument to a strict operator.
type ArgFn = unsafe extern "C" fn() -> i64;
#[no_mangle]
unsafe extern "C" fn apply_plus(a0: *const ArgFn, a1: *const ArgFn) -> i64 {
    let res = (*a0)() + (*a1)();

    TigreEngine::with_current(|engine| {
        // Update the top cell with the new number.
        // See `make_s` for details.
        let top_cell_ptr = CellPtr((a1 as *mut u8).offset(-CALL_LEN) as *mut Cell);
        let top_cell = engine.cell_mut(top_cell_ptr);
        debug_assert_eq!(top_cell.call_opcode, CALL_OPCODE);
        top_cell.set_call_addr(comb_LIT as usize);
        top_cell.arg = res;
    });

    res
}

// Cond combinator
global_asm!(
    r#"
    .global comb_cond
    comb_cond:
        // Load c, t, f pointers as arguments to apply_cond
        mov rdi, [rsp]
        mov rsi, [rsp+8]
        mov rdx, [rsp+16]
        call apply_cond
        // Pop arguments and jump to the updated node 
        add rsp, 24
        jmp rax
"#
);
extern "C" {
    pub fn comb_cond() -> usize;
}
#[no_mangle]
unsafe extern "C" fn apply_cond(c: *const ArgFn, t: *const CellPtr, f: *const CellPtr) -> CellPtr {
    // TODO: avoid loading pointers to both branches in the assembly
    let c_res = (*c)();
    let branch_ptr = match c_res {
        0 => f,
        1 => t,
        v => {
            // We expect never to reach this code.
            // In debug mode we panic, in release the behaviour is undefined.
            debug_assert!(v == 0 || v == 1, "condition variable should be 0 or 1");
            unsafe { std::hint::unreachable_unchecked() }
        }
    };

    TigreEngine::with_current(|engine| {
        // Update the top cell with an indirection to the taken branch.
        // See `make_s` for details.
        let top_cell_ptr = CellPtr((f as *mut u8).offset(-CALL_LEN) as *mut Cell);
        let top_cell = engine.cell_mut(top_cell_ptr);
        debug_assert_eq!(top_cell.call_opcode, CALL_OPCODE);
        top_cell.set_call_addr(comb_I as usize);
        top_cell.arg = (*branch_ptr).0 as i64;
    });
    *branch_ptr
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
