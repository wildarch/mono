use std::arch::global_asm;

use super::{Cell, CellPtr, TigreEngine, CALL_LEN, CALL_OPCODE};

#[macro_use]
mod macros;

pub const ALL_COMBINATORS: &[unsafe extern "C" fn() -> usize] = &[
    comb_LIT, comb_Abort, comb_I, comb_K, comb_S, comb_B, comb_C, comb_plus, comb_min, comb_eq,
    comb_lt, comb_times, comb_cond,
];

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

macros::comb3!(comb_S, "comb_S", make_s, "make_s", |engine, f, g, x| {
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
});

macros::comb3!(comb_B, "comb_B", make_b, "make_b", |engine, f, g, x| {
    // Make new cell for (g x)
    let gx = engine.make_cell(*g, *x);
    // Pointers on the stack are to the right pointer of a cell, after the call instruction.
    // To get a pointer to the full cell, we subtract the length of a call instruction.
    let top_cell_ptr = CellPtr((x as *mut u8).offset(-CALL_LEN) as *mut Cell);
    let top_cell = engine.cell_mut(top_cell_ptr);
    debug_assert_eq!(top_cell.call_opcode, CALL_OPCODE);
    top_cell.set_call_addr((*f).0 as usize);
    top_cell.arg = gx.0 as i64;

    top_cell_ptr
});

macros::comb3!(comb_C, "comb_C", make_c, "make_c", |engine, f, g, x| {
    // Make new cell for (f x)
    let fx = engine.make_cell(*f, *x);
    // Pointers on the stack are to the right pointer of a cell, after the call instruction.
    // To get a pointer to the full cell, we subtract the length of a call instruction.
    let top_cell_ptr = CellPtr((x as *mut u8).offset(-CALL_LEN) as *mut Cell);
    let top_cell = engine.cell_mut(top_cell_ptr);
    debug_assert_eq!(top_cell.call_opcode, CALL_OPCODE);
    top_cell.set_call_addr(fx.0 as usize);
    top_cell.arg = (*g).0 as i64;

    top_cell_ptr
});

// A pointer to a function that evaluates an argument to a strict operator.
type ArgFn = unsafe extern "C" fn() -> i64;
macros::comb_bin_op!(comb_plus, "comb_plus", apply_plus, "apply_plus", |a, b| a
    + b);
macros::comb_bin_op!(comb_min, "comb_min", apply_min, "apply_min", |a, b| a - b);
macros::comb_bin_op!(comb_eq, "comb_eq", apply_eq, "apply_eq", |a, b| if a == b {
    1
} else {
    0
});
macros::comb_bin_op!(comb_lt, "comb_lt", apply_lt, "apply_lt", |a, b| if a < b {
    1
} else {
    0
});
macros::comb_bin_op!(
    comb_times,
    "comb_times",
    apply_times,
    "apply_times",
    |a, b| a * b
);

macros::comb3!(
    comb_cond,
    "comb_cond",
    apply_cond,
    "apply_cond",
    |engine, c, t, f| {
        // Evaluate the c cell as a strict argument
        let c = c as *const ArgFn;
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

        // Update the top cell with an indirection to the taken branch.
        // See `make_s` for details.
        let top_cell_ptr = CellPtr((f as *mut u8).offset(-CALL_LEN) as *mut Cell);
        let top_cell = engine.cell_mut(top_cell_ptr);
        debug_assert_eq!(top_cell.call_opcode, CALL_OPCODE);
        top_cell.set_call_addr(comb_I as usize);
        top_cell.arg = (*branch_ptr).0 as i64;

        *branch_ptr
    }
);

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
