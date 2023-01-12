// Generate an assembly target and helper function for a combinator that takes three arguments.
macro_rules! comb3 {
    ($name:ident, $name_str:literal, $helper_func:ident, $helper_func_str:literal, $impl:expr) => {
        global_asm!(
            concat!(".global ", $name_str),
            concat!($name_str, ":"),
            r#"
                // Load f, g, x pointers as arguments
                mov rdi, [rsp]
                mov rsi, [rsp+8]
                mov rdx, [rsp+16]

                // Align stack to 16 bytes if needed
                mov rax, rsp
                and rax, 8
                cmp rax, 0
            "#,
            concat!("jne ", $name_str, "_need_align"),
            // No alignment needed
            concat!($name_str, "_no_align:"),
            concat!("call ", $helper_func_str),
            r#"
                // Pop arguments
                add rsp, 24
                jmp rax
            "#,
            // Alignment needed
            concat!($name_str, "_need_align:"),
            "sub rsp, 8",
            concat!("call ", $helper_func_str),
            r#"
                // Pop arguments
                add rsp, 32
                jmp rax
            "#,
        );
        extern "C" {
            pub fn $name() -> usize;
        }
        #[no_mangle]
        unsafe extern "C" fn $helper_func(
            f: *const CellPtr,
            g: *const CellPtr,
            x: *const CellPtr,
        ) -> CellPtr {
            // Place all code within `TigreEngine::with_current` since that also takes
            // care of catching unwinds before they escape Rust code.
            TigreEngine::with_current(|engine| {
                let fimpl: fn(
                    &mut TigreEngine,
                    *const CellPtr,
                    *const CellPtr,
                    *const CellPtr,
                ) -> CellPtr = $impl;
                fimpl(engine, f, g, x)
            })
        }
    };
}
pub(crate) use comb3;

// Generate an assembly target and helper function for a binary operator.
macro_rules! comb_bin_op {
    ($name:ident, $name_str:literal, $helper_func:ident, $helper_func_str:literal, $op:expr) => {
        global_asm!(
            concat!(".global ", $name_str),
            concat!($name_str, ":"),
            r#"
                // Load argument pointers as arguments
                mov rdi, [rsp]
                mov rsi, [rsp+8]

                // Align stack to 16 bytes if needed
                mov rax, rsp
                and rax, 8
                cmp rax, 0
            "#,
            concat!("jne ", $name_str, "_need_align"),
            // No alignment needed
            concat!($name_str, "_no_align:"),
            concat!("call ", $helper_func_str),
            r#"
                // Pop arguments
                add rsp, 16
                // Return the computed value
                ret
            "#,
            // Alignment needed
            concat!($name_str, "_need_align:"),
            "sub rsp, 8",
            concat!("call ", $helper_func_str),
            r#"
                // Pop arguments
                add rsp, 24
                // Return the computed value
                ret
            "#,
        );
        extern "C" {
            pub fn $name() -> usize;
        }
        #[no_mangle]
        unsafe extern "C" fn $helper_func(a0: *const ArgFn, a1: *const ArgFn) -> i64 {
            // Place all code within `TigreEngine::with_current` since that also takes
            // care of catching unwinds before they escape Rust code.
            TigreEngine::with_current(|engine| {
                let op: fn(i64, i64) -> i64 = $op;
                let res = op((*a0)(), (*a1)());

                // Update the top cell with the new number.
                // See `make_s` for details.
                let top_cell_ptr = CellPtr((a1 as *mut u8).offset(-CALL_LEN) as *mut Cell);
                let top_cell = engine.cell_mut(top_cell_ptr);
                debug_assert_eq!(top_cell.call_opcode, CALL_OPCODE);
                top_cell.set_call_addr(comb_LIT as usize);
                top_cell.arg = res;

                res
            })
        }
    };
}
pub(crate) use comb_bin_op;
