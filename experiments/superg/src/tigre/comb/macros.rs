// Generate an assembly target and helper function for a combinator that takes three arguments.
macro_rules! comb3 {
    ($name:ident, $helper_func:ident, $impl:expr) => {
        global_asm!(
            concat!(".global ", stringify!($name)),
            concat!(stringify!($name), ":"),
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
            concat!("jne ", stringify!($name), "_need_align"),
            // No alignment needed
            concat!(stringify!($name), "_no_align:"),
            concat!("call ", stringify!($helper_func)),
            r#"
                // Pop arguments
                add rsp, 24
                jmp rax
            "#,
            // Alignment needed
            concat!(stringify!($name), "_need_align:"),
            "sub rsp, 8",
            concat!("call ", stringify!($helper_func)),
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
    ($name:ident, $helper_func:ident, $op:expr) => {
        global_asm!(
            concat!(".global ", stringify!($name)),
            concat!(stringify!($name), ":"),
            r#"
                // Load argument pointers as arguments
                mov rdi, [rsp]
                mov rsi, [rsp+8]

                // Align stack to 16 bytes if needed
                mov rax, rsp
                and rax, 8
                cmp rax, 0
            "#,
            concat!("jne ", stringify!($name), "_need_align"),
            // No alignment needed
            concat!(stringify!($name), "_no_align:"),
            concat!("call ", stringify!($helper_func)),
            r#"
                // Pop arguments
                add rsp, 16
                // Return the computed value
                ret
            "#,
            // Alignment needed
            concat!(stringify!($name), "_need_align:"),
            "sub rsp, 8",
            concat!("call ", stringify!($helper_func)),
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

macro_rules! bulk_comb {
    ($name:ident, $arg_count:literal, $stack_off_no_align:literal, $stack_off_need_align:literal, $helper_func:ident, $impl:expr) => {
        global_asm!(
            concat!(".global ", stringify!($name)),
            concat!(stringify!($name), ":"),
            r#"
                // Load pointer to stacked arguments
                mov rdi, rsp

                // Align stack to 16 bytes if needed
                mov rax, rsp
                and rax, 8
                cmp rax, 0
            "#,
            concat!("jne ", stringify!($name), "_need_align"),
            // No alignment needed
            concat!(stringify!($name), "_no_align:"),
            concat!("call ", stringify!($helper_func)),
            // Pop arguments
            concat!("add rsp, ", stringify!($stack_off_no_align)),
            "jmp rax",
            // Alignment needed
            concat!(stringify!($name), "_need_align:"),
            "sub rsp, 8",
            concat!("call ", stringify!($helper_func)),
            // Pop arguments
            concat!("add rsp, ", stringify!($stack_off_need_align)),
            "jmp rax",
        );
        extern "C" {
            pub fn $name() -> usize;
        }
        #[no_mangle]
        unsafe extern "C" fn $helper_func(args: *const *const CellPtr) -> CellPtr {
            // Place all code within `TigreEngine::with_current` since that also takes
            // care of catching unwinds before they escape Rust code.
            TigreEngine::with_current(|engine| {
                let fimpl: fn(&mut TigreEngine, &[&CellPtr]) -> CellPtr = $impl;
                debug_assert_eq!($arg_count * 8, $stack_off_no_align);
                debug_assert_eq!($arg_count * 8 + 8, $stack_off_need_align);
                let args = args as *const &CellPtr;
                let args = std::slice::from_raw_parts(args, $arg_count);
                fimpl(engine, args)
            })
        }
    };
}

macro_rules! bulk_comb_s {
    ($name:ident, $arg_count:literal, $stack_off_no_align:literal, $stack_off_need_align:literal, $helper_func:ident) => {
        bulk_comb!(
            $name,
            $arg_count,
            $stack_off_no_align,
            $stack_off_need_align,
            $helper_func,
            |engine, args| {
                let f = *args[0];
                let g = *args[1];

                let mut left_cell = f;
                let mut right_cell = g;
                for arg in &args[2..] {
                    left_cell = engine.make_cell(left_cell, **arg);
                    right_cell = engine.make_cell(right_cell, **arg);
                }

                // Update the top cell to point to the new left_cell and right_cell
                // See `make_s` for details.
                let top_arg_ptr = *args.last().unwrap() as *const CellPtr;
                let top_cell_ptr = CellPtr((top_arg_ptr as *mut u8).offset(-CALL_LEN) as *mut Cell);
                let top_cell = engine.cell_mut(top_cell_ptr);
                debug_assert_eq!(top_cell.call_opcode, CALL_OPCODE);
                top_cell.set_call_addr(left_cell.0 as usize);
                top_cell.arg = right_cell.0 as i64;

                top_cell_ptr
            }
        );
    };
}
pub(crate) use bulk_comb_s;

macro_rules! bulk_comb_b {
    ($name:ident, $arg_count:literal, $stack_off_no_align:literal, $stack_off_need_align:literal, $helper_func:ident) => {
        bulk_comb!(
            $name,
            $arg_count,
            $stack_off_no_align,
            $stack_off_need_align,
            $helper_func,
            |engine, args| {
                let f = *args[0];
                let g = *args[1];

                let mut right_cell = g;
                for arg in &args[2..] {
                    right_cell = engine.make_cell(right_cell, **arg);
                }

                // Update the top cell to point to the new left_cell and right_cell
                // See `make_s` for details.
                let top_arg_ptr = *args.last().unwrap() as *const CellPtr;
                let top_cell_ptr = CellPtr((top_arg_ptr as *mut u8).offset(-CALL_LEN) as *mut Cell);
                let top_cell = engine.cell_mut(top_cell_ptr);
                debug_assert_eq!(top_cell.call_opcode, CALL_OPCODE);
                top_cell.set_call_addr(f.0 as usize);
                top_cell.arg = right_cell.0 as i64;

                top_cell_ptr
            }
        );
    };
}
pub(crate) use bulk_comb_b;

macro_rules! bulk_comb_c {
    ($name:ident, $arg_count:literal, $stack_off_no_align:literal, $stack_off_need_align:literal, $helper_func:ident) => {
        bulk_comb!(
            $name,
            $arg_count,
            $stack_off_no_align,
            $stack_off_need_align,
            $helper_func,
            |engine, args| {
                let f = *args[0];
                let g = *args[1];

                let mut left_cell = f;
                for arg in &args[2..] {
                    left_cell = engine.make_cell(left_cell, **arg);
                }

                // Update the top cell to point to the new left_cell and right_cell
                // See `make_s` for details.
                let top_arg_ptr = *args.last().unwrap() as *const CellPtr;
                let top_cell_ptr = CellPtr((top_arg_ptr as *mut u8).offset(-CALL_LEN) as *mut Cell);
                let top_cell = engine.cell_mut(top_cell_ptr);
                debug_assert_eq!(top_cell.call_opcode, CALL_OPCODE);
                top_cell.set_call_addr(left_cell.0 as usize);
                top_cell.arg = g.0 as i64;

                top_cell_ptr
            }
        );
    };
}
pub(crate) use bulk_comb_c;
