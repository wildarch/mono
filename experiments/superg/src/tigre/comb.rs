use std::arch::global_asm;

global_asm!(
    r#"
    .global comb_LIT
    comb_LIT:
        pop rax
        ret
    .global comb_LIT_end
    comb_LIT_end:
"#
);
extern "C" {
    pub fn comb_LIT() -> usize;
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
            // Emulates a cell with lhs LIT and rhs 42
            push 42
            jmp comb_LIT
    "#
    );
    extern "C" {
        fn lit_dispatch_helper() -> u64;
    }
}
