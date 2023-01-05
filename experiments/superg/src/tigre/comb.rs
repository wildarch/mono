use std::arch::global_asm;

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

global_asm!(
    r#"
    .global comb_I
    comb_I:
        mov rcx, [rsp]
        add rsp, 8
        jmp [rcx]
"#
);
extern "C" {
    pub fn comb_I() -> usize;
}

global_asm!(
    r#"
    .global comb_Abort
    comb_Abort:
        jmp comb_Abort
"#
);
extern "C" {
    pub fn comb_Abort() -> usize;
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
