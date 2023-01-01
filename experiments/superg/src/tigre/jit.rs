pub fn assemble_call(off: i32) -> [u8; 5] {
    let mut buf = [0u8; 5];
    // CALL rel32
    buf[0] = 0xE8;
    buf[1..5].copy_from_slice(&off.to_le_bytes());
    buf
}

pub struct JitMem {
    buf: *mut u8,
    len: usize,
}

impl JitMem {
    pub fn new(len: usize) -> JitMem {
        // Allow RWX, everything!
        let prot = libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;
        let flags = libc::MAP_ANONYMOUS | libc::MAP_PRIVATE;
        // Unused because the mmap is anonymous
        let fd = -1;
        let offset = 0;
        let buf =
            unsafe { libc::mmap(std::ptr::null_mut(), len, prot, flags, fd, offset) as *mut u8 };
        if buf == libc::MAP_FAILED as *mut u8 {
            panic!("mmap failed: {}", std::io::Error::last_os_error());
        }

        JitMem { buf, len }
    }

    pub fn slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buf, self.len) }
    }
}

impl core::ops::Drop for JitMem {
    fn drop(&mut self) {
        let res = unsafe { libc::munmap(self.buf as *mut libc::c_void, self.len) };
        if res != 0 {
            panic!(
                "munmap({:?}, {}) failed: {}",
                self.buf,
                self.len,
                std::io::Error::last_os_error()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_exec() {
        let code = &[
            0x55, //    push   %rbp
            0x48, 0x89, 0xe5, //    mov    %rsp,%rbp
            0xb8, 0x37, 0x00, 0x00, 0x00, //    mov    $0x37,%eax
            0xc9, //    leaveq
            0xc3, //    retq
        ];

        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                code.len(),
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            );

            std::ptr::copy(code.as_ptr(), ptr as *mut u8, code.len());

            let func: unsafe extern "C" fn() -> u32 = std::mem::transmute(ptr);
            let res = func();
            assert_eq!(res, 55);
        }
    }

    #[test]
    fn write_exec_subroutine() {
        unsafe extern "C" fn target_func() -> u32 {
            42
        }

        let code = &mut [
            0xE9, 0x00, 0x00, 0x00, 0x00, // jmp
        ];

        const PAGE_SIZE: usize = 4096;

        // Possible addresses fall within the range of i32
        let min_addr = (target_func as usize).saturating_sub(i32::MAX as usize);
        let max_addr = (target_func as usize) + (i32::MAX as usize);
        // And are aligned to a page boundary
        let min_addr = (min_addr / PAGE_SIZE) * PAGE_SIZE + PAGE_SIZE;
        let max_addr = (max_addr / PAGE_SIZE) * PAGE_SIZE;

        println!("target: {}", target_func as isize);
        println!("min code addr: {}", min_addr);
        println!("max code addr: {}", max_addr);
        println!("allowed range size: {}", max_addr - min_addr);

        unsafe {
            let mut ptr = std::ptr::null_mut();
            for addr in (min_addr..max_addr).step_by(PAGE_SIZE) {
                ptr = libc::mmap(
                    addr as *mut libc::c_void,
                    code.len(),
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                    -1,
                    0,
                );
                if ptr != libc::MAP_FAILED {
                    break;
                }
            }
            assert!(!ptr.is_null());
            assert_ne!(ptr, libc::MAP_FAILED);

            // Setup jump address
            let base_addr = ptr.add(code.len()) as *const u8;
            println!("base at {:?}", base_addr);
            let target_addr = target_func as *const u8;
            println!("target at {:?}", target_addr);

            let offset = target_addr.offset_from(base_addr);
            println!("offset: {} (min: {}, max: {})", offset, i32::MIN, i32::MAX);

            // Encode the jump target
            let offset: i32 = offset.try_into().unwrap();
            code[1..5].copy_from_slice(&offset.to_le_bytes()[0..4]);
            std::ptr::copy(code.as_ptr(), ptr as *mut u8, code.len());

            let func: unsafe extern "C" fn() -> u32 = std::mem::transmute(ptr);
            let res = func();
            assert_eq!(res, 42);
        }
    }

    #[test]
    fn call_subroutine() {
        unsafe extern "C" fn target_func() -> u32 {
            42
        }

        let code = &mut [
            0xE8, 0x00, 0x00, 0x00, 0x00, // call
            RET,
        ];

        const PAGE_SIZE: usize = 4096;

        // Possible addresses fall within the range of i32
        let min_addr = (target_func as usize).saturating_sub(i32::MAX as usize);
        let max_addr = (target_func as usize) + (i32::MAX as usize);
        // And are aligned to a page boundary
        let min_addr = (min_addr / PAGE_SIZE) * PAGE_SIZE + PAGE_SIZE;
        let max_addr = (max_addr / PAGE_SIZE) * PAGE_SIZE;

        println!("target: {}", target_func as isize);
        println!("min code addr: {}", min_addr);
        println!("max code addr: {}", max_addr);
        println!("allowed range size: {}", max_addr - min_addr);

        unsafe {
            let mut ptr = std::ptr::null_mut();
            for addr in (min_addr..max_addr).step_by(PAGE_SIZE) {
                ptr = libc::mmap(
                    addr as *mut libc::c_void,
                    code.len(),
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                    -1,
                    0,
                );
                if ptr != libc::MAP_FAILED {
                    break;
                }
            }
            assert!(!ptr.is_null());
            assert_ne!(ptr, libc::MAP_FAILED);

            const CALL_LEN: usize = 5;
            // Setup jump address
            let base_addr = ptr.add(CALL_LEN) as *const u8;
            println!("base at {:?}", base_addr);
            let target_addr = target_func as *const u8;
            println!("target at {:?}", target_addr);

            let offset = target_addr.offset_from(base_addr);
            println!("offset: {} (min: {}, max: {})", offset, i32::MIN, i32::MAX);

            // Encode the jump target
            let offset: i32 = offset.try_into().unwrap();
            code[0..5].copy_from_slice(&assemble_call(offset));
            code[5] = RET;
            std::ptr::copy(code.as_ptr(), ptr as *mut u8, code.len());

            let func: unsafe extern "C" fn() -> u32 = std::mem::transmute(ptr);
            let res = func();
            assert_eq!(res, 42);
        }
    }

    // Opcode for return
    const RET: u8 = 0xC3;
}
