pub mod comb;
pub mod jit;

use std::{cell::UnsafeCell, collections::HashMap, panic::UnwindSafe};

use jit::JitMem;

use crate::{
    ast,
    compiled_expr::{Comb, CompiledExpr, ExprCompiler},
    Engine,
};

use self::jit::JitPlacer;

// Cells are 16 bytes for optimal alignment.
// See https://groups.google.com/g/golang-nuts/c/dhGbgC1pAmA/m/Rcqwcd5mGmoJ.
#[repr(packed)]
pub struct Cell {
    // CALL rel32 (0xE8)
    call_opcode: u8,
    call_rel_addr: i32,
    // right-hand side.
    // Can be an integer or a pointer
    arg: i64,
    used: bool,
    _padding: [u8; 2],
}

impl Cell {
    pub fn set_call_addr(&mut self, target_addr: usize) {
        let base_addr = self as *const Cell as isize + CALL_LEN;
        let call_rel_addr = target_addr as isize - base_addr;
        self.call_rel_addr = call_rel_addr.try_into().expect("have to jump too far");
    }
}

const CALL_OPCODE: u8 = 0xE8;
// OPCODE  + rel32
const CALL_LEN: isize = 5;

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
struct CellPtr(*mut Cell);

trait IntoCellPtr {
    fn into_cell_ptr(&self, engine: &TigreEngine) -> CellPtr;
}

impl IntoCellPtr for CellPtr {
    fn into_cell_ptr(&self, _: &TigreEngine) -> CellPtr {
        *self
    }
}

impl IntoCellPtr for Comb {
    fn into_cell_ptr(&self, _: &TigreEngine) -> CellPtr {
        match self {
            Comb::I => CellPtr(comb::comb_I as *mut Cell),
            Comb::Abort => CellPtr(comb::comb_Abort as *mut Cell),
            Comb::K => CellPtr(comb::comb_K as *mut Cell),
            Comb::S => CellPtr(comb::comb_S as *mut Cell),
            Comb::B => CellPtr(comb::comb_B as *mut Cell),
            Comb::C => CellPtr(comb::comb_C as *mut Cell),
            Comb::Plus => CellPtr(comb::comb_plus as *mut Cell),
            Comb::Minus => CellPtr(comb::comb_min as *mut Cell),
            Comb::Cond => CellPtr(comb::comb_cond as *mut Cell),
            Comb::Eq => CellPtr(comb::comb_eq as *mut Cell),
            Comb::Times => CellPtr(comb::comb_times as *mut Cell),
            Comb::Lt => CellPtr(comb::comb_lt as *mut Cell),
            Comb::Sn(2) => CellPtr(comb::comb_S2 as *mut Cell),
            Comb::Bn(2) => CellPtr(comb::comb_B2 as *mut Cell),
            Comb::Cn(2) => CellPtr(comb::comb_C2 as *mut Cell),
            _ => unimplemented!("into_cell_ptr {:?}", self),
        }
    }
}

struct Lit;
impl IntoCellPtr for Lit {
    fn into_cell_ptr(&self, _: &TigreEngine) -> CellPtr {
        CellPtr(comb::comb_LIT as *mut Cell)
    }
}

impl IntoCellPtr for i32 {
    fn into_cell_ptr(&self, _: &TigreEngine) -> CellPtr {
        CellPtr(*self as *mut Cell)
    }
}

const JIT_MEM_LEN: usize = std::mem::size_of::<Cell>() * 1_000_000;

pub struct TigreEngine {
    mem: JitMem,
    next_free_cell: usize,
    // Points to the cell containing the compiled definition
    def_lookup: HashMap<String, CellPtr>,
}

impl Engine for TigreEngine {
    fn compile<C: ExprCompiler>(compiler: &mut C, program: &ast::Program) -> TigreEngine {
        let mut jit_placer = JitPlacer::new();
        for comb in comb::ALL_COMBINATORS {
            jit_placer.add_target(*comb as usize);
        }
        let mut engine = TigreEngine {
            mem: jit_placer
                .place_jit(JIT_MEM_LEN)
                .expect("failed to create JIT memory"),
            next_free_cell: 0,
            def_lookup: HashMap::new(),
        };
        // Reserve a cell for each definition
        for def in &program.defs {
            let cell_ptr = engine.make_cell(Comb::I, Comb::Abort);
            if let Some(_) = engine.def_lookup.insert(def.name.clone(), cell_ptr) {
                panic!("Duplicate definition of {}", def.name);
            }
        }

        // Compile all definitions
        for def in &program.defs {
            engine.alloc_def(&def.name, compiler.compile(&def.as_lam()));
        }
        engine
    }

    fn run(&mut self) -> i32 {
        let main_ptr = self.def_lookup.get("main").expect("No main function");
        let func: fn() -> i64 = unsafe { std::mem::transmute(main_ptr.0) };

        // Configure the global engine reference
        ENGINE.with(|engine_cell| {
            let engine_cell = UnsafeCell::raw_get(engine_cell as *const UnsafeCell<_>);
            unsafe { *engine_cell = self as *mut TigreEngine };
        });

        let res = func();

        // Deregister global engine reference
        ENGINE.with(|engine_cell| {
            let engine_cell = UnsafeCell::raw_get(engine_cell as *const UnsafeCell<_>);
            unsafe { *engine_cell = std::ptr::null_mut() };
        });
        res.try_into().expect("result does not fit into i32")
    }
}

impl TigreEngine {
    fn make_cell<CP0: IntoCellPtr, CP1: IntoCellPtr>(&mut self, lhs: CP0, rhs: CP1) -> CellPtr {
        let cell_ptr = self.alloc_cell();

        let target_addr = lhs.into_cell_ptr(self).0 as usize;
        let arg = rhs.into_cell_ptr(self);

        let cell = self.cell_mut(cell_ptr);
        cell.used = true;
        cell.call_opcode = CALL_OPCODE;
        cell.set_call_addr(target_addr);
        cell.arg = arg.0 as i64;
        cell_ptr
    }

    fn alloc_cell(&mut self) -> CellPtr {
        let cells = self.mem.ptr_at::<Cell>(0) as *mut Cell;
        let cells = unsafe {
            std::slice::from_raw_parts_mut(
                cells,
                self.mem.slice_mut().len() / std::mem::size_of::<Cell>(),
            )
        };
        loop {
            let cell = &mut cells[self.next_free_cell];
            if !cell.used {
                return CellPtr(cell as *mut Cell);
            }
            self.next_free_cell += 1;
            if self.next_free_cell >= cells.len() {
                panic!("Out of heap!");
            }
        }
    }

    fn cell_mut(&mut self, ptr: CellPtr) -> &mut Cell {
        unsafe { &mut *ptr.0 }
    }

    fn alloc_def(&mut self, name: &str, expr: CompiledExpr) {
        let cell_ptr = self.alloc_expr(expr);
        let def_ptr = self.def_lookup.get(name).unwrap();
        self.cell_mut(*def_ptr).arg = cell_ptr.0 as i64;
    }

    fn alloc_expr(&mut self, expr: CompiledExpr) -> CellPtr {
        match expr {
            CompiledExpr::Int(i) => self.make_cell(Lit, i),
            CompiledExpr::Comb(c) => c.into_cell_ptr(self),
            CompiledExpr::Ap(l, r) => {
                let l = self.alloc_expr(*l);
                let r = self.alloc_expr(*r);
                self.make_cell(l, r)
            }
            CompiledExpr::Var(v) => *self
                .def_lookup
                .get(&v)
                .expect(&format!("unknown def '{v}'")),
        }
    }

    pub fn with_current<R, F: FnOnce(&mut TigreEngine) -> R + UnwindSafe>(f: F) -> R {
        // This function is always called from a combinator implementation.
        // Unwinding into non-rust code is undefined behaviour, so catch any panics here.
        let maybe_panic = std::panic::catch_unwind(|| {
            let engine_ptr = ENGINE.with(|engine_cell| unsafe { *engine_cell.get() });
            let engine = unsafe {
                debug_assert!(!engine_ptr.is_null());
                &mut *engine_ptr
            };
            f(engine)
        });

        match maybe_panic {
            Ok(res) => res,
            Err(e) => {
                if let Some(msg) = e.downcast_ref::<&str>() {
                    eprintln!("panic within TigreEngine::run: {}", msg);
                } else {
                    eprintln!("panic within TigreEngine::run");
                }
                std::process::abort();
            }
        }
    }
}

thread_local!(static ENGINE: UnsafeCell<*mut TigreEngine> = UnsafeCell::new(std::ptr::null_mut()));

#[cfg(test)]
mod tests {
    use crate::{bracket::BracketCompiler, lex, parse};

    use super::*;

    #[test]
    fn cell_size() {
        assert_eq!(std::mem::size_of::<Cell>(), 16);
    }

    #[test]
    fn test_lit() {
        assert_runs_to_int("(defun main () 42)", 42);
    }

    fn assert_runs_to_int(program: &str, v: i32) {
        let parsed = parse(lex(program));
        let mut engine = TigreEngine::compile(&mut BracketCompiler, &parsed);

        let res = engine.run();
        assert_eq!(res, v);
    }
}
