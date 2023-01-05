pub mod comb;
pub mod jit;

use std::collections::HashMap;

use jit::JitMem;

use crate::{
    ast,
    compiled_expr::{Comb, CompiledExpr, ExprCompiler},
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

const CALL_OPCODE: u8 = 0xE8;
// OPCODE  + rel32
const CALL_LEN: isize = 5;

impl Cell {
    fn new(lhs: i32, rhs: i32) -> Cell {
        Cell {
            call_opcode: CALL_OPCODE,
            call_rel_addr: lhs,
            arg: rhs as i64,
            used: true,
            _padding: [0u8; 2],
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct CellPtr(*mut Cell);

trait IntoCellPtr {
    fn into_cell_ptr(&self, engine: &TigreEngine) -> CellPtr;
}

impl IntoCellPtr for Comb {
    fn into_cell_ptr(&self, engine: &TigreEngine) -> CellPtr {
        match self {
            Comb::I => CellPtr(comb::comb_I as *mut Cell),
            Comb::Abort => CellPtr(comb::comb_Abort as *mut Cell),
            _ => unimplemented!("into_cell_ptr {:?}", self),
        }
    }
}

struct Lit;
impl IntoCellPtr for Lit {
    fn into_cell_ptr(&self, engine: &TigreEngine) -> CellPtr {
        CellPtr(comb::comb_LIT as *mut Cell)
    }
}

impl IntoCellPtr for i32 {
    fn into_cell_ptr(&self, engine: &TigreEngine) -> CellPtr {
        CellPtr(*self as *mut Cell)
    }
}

const JIT_MEM_LEN: usize = std::mem::size_of::<Cell>() * 10_000;

pub struct TigreEngine {
    mem: JitMem,
    next_free_cell: usize,
    // Points to the cell containing the compiled definition
    def_lookup: HashMap<String, CellPtr>,
}

impl TigreEngine {
    pub fn compile<C: ExprCompiler>(compiler: &mut C, program: &ast::Program) -> TigreEngine {
        let mut jit_placer = JitPlacer::new();
        jit_placer.add_target(comb::comb_LIT as usize);
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

    pub fn run(&mut self) -> i64 {
        let main_ptr = self.def_lookup.get("main").expect("No main function");
        let func: fn() -> i64 = unsafe { std::mem::transmute(main_ptr.0) };
        func()
    }

    fn make_cell<CP0: IntoCellPtr, CP1: IntoCellPtr>(&mut self, lhs: CP0, rhs: CP1) -> CellPtr {
        let cell_ptr = self.alloc_cell();

        let base_addr = cell_ptr.0 as isize + CALL_LEN;
        let target_addr = lhs.into_cell_ptr(self).0 as isize;
        let call_rel_addr = target_addr - base_addr;
        let arg = rhs.into_cell_ptr(self);

        let cell = self.cell_mut(cell_ptr);
        cell.used = true;
        cell.call_opcode = CALL_OPCODE;
        cell.call_rel_addr = call_rel_addr.try_into().expect("Have to jump too far");
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
            _ => unimplemented!("alloc {:?}", expr),
        }
    }
}

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
        let program = "(defun main () 42)";
        let parsed = parse(lex(program));
        let mut engine = TigreEngine::compile(&mut BracketCompiler, &parsed);

        let res = engine.run();
        assert_eq!(res, 42)
    }
}
