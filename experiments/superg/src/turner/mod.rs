use crate::compiled_expr::{Comb, CompiledExpr};
use std::io::Write;
use std::{collections::HashMap, fs::File, io::BufWriter, path::PathBuf};

use crate::{ast, bracket};

// TODO: Allow dynamically growing the heap.
const CELLS: usize = 250_000;

/// The core atom in the reduction machine.
/// Low pointer values represent combinators, high values are indices into the global cells array.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CellPtr(i32);

impl CellPtr {
    pub fn comb(&self, engine: &TurnerEngine) -> Option<Comb> {
        engine.comb_inst.get(self.0 as usize).copied()
    }

    pub fn is_comb(&self, engine: &TurnerEngine) -> bool {
        (self.0 as usize) < engine.comb_impl.len()
    }
}

trait IntoCellPtr {
    fn into_cell_ptr(self, engine: &TurnerEngine) -> CellPtr;
}

impl IntoCellPtr for CellPtr {
    fn into_cell_ptr(self, _: &TurnerEngine) -> CellPtr {
        self
    }
}

impl IntoCellPtr for Comb {
    fn into_cell_ptr(self, engine: &TurnerEngine) -> CellPtr {
        CellPtr(engine.comb_map.get(&self).copied().unwrap() as i32)
    }
}

impl Comb {
    fn implementation(&self) -> fn(&mut TurnerEngine) -> Option<CellPtr> {
        match self {
            _ => TurnerEngine::run_todo_comb,
        }
    }
}

/// Tag byte.
/// Contains bitflags for a cell.
#[derive(Debug, Copy, Clone)]
struct Tag(u8);

impl Tag {
    const WANTED: u8 = 1u8 << 7;
    const RHS_INT: u8 = 1u8 << 6;
    pub fn wanted() -> Tag {
        Tag(Tag::WANTED)
    }

    pub fn set_rhs_int(self) -> Tag {
        Tag(self.0 | Tag::RHS_INT)
    }

    pub fn set_wanted(self) -> Tag {
        Tag(self.0 | Tag::WANTED)
    }

    pub fn set_unwanted(self) -> Tag {
        Tag(self.0 & (!Tag::WANTED))
    }

    pub fn is_wanted(self) -> bool {
        return self.0 & Tag::WANTED != 0;
    }

    pub fn is_rhs_int(self) -> bool {
        return self.0 & Tag::RHS_INT != 0;
    }
}

pub struct TurnerEngine {
    // Conceptually, a cell looks like this:
    //
    // |=====|======|======|
    // | TAG | HEAD | TAIL |
    // |=====|======|======|
    //
    // We could store that in a Vec<(Tag, CellPtr,CellPtr), or even define a Cell type, but Miranda chooses this decomposed form, so we stick with that.
    // TODO: Experiment with storing tuples instead.
    tag: Vec<Tag>,
    hd: Vec<CellPtr>,
    tl: Vec<CellPtr>,
    // The next free cell
    next_cell: CellPtr,
    // Points to the cell containing the compiled definition
    def_lookup: HashMap<String, CellPtr>,
    stack: Vec<CellPtr>,

    // Combinator implementation lookup table
    comb_impl: Vec<fn(&mut TurnerEngine) -> Option<CellPtr>>,
    // Combinator instance lookup table
    comb_inst: Vec<Comb>,
    // Maps combinator to the index in comb_impl/comb_inst
    comb_map: HashMap<Comb, usize>,
}

impl TurnerEngine {
    pub fn compile(program: &ast::Program) -> TurnerEngine {
        let mut engine = TurnerEngine {
            tag: vec![Tag::wanted(); CELLS],
            hd: vec![CellPtr(0i32); CELLS],
            tl: vec![CellPtr(0i32); CELLS],
            // We don't allocate cells when the pointer to it would be ambiguous
            next_cell: CellPtr(Comb::all().len() as i32),
            def_lookup: HashMap::new(),
            stack: Vec::new(),
            comb_impl: Vec::new(),
            comb_inst: Vec::new(),
            comb_map: HashMap::new(),
        };

        // TODO: populate during graph loading
        for c in Comb::all() {
            engine.comb_impl.push(c.implementation());
            engine.comb_inst.push(*c);
            engine.comb_map.insert(*c, engine.comb_impl.len() - 1);
        }

        // Reserve a spot for each definition
        for def in &program.defs {
            let cell_ptr = engine.make_cell(Tag::wanted(), Comb::I, Comb::Abort);
            if let Some(_) = engine.def_lookup.insert(def.name.clone(), cell_ptr) {
                panic!("Duplicate definition of {}", def.name);
            }
        }

        // Compile all definitions
        for def in &program.defs {
            engine.compile_def(def);
        }
        engine
    }

    pub fn run(&mut self) -> CellPtr {
        self.stack.clear();
        self.stack
            .push(*self.def_lookup.get("main").expect("no main function found"));
        loop {
            if let Some(cell_ptr) = self.step() {
                return cell_ptr;
            }
        }
    }

    fn step(&mut self) -> Option<CellPtr> {
        let top = self.stack.last().unwrap();
        if let Some(comb) = top.comb(self) {
            match comb {
                Comb::S => self.run_s_comb(),
                Comb::K => self.run_k_comb(),
                Comb::I => self.run_i_comb(),
                Comb::Y => todo!(),
                Comb::U => todo!(),
                Comb::P => todo!(),
                Comb::B => todo!(),
                Comb::C => todo!(),
                Comb::Plus => self.run_plus_comb(),
                Comb::Minus => self.run_minus_comb(),
                Comb::Times => self.run_times_comb(),
                Comb::Divide => todo!(),
                Comb::Cond => self.run_cond_comb(),
                Comb::Eq => self.run_eq_comb(),
                Comb::Neq => todo!(),
                Comb::Gt => todo!(),
                Comb::Gte => todo!(),
                Comb::Lt => self.run_lt_comb(),
                Comb::Lte => todo!(),
                Comb::And => todo!(),
                Comb::Or => todo!(),
                Comb::Not => todo!(),
                Comb::Abort => todo!(),
            }
        } else {
            // An application, so push the left subtree
            self.stack.push(self.hd(*top));
            None
        }
    }

    fn run_s_comb(&mut self) -> Option<CellPtr> {
        self.run_comb3(|engine, args| {
            // S f g x => f x (g x)
            let x = engine.tl(args[2]);
            let g = engine.tl(args[1]);
            let f = engine.tl(args[0]);
            // If x is an int, we should transfer that to the new location
            let mut tag = Tag::wanted();
            if engine.tag(args[2]).is_rhs_int() {
                tag = tag.set_rhs_int();
            }
            // Make lower cells
            let left_cell = engine.make_cell(tag, f, x);
            let right_cell = engine.make_cell(tag, g, x);
            StepResult::CellContents(Tag::wanted(), left_cell, right_cell)
        })
    }

    fn run_k_comb(&mut self) -> Option<CellPtr> {
        self.run_comb2(|engine, args| {
            // K x y = x
            if let Some(v) = engine.int_rhs(args[0]) {
                StepResult::Value(v)
            } else {
                StepResult::Cell(engine.tl(args[0]))
            }
        })
    }

    fn run_i_comb(&mut self) -> Option<CellPtr> {
        self.run_comb1(|engine, args| {
            // I x = x
            if let Some(v) = engine.int_rhs(args[0]) {
                StepResult::Value(v)
            } else {
                StepResult::Cell(engine.tl(args[0]))
            }
        })
    }

    fn run_plus_comb(&mut self) -> Option<CellPtr> {
        self.run_strict_binop(|a, b| a + b)
    }

    fn run_minus_comb(&mut self) -> Option<CellPtr> {
        self.run_strict_binop(|a, b| a - b)
    }

    fn run_times_comb(&mut self) -> Option<CellPtr> {
        self.run_strict_binop(|a, b| a * b)
    }

    fn run_eq_comb(&mut self) -> Option<CellPtr> {
        self.run_strict_binop(|a, b| if a == b { 1 } else { 0 })
    }

    fn run_lt_comb(&mut self) -> Option<CellPtr> {
        self.run_strict_binop(|a, b| if a < b { 1 } else { 0 })
    }

    fn run_cond_comb(&mut self) -> Option<CellPtr> {
        self.run_comb3(|engine, args| {
            // COND c t f = if(c) t else f
            if let Some(c) = engine.int_rhs(args[0]) {
                let branch_ptr = match c {
                    0 => args[2],
                    1 => args[1],
                    v => {
                        // We expect never to reach this code.
                        // In debug mode we panic, in release the behaviour is undefined.
                        debug_assert!(v == 0 || v == 1, "condition variable should be 0 or 1");
                        unsafe { std::hint::unreachable_unchecked() }
                    }
                };
                let branch_tl = engine.tl(branch_ptr);
                // Check if tl is int
                if engine.tag(branch_ptr).is_rhs_int() {
                    StepResult::Value(branch_tl.0)
                } else {
                    StepResult::Cell(branch_tl)
                }
            } else {
                StepResult::EvaluateArg(engine.tl(args[0]))
            }
        })
    }

    fn run_todo_comb(&mut self) -> Option<CellPtr> {
        let top = self.stack.last().unwrap();
        let comb = top.comb(self).unwrap();
        todo!("combinator {:?} not implemented", comb)
    }

    fn run_strict_binop(&mut self, op: fn(i32, i32) -> i32) -> Option<CellPtr> {
        self.run_comb2(|engine, args| {
            let a = engine.int_rhs(args[0]);
            let b = engine.int_rhs(args[1]);

            match (a, b) {
                (Some(a), Some(b)) => StepResult::Value(op(a, b)),
                (Some(_), None) => StepResult::EvaluateArg(engine.tl(args[1])),
                (None, Some(_)) => StepResult::EvaluateArg(engine.tl(args[0])),
                (None, None) => StepResult::EvaluateArg2(engine.tl(args[0]), engine.tl(args[1])),
            }
        })
    }

    fn int_rhs(&mut self, cell_ptr: CellPtr) -> Option<i32> {
        let tag = self.tag(cell_ptr);
        let rhs_ptr = self.tl(cell_ptr);
        if tag.is_rhs_int() {
            return Some(rhs_ptr.0);
        }
        if rhs_ptr.is_comb(self) {
            // rhs points to a combinator
            return None;
        }
        // Check to see if RHS happens to point to an indirection node
        if self.hd(rhs_ptr).comb(self) == Some(Comb::I) {
            // TODO: Consider making indirection compression a job for I reduction
            // Remove one layer of indirection
            self.set_tag(cell_ptr, self.tag(rhs_ptr));
            self.set_tl(cell_ptr, self.tl(rhs_ptr));
            return self.int_rhs(cell_ptr);
        }
        None
    }

    fn compile_def(&mut self, def: &ast::Def) {
        let compiled_expr = bracket::compile(&def.as_lam());
        let cell_ptr = self.alloc_compiled_expr(compiled_expr);
        let def_ptr = self.def_lookup.get(&def.name).unwrap();
        // Set up the indirection from the definition to the compiled expression
        self.set_tl(*def_ptr, cell_ptr);
    }

    fn make_cell<CP0: IntoCellPtr, CP1: IntoCellPtr>(
        &mut self,
        tag: Tag,
        hd: CP0,
        tl: CP1,
    ) -> CellPtr {
        // Search for an unwanted cell
        let mut cell_idx = self.next_cell.0 as usize;
        let hd = hd.into_cell_ptr(self);
        let tl = tl.into_cell_ptr(self);
        while cell_idx >= self.tag.len() || self.tag[cell_idx].is_wanted() {
            cell_idx += 1;

            // We have exhausted the memory
            if cell_idx >= self.tag.len() {
                // Time for gc!
                let initial_queue = if tag.is_rhs_int() {
                    vec![hd]
                } else {
                    vec![hd, tl]
                };
                self.garbage_collect(initial_queue);
                cell_idx = Comb::all().len();
                continue;
            }
        }
        let cell_ptr = CellPtr(cell_idx as i32);
        self.set_tag(cell_ptr, tag);
        self.set_hd(cell_ptr, hd);
        self.set_tl(cell_ptr, tl);
        self.next_cell = CellPtr(cell_idx as i32 + 1);
        cell_ptr
    }

    // Simple mark and sweep garbage collect
    fn garbage_collect(&mut self, mut queue: Vec<CellPtr>) {
        // Phase 1: Mark everything unwanted
        for t in self.tag.iter_mut().skip(Comb::all().len()) {
            *t = t.set_unwanted();
        }

        // Phase 2: Start at the stack and mark everything reachable from it
        queue.extend(&self.stack);
        let mut cells_wanted = 0;
        while let Some(cell_ptr) = queue.pop() {
            // Skip combinators
            if cell_ptr.is_comb(self) {
                continue;
            }
            // Skip if already marked
            let tag = self.tag(cell_ptr);
            let visited = tag.is_wanted();
            if visited {
                continue;
            }

            // Mark this cell as wanted
            self.set_tag(cell_ptr, self.tag(cell_ptr).set_wanted());
            cells_wanted += 1;

            // Check children
            let hd = self.hd(cell_ptr);
            if !hd.is_comb(self) {
                // hd is ptr
                let hd_visited = self.tag(hd).is_wanted();
                if !hd_visited {
                    queue.push(hd);
                }
            }

            if !tag.is_rhs_int() {
                // tl is a ptr or comb, not int
                let tl = self.tl(cell_ptr);
                if !tl.is_comb(self) {
                    // tl is ptr
                    let tl_visited = self.tag(tl).is_wanted();
                    if !tl_visited {
                        queue.push(tl);
                    }
                }
            }
        }

        if cells_wanted >= CELLS {
            panic!("Out of memory!")
        }
        // Reset next cell pointer
        self.next_cell = CellPtr(Comb::all().len() as i32);
    }

    fn alloc_compiled_expr(&mut self, expr: CompiledExpr) -> CellPtr {
        match expr {
            CompiledExpr::Comb(c) => c.into_cell_ptr(self),
            CompiledExpr::Var(s) => self
                .def_lookup
                .get(&s)
                .copied()
                .expect(&format!("Missing definition for {:?}", s)),
            CompiledExpr::Ap(l, r) => {
                let mut tag = Tag::wanted();
                if let CompiledExpr::Int(_) = *r {
                    tag = tag.set_rhs_int();
                }
                let l = self.alloc_compiled_expr(*l);
                let r = self.alloc_compiled_expr(*r);
                self.make_cell(tag, l, r)
            }
            CompiledExpr::Int(i) => CellPtr(i),
        }
    }

    // In debug mode, checks that this is a valid CellPtr.
    // In release mode it is a no-op.
    fn debug_assert_ptr(&self, ptr: CellPtr) {
        // Responsibility of the caller to
        // check this in release mode.
        // Even if ptr is actually to a combinator,
        // it still points to a valid element in the array,
        // so it is a correctness bug rather than a memory error.
        debug_assert!(!ptr.is_comb(self));
        // Bounds check only in debug mode.
        // We assume CellPtr is never fabricated,
        // so it always contains a valid index.
        debug_assert!(ptr.0 >= 0);
        debug_assert!(
            (ptr.0 as usize) < self.tag.len()
                && (ptr.0 as usize) < self.hd.len()
                && (ptr.0 as usize) < self.tl.len()
        );
    }

    fn set_tag(&mut self, ptr: CellPtr, t: Tag) {
        self.debug_assert_ptr(ptr);
        *unsafe { self.tag.get_unchecked_mut(ptr.0 as usize) } = t;
    }

    fn tag(&self, ptr: CellPtr) -> Tag {
        self.debug_assert_ptr(ptr);
        *unsafe { self.tag.get_unchecked(ptr.0 as usize) }
    }

    fn set_hd<I: IntoCellPtr>(&mut self, ptr: CellPtr, v: I) {
        self.debug_assert_ptr(ptr);
        *unsafe { self.hd.get_unchecked_mut(ptr.0 as usize) } = v.into_cell_ptr(self);
    }

    fn hd(&self, ptr: CellPtr) -> CellPtr {
        self.debug_assert_ptr(ptr);
        *unsafe { self.hd.get_unchecked(ptr.0 as usize) }
    }

    fn set_tl<I: IntoCellPtr>(&mut self, ptr: CellPtr, v: I) {
        self.debug_assert_ptr(ptr);
        *unsafe { self.tl.get_unchecked_mut(ptr.0 as usize) } = v.into_cell_ptr(self);
    }

    fn tl(&self, ptr: CellPtr) -> CellPtr {
        self.debug_assert_ptr(ptr);
        *unsafe { self.tl.get_unchecked(ptr.0 as usize) }
    }

    pub fn get_int(&self, ptr: CellPtr) -> Option<i32> {
        if self.tag(ptr).is_rhs_int() {
            Some(self.tl(ptr).0)
        } else {
            None
        }
    }

    pub fn with_debug(self) -> TurnerEngineDebug {
        TurnerEngineDebug {
            engine: self,
            dump_path: None,
            step_counter: 0,
            step_limit: None,
        }
    }
}

enum StepResult {
    Value(i32),
    Cell(CellPtr),
    CellContents(Tag, CellPtr, CellPtr),
    EvaluateArg(CellPtr),
    EvaluateArg2(CellPtr, CellPtr),
}

// Generates helper functions for executing combinators that take $arg_count number of arguments to evaluate.
macro_rules! run_comb {
    ($name:ident, $arg_count:literal) => {
        fn $name<F: FnOnce(&mut TurnerEngine, [CellPtr; $arg_count]) -> StepResult>(
            &mut self,
            handler: F,
        ) -> Option<CellPtr> {
            const FRAME_SIZE: usize = $arg_count + 1;
            debug_assert!(self.stack.len() >= FRAME_SIZE);
            // Arguments are in reverse order because the stack is LIFO.
            let mut args = [CellPtr(0); $arg_count];
            for i in 0..$arg_count {
                args[i] = *unsafe { self.stack.get_unchecked(self.stack.len() - 2 - i) };
            }
            let frame_start = args[$arg_count - 1];

            match handler(self, args) {
                StepResult::Value(v) => {
                    // Make an indirection node
                    self.set_tag(frame_start, Tag::wanted().set_rhs_int());
                    self.set_hd(frame_start, Comb::I);
                    self.set_tl(frame_start, CellPtr(v));

                    // If nothing is on the stack, return the node
                    if self.stack.len() == FRAME_SIZE {
                        return Some(frame_start);
                    }
                    // Remove this frame, returning to the parent expression on the stack
                    unsafe { self.stack.set_len(self.stack.len() - FRAME_SIZE) };
                    None
                }
                StepResult::Cell(c) => {
                    // Make an indirection node
                    self.set_tag(frame_start, Tag::wanted());
                    self.set_hd(frame_start, Comb::I);
                    self.set_tl(frame_start, c);

                    // Make the new cell top of the stack
                    let new_len = self.stack.len() - $arg_count;
                    unsafe {
                        *self.stack.get_unchecked_mut(new_len - 1) = c;
                        self.stack.set_len(new_len);
                    }
                    None
                }
                StepResult::CellContents(tag, hd, tl) => {
                    // Create the new cell over the old one
                    self.set_tag(frame_start, tag);
                    self.set_hd(frame_start, hd);
                    self.set_tl(frame_start, tl);

                    // Make the new cell top of the stack
                    let new_len = self.stack.len() - $arg_count;
                    unsafe {
                        self.stack.set_len(new_len);
                    }
                    None
                }
                StepResult::EvaluateArg(a) => {
                    self.stack.push(a);
                    None
                }
                StepResult::EvaluateArg2(a, b) => {
                    self.stack.push(a);
                    self.stack.push(b);
                    None
                }
            }
        }
    };
}

impl TurnerEngine {
    run_comb!(run_comb1, 1);
    run_comb!(run_comb2, 2);
    run_comb!(run_comb3, 3);
}

/// To avoid slowing down [`TurnerEngine`] in benchmarks, all debugging related routines are in this wrapper.
pub struct TurnerEngineDebug {
    engine: TurnerEngine,
    // The directory to dump debug information to
    dump_path: Option<PathBuf>,
    step_counter: usize,
    step_limit: Option<usize>,
}

impl TurnerEngineDebug {
    fn step(&mut self) -> Option<CellPtr> {
        self.step_counter += 1;
        if let Some(limit) = self.step_limit {
            if self.step_counter > limit {
                panic!("Max cycle reached");
            }
        }
        self.dump_dot().expect("Dump failed");
        self.engine.step()
    }

    pub fn run(&mut self) -> CellPtr {
        // This is a copy of run under TurnerEngine.
        // TODO: deduplicate?
        self.engine.stack.clear();
        self.engine.stack.push(
            *self
                .engine
                .def_lookup
                .get("main")
                .expect("no main function found"),
        );
        loop {
            if let Some(cell_ptr) = self.step() {
                return cell_ptr;
            }
        }
    }

    pub fn get_int(&self, cell_ptr: CellPtr) -> Option<i32> {
        self.engine.get_int(cell_ptr)
    }

    pub fn dump_dot(&mut self) -> std::io::Result<()> {
        let mut w = if let Some(dump_path) = &self.dump_path {
            let f = File::create(dump_path.join(format!("step{}.dot", self.step_counter)))?;
            BufWriter::new(f)
        } else {
            return Ok(());
        };
        // Garbage collect so we don't render dead cells
        self.engine.garbage_collect(vec![]);
        writeln!(w, "digraph {{")?;
        // Nodes
        writeln!(w, "node [shape=record];")?;
        for c in 0..CELLS {
            let tag = self.engine.tag[c];
            if !tag.is_wanted() {
                // unwanted
                continue;
            }
            let hd = self.engine.hd[c];
            let hd = if let Some(comb) = hd.comb(&self.engine) {
                format!("{:?}", comb)
            } else {
                String::new()
            };
            let tl = self.engine.tl[c];
            let tl = if tag.is_rhs_int() {
                format!("{}", tl.0)
            } else if let Some(comb) = tl.comb(&self.engine) {
                format!("{:?}", comb)
            } else {
                String::new()
            };
            writeln!(w, "cell{} [label=\"<hd> {}|<tl> {}\"];", c, hd, tl)?;
        }
        // Stack
        if !self.engine.stack.is_empty() {
            write!(w, "stack [pos=\"0,0!\", label=\"{{")?;
            for (i, c) in self.engine.stack.iter().enumerate() {
                if let Some(comb) = c.comb(&self.engine) {
                    write!(w, "<s{}> {:?}", i, comb)?;
                } else {
                    write!(w, "<s{}> ", i)?;
                }
                if i != self.engine.stack.len() - 1 {
                    write!(w, "|")?;
                }
            }
            writeln!(w, "}}\"];")?;
        }

        // Edges
        for c in 0..CELLS {
            let tag = self.engine.tag[c];
            if !tag.is_wanted() {
                continue;
            }
            let hd = self.engine.hd[c];
            if !hd.is_comb(&self.engine) {
                writeln!(w, "cell{}:hd -> cell{};", c, hd.0)?;
            }
            let tl = self.engine.tl[c];
            if !tl.is_comb(&self.engine) && !tag.is_rhs_int() {
                writeln!(w, "cell{}:tl -> cell{};", c, tl.0)?;
            }
        }
        // Defs
        for (n, p) in self.engine.def_lookup.iter() {
            writeln!(w, "{} -> cell{}", n, p.0)?;
        }

        // Stack edges
        if !self.engine.stack.is_empty() {
            for (i, c) in self.engine.stack.iter().enumerate() {
                if !c.is_comb(&self.engine) {
                    writeln!(w, "stack:s{} -> cell{}", i, c.0)?;
                }
            }
        }

        writeln!(w, "}}")?;
        Ok(())
    }

    pub fn set_dump_path(&mut self, dump_path: String) {
        let dump_path = PathBuf::from(dump_path);
        std::fs::create_dir_all(&dump_path).expect("failed to create dump directory");
        // Clean up old dump files
        for entry in std::fs::read_dir(&dump_path).unwrap() {
            let entry = entry.unwrap();
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".dot") {
                    let path = dump_path.join(entry.file_name());
                    println!("Deleting file {:?}", path);
                    std::fs::remove_file(path).expect("failed to delete file");
                }
            }
        }
        self.dump_path = Some(dump_path);
    }

    pub fn set_step_limit(&mut self, l: usize) {
        self.step_limit = Some(l);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    #[test]
    fn test_id() {
        assert_runs_to_int(
            "test_id",
            r#"
(defun id (x) x)
(defun main () (id 42))
        "#,
            42,
            StepLimit(10),
        );
    }

    #[test]
    fn test_k() {
        assert_runs_to_int(
            "test_k",
            r#"
(defun k (x y) x)
(defun main () (k 42 84))
        "#,
            42,
            StepLimit(20),
        );
    }

    #[test]
    fn test_s() {
        assert_runs_to_int(
            "test_s",
            r#"
(defun s (f g x) (f x (g x)))
(defun k (x y) x)
(defun main () (s k k 42))
        "#,
            42,
            StepLimit(200),
        );
    }

    #[test]
    fn test_add() {
        assert_runs_to_int(
            "test_add",
            r#"
(defun main () (+ 2 40))
        "#,
            42,
            StepLimit(10),
        );
    }

    #[test]
    fn test_add_indirect() {
        assert_runs_to_int(
            "test_add_indirect",
            r#"
(defun id (x) x)
(defun main () (+ (id 2) (id 40)))
        "#,
            42,
            StepLimit(20),
        );
    }

    #[test]
    fn test_cond() {
        assert_runs_to_int(
            "test_cond",
            r#"
(defun main () (if 0 1000 (if 1 42 2000)))
        "#,
            42,
            StepLimit(10),
        )
    }

    #[test]
    fn test_cond_add() {
        assert_runs_to_int(
            "test_cond_add1",
            r#"
(defun main () (+ 2 (if 0 30 40)))
        "#,
            42,
            StepLimit(10),
        );

        assert_runs_to_int(
            "test_cond_add2",
            r#"
(defun main () (if 0 30 (+ 40 2)))
        "#,
            42,
            StepLimit(10),
        );
    }

    #[test]
    fn test_eq() {
        assert_runs_to_int(
            "test_eq",
            r#"
    (defun id (x) x)
    (defun k (x y) x)
    (defun main () (= (k 1 1000) 0))
            "#,
            0,
            StepLimit(20),
        );
    }

    #[test]
    fn test_cond_eq() {
        assert_runs_to_int(
            "test_cond_eq",
            r#"
    (defun main () (if (= 2 2) 42 43))
            "#,
            42,
            StepLimit(10),
        );
    }

    #[test]
    fn test_factorial() {
        assert_runs_to_int(
            "test_factorial",
            r#"
    (defun fac (n)
      (if (= n 1)
          1
          (* n (fac (- n 1)))))
    (defun main () (fac 5))
            "#,
            120,
            StepLimit(1000),
        );
    }

    #[test]
    fn test_fibonacci() {
        assert_runs_to_int(
            "test_fibonacci",
            r#"
    (defun fib (n)
      (if (< n 2) 
          n
          (+ (fib (- n 1)) (fib (- n 2)))))
    (defun main () (fib 5))
            "#,
            5,
            StepLimit(2000),
        );
    }

    #[test]
    fn test_ackermann() {
        let program = r#"
(defun ack (x z) (if (= x 0)
                     (+ z 1)
                     (if (= z 0)
                         (ack (- x 1) 1)
                         (ack (- x 1) (ack x (- z 1))))))
(defun main () (ack 3 4))
    "#;
        assert_runs_to_int("test_ackermann", program, 125, StepLimit(10_000_000));
    }

    struct StepLimit(usize);

    fn assert_runs_to_int(_test_name: &str, program: &str, v: i32, l: StepLimit) {
        let parsed = parse(lex(program));
        let mut engine = TurnerEngine::compile(&parsed).with_debug();
        // disabled by default because it slows things down a lot, enable for debugging
        //engine.set_dump_path(format!("/tmp/{}", _test_name));
        engine.set_step_limit(l.0);

        let ptr = engine.run();
        assert_eq!(engine.get_int(ptr), Some(v));
    }
}
