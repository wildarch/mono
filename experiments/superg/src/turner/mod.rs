use crate::compiled_expr::{Comb, CompiledExpr};
use std::io::Write;
use std::{collections::HashMap, fs::File, io::BufWriter, path::PathBuf};

use crate::ast;

const CELLS: usize = 250_000;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CellPtr(i32);

impl CellPtr {
    pub fn comb(&self) -> Option<Comb> {
        Comb::all().get(self.0 as usize).copied()
    }
}

impl Into<CellPtr> for Comb {
    fn into(self) -> CellPtr {
        CellPtr(self as i32)
    }
}

const TAG_WANTED: u8 = 1u8 << 7;
const TAG_RHS_INT: u8 = 1u8 << 6;

pub struct TurnerEngine {
    tag: Vec<u8>,
    hd: Vec<CellPtr>,
    tl: Vec<CellPtr>,
    // The next free cell
    next_cell: CellPtr,
    // Points to the cell containing the compiled definition
    def_lookup: HashMap<String, CellPtr>,
    stack: Vec<CellPtr>,
    // The directory to dump debug information to
    dump_path: Option<PathBuf>,
    step_counter: usize,
    step_limit: Option<usize>,
}

impl TurnerEngine {
    pub fn compile(program: &ast::Program) -> TurnerEngine {
        let mut engine = TurnerEngine {
            tag: vec![0; CELLS],
            hd: vec![CellPtr(0i32); CELLS],
            tl: vec![CellPtr(0i32); CELLS],
            // We don't allocate cells when the pointer to it would be ambiguous
            next_cell: CellPtr(Comb::all().len() as i32),
            def_lookup: HashMap::new(),
            stack: Vec::new(),
            dump_path: None,
            step_counter: 0,
            step_limit: None,
        };

        // Reserve a spot for each definition
        for def in &program.defs {
            let cell_ptr = engine.make_cell(TAG_WANTED, Comb::I, Comb::Abort);
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
            self.step_counter += 1;
            if let Some(limit) = self.step_limit {
                if self.step_counter > limit {
                    panic!("Max cycle reached");
                }
            }
            self.dump_dot().expect("Dump failed");
            let top = self.stack.last().unwrap();
            if let Some(comb) = top.comb() {
                match comb {
                    Comb::S => {
                        debug_assert!(
                            self.stack.len() >= 4,
                            "(step {}) S requires 4 arguments on the stack: {:?}",
                            self.step_counter,
                            self.stack
                        );
                        // S f g x => f x (g x)
                        let l0 = self.stack[self.stack.len() - 2];
                        let l1 = self.stack[self.stack.len() - 3];
                        let l2 = self.stack[self.stack.len() - 4];

                        // If x is an int, we should transfer that to the new location
                        let x_tag = self.tag(l2) & TAG_RHS_INT;
                        let x = self.tl(l2);
                        let g = self.tl(l1);
                        let f = self.tl(l0);

                        // Make lower cells
                        // TODO: check if we guarantee nobody else holds a reference to cells l0 and l1, so we can overwrite them
                        let left_cell = self.make_cell(x_tag | TAG_WANTED, f, x);
                        let right_cell = self.make_cell(x_tag | TAG_WANTED, g, x);
                        // Overwrite upper cell
                        self.set_tag(l2, TAG_WANTED);
                        self.set_hd(l2, left_cell);
                        self.set_tl(l2, right_cell);
                        // Truncate to upper cell
                        self.stack.truncate(self.stack.len() - 3);
                    }
                    Comb::K => {
                        debug_assert!(
                            self.stack.len() >= 3,
                            "(step {}) K requires 3 arguments on the stack: {:?}",
                            self.step_counter,
                            self.stack
                        );
                        // K x y = x
                        let l0 = self.stack[self.stack.len() - 2];
                        let l1 = self.stack[self.stack.len() - 3];
                        let x = self.tl(l0);
                        // If x is an int, we should transfer that to the new location
                        let x_tag = self.tag(l0) & TAG_RHS_INT;

                        // Make the indirection node
                        self.set_tag(l1, x_tag | TAG_WANTED);
                        self.set_hd(l1, Comb::I);
                        self.set_tl(l1, x);

                        // Check if the value is an int, then we are done
                        if x_tag & TAG_RHS_INT != 0 {
                            if self.stack.len() == 3 {
                                self.stack.truncate(self.stack.len() - 2);
                                // No larger expression to evaluate next
                                return l1;
                            }
                            // Continue with the larger expression on the stack
                            self.stack.truncate(self.stack.len() - 3);
                        } else {
                            // Put x on the stack
                            let new_len = self.stack.len() - 2;
                            self.stack[new_len - 1] = x;
                            self.stack.truncate(new_len);
                        }
                    }
                    Comb::I => {
                        debug_assert!(
                            self.stack.len() >= 2,
                            "(step {}) I requires 2 arguments on the stack: {:?}",
                            self.step_counter,
                            self.stack
                        );
                        // TODO: Compress multiple indirections
                        // Check if we are done
                        let l0 = self.stack[self.stack.len() - 2];
                        let tag0 = self.tag(l0);
                        if tag0 & TAG_RHS_INT != 0 {
                            // Indirection node!
                            if self.stack.len() == 2 {
                                self.stack.truncate(self.stack.len() - 1);
                                // No larger expression to evaluate next
                                return l0;
                            } else {
                                // Continue with the larger expression on the stack
                                self.stack.truncate(self.stack.len() - 2);
                            }
                        } else {
                            // Take the argument, and evaluate that instead
                            let arg = self.tl(l0);
                            // Special case: for I (I x) = I x
                            if arg.comb() == Some(Comb::I) {
                                let l1 = self.stack[self.stack.len() - 3];
                                // Fixup the new indirection node
                                self.set_hd(l1, Comb::I);

                                self.stack.truncate(self.stack.len() - 2);
                            } else {
                                // Replace two items with one arg
                                let new_len = self.stack.len() - 1;
                                self.stack[new_len - 1] = arg;
                                self.stack.truncate(new_len);
                            }
                        }
                    }
                    Comb::Y => todo!("Y not implemented. Stack: {:?}", self.stack),
                    Comb::U => todo!(),
                    Comb::P => todo!(),
                    Comb::B => todo!(),
                    Comb::C => todo!(),
                    Comb::Plus => {
                        if let Some(ptr) = self.run_strict_binop(|a, b| a + b) {
                            return ptr;
                        }
                    }
                    Comb::Minus => {
                        if let Some(ptr) = self.run_strict_binop(|a, b| a - b) {
                            return ptr;
                        }
                    }
                    Comb::Times => {
                        if let Some(ptr) = self.run_strict_binop(|a, b| a * b) {
                            return ptr;
                        }
                    }
                    Comb::Divide => {
                        if let Some(ptr) = self.run_strict_binop(|a, b| a / b) {
                            return ptr;
                        }
                    }
                    Comb::Cond => {
                        debug_assert!(
                            self.stack.len() >= 4,
                            "(step {}) Cond requires 4 arguments on the stack: {:?}",
                            self.step_counter,
                            self.stack
                        );
                        // TODO: dedup with other strict combinators
                        let l0 = self.stack[self.stack.len() - 2];
                        let c = self.int_rhs(l0);
                        if let Some(c) = c {
                            // Condition already evaluated
                            let l1 = self.stack[self.stack.len() - 3];
                            let l2 = self.stack[self.stack.len() - 4];

                            let branch_ptr = match c {
                                0 => l2,
                                1 => l1,
                                v => panic!("Illegal condition value {}", v),
                            };
                            let tag = self.tag(branch_ptr) & TAG_RHS_INT;
                            let branch_tl = self.tl(branch_ptr);
                            // Make an indirection node to branch_ptr's RHS
                            self.set_tag(l2, tag | TAG_WANTED);
                            self.set_hd(l2, Comb::I);
                            self.set_tl(l2, branch_tl);

                            // Check if the value is an int, then we are done
                            if tag & TAG_RHS_INT != 0 {
                                if self.stack.len() == 4 {
                                    self.stack.truncate(self.stack.len() - 3);
                                    // No larger expression to evaluate next
                                    return branch_ptr;
                                }
                                // Continue with the larger expression on the stack
                                self.stack.truncate(self.stack.len() - 4);
                            } else {
                                // Update stack to the chosen branch
                                let new_len = self.stack.len() - 3;
                                self.stack[new_len - 1] = branch_tl;
                                self.stack.truncate(new_len);
                            }
                        } else {
                            self.stack.push(self.tl(l0));
                        }
                    }
                    Comb::Eq => {
                        if let Some(ptr) = self.run_strict_binop(|a, b| if a == b { 1 } else { 0 })
                        {
                            return ptr;
                        }
                    }
                    Comb::Neq => todo!(),
                    Comb::Gt => todo!(),
                    Comb::Gte => todo!(),
                    Comb::Lt => {
                        if let Some(ptr) = self.run_strict_binop(|a, b| if a < b { 1 } else { 0 }) {
                            return ptr;
                        }
                    }
                    Comb::Lte => todo!(),
                    Comb::And => todo!(),
                    Comb::Or => todo!(),
                    Comb::Not => todo!(),
                    Comb::Abort => todo!(),
                }
            } else {
                // An application, so push the left subtree
                self.stack.push(self.hd(*top));
            }
        }
    }

    fn run_strict_binop(&mut self, op: fn(i32, i32) -> i32) -> Option<CellPtr> {
        let l0 = self.stack[self.stack.len() - 2];
        let l1 = self.stack[self.stack.len() - 3];

        let a = self.int_rhs(l0);
        let b = self.int_rhs(l1);

        if let (Some(a), Some(b)) = (a, b) {
            // Already reduced
            self.set_tag(l1, TAG_WANTED | TAG_RHS_INT);
            self.set_hd(l1, Comb::I);
            self.set_tl(l1, CellPtr(op(a, b)));

            if self.stack.len() == 3 {
                self.stack.truncate(self.stack.len() - 2);
                // No larger expression to evaluate next
                return Some(l1);
            } else {
                // Continue with the larger expression on the stack
                self.stack.truncate(self.stack.len() - 3);
            }
        }

        // Push a and/or b onto the stack to evaluate
        if b.is_none() {
            self.stack.push(self.tl(l1));
        }
        if a.is_none() {
            self.stack.push(self.tl(l0));
        }
        None
    }

    fn int_rhs(&self, cell_ptr: CellPtr) -> Option<i32> {
        let tag = self.tag(cell_ptr);
        let rhs_ptr = self.tl(cell_ptr);
        if tag & TAG_RHS_INT != 0 {
            return Some(rhs_ptr.0);
        }
        // Check to see if RHS happens to point to an indirection node
        if self.hd(rhs_ptr).comb() == Some(Comb::I) {
            return self.int_rhs(rhs_ptr);
        }
        None
    }

    fn compile_def(&mut self, def: &ast::Def) {
        let mut compiled_expr = CompiledExpr::compile(&def.expr);
        for param in def.params.iter().rev() {
            compiled_expr = compiled_expr.abstract_var(param);
        }
        let cell_ptr = self.alloc_compiled_expr(compiled_expr);
        let def_ptr = self.def_lookup.get(&def.name).unwrap();
        // Set up the indirection from the definition to the compiled expression
        self.set_tl(*def_ptr, cell_ptr);
    }

    fn make_cell<CP0: Into<CellPtr>, CP1: Into<CellPtr>>(
        &mut self,
        tag: u8,
        hd: CP0,
        tl: CP1,
    ) -> CellPtr {
        // Search for an unwanted cell
        let mut cell_idx = self.next_cell.0 as usize;
        let hd = hd.into();
        let tl = tl.into();
        while cell_idx >= self.tag.len() || self.tag[cell_idx] & TAG_WANTED != 0 {
            cell_idx += 1;

            // We have exhausted the memory
            if cell_idx >= self.tag.len() {
                // Time for gc!
                let initial_queue = if tag & TAG_RHS_INT != 0 {
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
            *t &= !TAG_WANTED;
        }

        // Phase 2: Start at the stack and mark everything reachable from it
        queue.extend(&self.stack);
        let mut cells_wanted = 0;
        while let Some(cell_ptr) = queue.pop() {
            // Skip combinators
            if cell_ptr.comb().is_some() {
                continue;
            }
            // Skip if already marked
            let tag = self.tag(cell_ptr);
            let visited = tag & TAG_WANTED != 0;
            if visited {
                continue;
            }

            // Mark this cell as wanted
            self.set_tag(cell_ptr, self.tag(cell_ptr) | TAG_WANTED);
            cells_wanted += 1;

            // Check children
            let hd = self.hd(cell_ptr);
            if hd.comb().is_none() {
                // hd is ptr
                let hd_visited = self.tag(hd) & TAG_WANTED != 0;
                if !hd_visited {
                    queue.push(hd);
                }
            }

            if tag & TAG_RHS_INT == 0 {
                // tl is a ptr or comb, not int
                let tl = self.tl(cell_ptr);
                if tl.comb().is_none() {
                    // tl is ptr
                    let tl_visited = self.tag(tl) & TAG_WANTED != 0;
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
            CompiledExpr::Comb(c) => c.into(),
            CompiledExpr::Var(s) => self
                .def_lookup
                .get(&s)
                .copied()
                .expect(&format!("Missing definition for {:?}", s)),
            CompiledExpr::Ap(l, r) => {
                let mut tag = TAG_WANTED;
                if let CompiledExpr::Int(_) = *r {
                    tag |= TAG_RHS_INT;
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
        debug_assert!(ptr.comb().is_none());
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

    fn set_tag(&mut self, ptr: CellPtr, t: u8) {
        self.debug_assert_ptr(ptr);
        *unsafe { self.tag.get_unchecked_mut(ptr.0 as usize) } = t;
    }

    fn tag(&self, ptr: CellPtr) -> u8 {
        self.debug_assert_ptr(ptr);
        *unsafe { self.tag.get_unchecked(ptr.0 as usize) }
    }

    fn set_hd<I: Into<CellPtr>>(&mut self, ptr: CellPtr, v: I) {
        self.debug_assert_ptr(ptr);
        *unsafe { self.hd.get_unchecked_mut(ptr.0 as usize) } = v.into();
    }

    fn hd(&self, ptr: CellPtr) -> CellPtr {
        self.debug_assert_ptr(ptr);
        *unsafe { self.hd.get_unchecked(ptr.0 as usize) }
    }

    fn set_tl<I: Into<CellPtr>>(&mut self, ptr: CellPtr, v: I) {
        self.debug_assert_ptr(ptr);
        *unsafe { self.tl.get_unchecked_mut(ptr.0 as usize) } = v.into();
    }

    fn tl(&self, ptr: CellPtr) -> CellPtr {
        self.debug_assert_ptr(ptr);
        *unsafe { self.tl.get_unchecked(ptr.0 as usize) }
    }

    pub fn dump_dot(&mut self) -> std::io::Result<()> {
        let mut w = if let Some(dump_path) = &self.dump_path {
            let f = File::create(dump_path.join(format!("step{}.dot", self.step_counter)))?;
            BufWriter::new(f)
        } else {
            return Ok(());
        };
        // Garbage collect so we don't render dead cells
        self.garbage_collect(vec![]);
        writeln!(w, "digraph {{")?;
        // Nodes
        writeln!(w, "node [shape=record];")?;
        for c in 0..CELLS {
            let tag = self.tag[c];
            if tag & TAG_WANTED == 0 {
                // unwanted
                continue;
            }
            let hd = self.hd[c];
            let hd = if let Some(comb) = hd.comb() {
                format!("{:?}", comb)
            } else {
                String::new()
            };
            let tl = self.tl[c];
            let tl = if tag & TAG_RHS_INT != 0 {
                format!("{}", tl.0)
            } else if let Some(comb) = tl.comb() {
                format!("{:?}", comb)
            } else {
                String::new()
            };
            writeln!(w, "cell{} [label=\"<hd> {}|<tl> {}\"];", c, hd, tl)?;
        }
        // Stack
        if !self.stack.is_empty() {
            write!(w, "stack [pos=\"0,0!\", label=\"{{")?;
            for (i, c) in self.stack.iter().enumerate() {
                if let Some(comb) = c.comb() {
                    write!(w, "<s{}> {:?}", i, comb)?;
                } else {
                    write!(w, "<s{}> ", i)?;
                }
                if i != self.stack.len() - 1 {
                    write!(w, "|")?;
                }
            }
            writeln!(w, "}}\"];")?;
        }

        // Edges
        for c in 0..CELLS {
            let tag = self.tag[c];
            if tag & TAG_WANTED == 0 {
                // unwanted
                continue;
            }
            let hd = self.hd[c];
            if hd.comb().is_none() {
                writeln!(w, "cell{}:hd -> cell{};", c, hd.0)?;
            }
            let tl = self.tl[c];
            if tl.comb().is_none() && tag & TAG_RHS_INT == 0 {
                writeln!(w, "cell{}:tl -> cell{};", c, tl.0)?;
            }
        }
        // Defs
        for (n, p) in self.def_lookup.iter() {
            writeln!(w, "{} -> cell{}", n, p.0)?;
        }

        // Stack edges
        if !self.stack.is_empty() {
            for (i, c) in self.stack.iter().enumerate() {
                if c.comb().is_none() {
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

    pub fn get_int(&self, ptr: CellPtr) -> Option<i32> {
        if self.tag[ptr.0 as usize] | TAG_RHS_INT != 0 {
            Some(self.tl(ptr).0)
        } else {
            None
        }
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
        let mut engine = TurnerEngine::compile(&parsed);
        // disabled by default because it slows things down a lot, enable for debugging
        //engine.set_dump_path(format!("/tmp/{}", _test_name));
        engine.set_step_limit(l.0);

        let ptr = engine.run();
        assert_eq!(engine.get_int(ptr), Some(v));
    }
}
