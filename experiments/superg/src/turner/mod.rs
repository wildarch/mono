use std::io::Write;
use std::{collections::HashMap, fs::File, io::BufWriter, path::PathBuf};

use crate::ast;

const CELLS: usize = 250_000;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CellPtr(i32);

impl CellPtr {
    pub fn comb(&self) -> Option<Comb> {
        ALL_COMBS.get(self.0 as usize).copied()
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum Comb {
    S,
    K,
    I,
    Y,
    U,
    P,
    Plus,
    Minus,
    Times,
    Divide,
    Cond,
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    And,
    Or,
    Not,
    Abort,
}
impl Into<CellPtr> for Comb {
    fn into(self) -> CellPtr {
        CellPtr(self as i32)
    }
}
const ALL_COMBS: &'static [Comb] = &[
    Comb::S,
    Comb::K,
    Comb::I,
    Comb::Y,
    Comb::U,
    Comb::P,
    Comb::Plus,
    Comb::Minus,
    Comb::Times,
    Comb::Divide,
    Comb::Cond,
    Comb::Eq,
    Comb::Neq,
    Comb::Gt,
    Comb::Gte,
    Comb::Lt,
    Comb::Lte,
    Comb::And,
    Comb::Or,
    Comb::Not,
    Comb::Abort,
];

#[derive(Debug, Clone)]
enum CompiledExpr {
    Comb(Comb),
    Ap(Box<CompiledExpr>, Box<CompiledExpr>),
    Var(String),
    Int(i32),
}

impl Into<CompiledExpr> for Comb {
    fn into(self) -> CompiledExpr {
        CompiledExpr::Comb(self)
    }
}

fn cap<A: Into<CompiledExpr>, B: Into<CompiledExpr>>(a: A, b: B) -> CompiledExpr {
    CompiledExpr::Ap(Box::new(a.into()), Box::new(b.into()))
}

impl CompiledExpr {
    pub fn compile(e: &ast::Expr) -> CompiledExpr {
        match e {
            ast::Expr::Int(i) => CompiledExpr::Int(*i),
            ast::Expr::Var(s) => match s.as_str() {
                "if" => CompiledExpr::Comb(Comb::Cond),
                s => CompiledExpr::Var(s.to_owned()),
            },
            ast::Expr::BinOp(l, o, r) => {
                let op_comb = match o {
                    ast::BinOp::Cons => Comb::P,
                    ast::BinOp::Plus => Comb::Plus,
                    ast::BinOp::Minus => Comb::Minus,
                    ast::BinOp::Times => Comb::Times,
                    ast::BinOp::Divide => Comb::Divide,
                    ast::BinOp::Eq => Comb::Eq,
                    ast::BinOp::Neq => Comb::Neq,
                    ast::BinOp::Gt => Comb::Gt,
                    ast::BinOp::Gte => Comb::Gte,
                    ast::BinOp::Lt => Comb::Lt,
                    ast::BinOp::Lte => Comb::Lte,
                    ast::BinOp::And => Comb::And,
                    ast::BinOp::Or => Comb::Or,
                };
                let l = CompiledExpr::compile(l);
                let r = CompiledExpr::compile(r);
                cap(cap(op_comb, l), r)
            }
            ast::Expr::Not(e) => cap(Comb::Not, CompiledExpr::compile(e)),
            ast::Expr::Ap(l, r) => {
                let l = CompiledExpr::compile(l);
                let r = CompiledExpr::compile(r);
                cap(l, r)
            }
        }
    }

    pub fn abstract_var(self, n: &str) -> CompiledExpr {
        match self {
            CompiledExpr::Comb(c) => cap(Comb::K, c),
            CompiledExpr::Ap(l, r) => cap(l.abstract_var(n), r.abstract_var(n)),
            CompiledExpr::Var(s) => {
                if s == n {
                    CompiledExpr::Comb(Comb::I)
                } else {
                    cap(CompiledExpr::Comb(Comb::K), CompiledExpr::Var(s))
                }
            }
            i @ CompiledExpr::Int(_) => cap(CompiledExpr::Comb(Comb::K), i),
        }
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
}

impl TurnerEngine {
    pub fn compile(program: &ast::Program) -> TurnerEngine {
        let mut engine = TurnerEngine {
            tag: vec![0; CELLS],
            hd: vec![CellPtr(0i32); CELLS],
            tl: vec![CellPtr(0i32); CELLS],
            // We don't allocate cells when the pointer to it would be ambiguous
            next_cell: CellPtr(ALL_COMBS.len() as i32),
            def_lookup: HashMap::new(),
            stack: Vec::new(),
            dump_path: None,
            step_counter: 0,
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
            self.dump_dot().expect("Dump failed");
            let top = self.stack.last().unwrap();
            if let Some(comb) = top.comb() {
                match comb {
                    Comb::S => {
                        /*
                        // S f g x => f x (g x)
                        let l0 = self.stack[self.stack.len() - 2];
                        let l1 = self.stack[self.stack.len() - 3];
                        let l2 = self.stack[self.stack.len() - 4];
                        */
                        todo!()
                    }
                    Comb::K => {
                        // K x y = x
                        let l0 = self.stack[self.stack.len() - 2];
                        let l1 = self.stack[self.stack.len() - 3];
                        let x = self.tl[l0.0 as usize];
                        // If x is an int, we should transfer that to the new location
                        let x_tag = self.tag[l0.0 as usize] & TAG_RHS_INT;

                        // Make the indirection node
                        self.tag[l0.0 as usize] = x_tag;
                        self.hd[l0.0 as usize] = CellPtr(Comb::I as i32);
                        self.tl[l0.0 as usize] = x;

                        // Check if the value is an int, then we are done
                        if x_tag & TAG_RHS_INT != 0 {
                            return l0;
                        }

                        // Put x on the stack
                        let new_len = self.stack.len() - 2;
                        self.stack[new_len - 1] = x;
                        self.stack.truncate(new_len);
                    }
                    Comb::I => {
                        // Check if we are done
                        let l0 = self.stack[self.stack.len() - 2];
                        let tag0 = self.tag[l0.0 as usize];
                        if tag0 & TAG_RHS_INT != 0 {
                            // Indirection node!
                            return l0;
                        } else {
                            // Take the argument, and evaluate that instead
                            let arg = self.tl[l0.0 as usize];
                            // Replace two items with one arg
                            let new_len = self.stack.len() - 1;
                            self.stack[new_len - 1] = arg;
                            self.stack.truncate(new_len);
                        }
                    }
                    Comb::Y => todo!(),
                    Comb::U => todo!(),
                    Comb::P => todo!(),
                    Comb::Plus => todo!(),
                    Comb::Minus => todo!(),
                    Comb::Times => todo!(),
                    Comb::Divide => todo!(),
                    Comb::Cond => todo!(),
                    Comb::Eq => todo!(),
                    Comb::Neq => todo!(),
                    Comb::Gt => todo!(),
                    Comb::Gte => todo!(),
                    Comb::Lt => todo!(),
                    Comb::Lte => todo!(),
                    Comb::And => todo!(),
                    Comb::Or => todo!(),
                    Comb::Not => todo!(),
                    Comb::Abort => todo!(),
                }
            } else {
                // An application, so push the left subtree
                self.stack.push(self.hd[top.0 as usize]);
            }
        }
    }

    fn compile_def(&mut self, def: &ast::Def) {
        let mut compiled_expr = CompiledExpr::compile(&def.expr);
        for param in def.params.iter().rev() {
            compiled_expr = compiled_expr.abstract_var(param);
        }
        println!("Compiled {}: {:#?}", def.name, compiled_expr);
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
        while self.tag[cell_idx] & TAG_WANTED != 0 {
            cell_idx += 1;
        }
        self.tag[cell_idx] = tag;
        self.hd[cell_idx] = hd.into();
        self.tl[cell_idx] = tl.into();
        self.next_cell = CellPtr(cell_idx as i32 + 1);
        CellPtr(cell_idx as i32)
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

    fn set_tl<I: Into<CellPtr>>(&mut self, ptr: CellPtr, v: I) {
        self.tl[ptr.0 as usize] = v.into();
    }

    pub fn dump_dot(&self) -> std::io::Result<()> {
        let mut w = if let Some(dump_path) = &self.dump_path {
            let f = File::create(dump_path.join(format!("step{}.dot", self.step_counter)))?;
            BufWriter::new(f)
        } else {
            return Ok(());
        };
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    #[test]
    fn parse_factorial() {
        let program = r#"
(defun fac (n) 
  (if (= n 1)
      1
      (* n (fac (- n 1)))))
(defun main () (fac 5))
        "#;

        let parsed = parse(lex(program));
        let mut engine = TurnerEngine::compile(&parsed);

        let ptr = engine.run();

        assert_eq!(ptr, CellPtr(120));
    }

    #[test]
    fn test_id() {
        assert_runs_to_int(
            "test_id",
            r#"
(defun id (x) x)
(defun main () (id 42))
        "#,
            42,
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
        );
    }

    fn assert_runs_to_int(test_name: &str, program: &str, v: i32) {
        let parsed = parse(lex(program));
        let mut engine = TurnerEngine::compile(&parsed);
        engine.set_dump_path(format!("/tmp/{}", test_name));

        let ptr = engine.run();
        assert!(engine.tag[ptr.0 as usize] | TAG_RHS_INT != 0);
        assert_eq!(engine.tl[ptr.0 as usize], CellPtr(v));
    }
}
