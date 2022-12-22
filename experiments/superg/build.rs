use std::fs::File;
use std::io::Write;

// Generate bulk combinators Sn,Bn and Cn up to this n
const BULK_COMB_MAX_N: usize = 10;

fn main() -> std::io::Result<()> {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // This script depends only on itself
    println!("cargo:rerun-if-changed=build.rs");

    let combgen = std::fs::File::create(&format!("{}/combgen.rs", out_dir))?;
    generate_combgen(combgen)?;

    Ok(())
}

fn generate_combgen(mut f: File) -> std::io::Result<()> {
    let max_combn = std::cmp::max(
        // Regular combinators use up to 3 arguments.
        3,
        // Bulk combinator take two functions f and g and distribute n arguments over them.
        2 + BULK_COMB_MAX_N,
    );

    writeln!(f, "use super::*;")?;
    writeln!(f, "impl TurnerEngine {{")?;

    for i in 1..=max_combn {
        writeln!(f, "    macros::run_comb!(run_comb{i}, {i});")?;
    }

    for i in 2..=BULK_COMB_MAX_N {
        writeln!(
            f,
            "    macros::run_sn_comb!(run_s{}_comb, run_comb{});",
            i,
            i + 2
        )?;
        writeln!(
            f,
            "    macros::run_bn_comb!(run_b{}_comb, run_comb{});",
            i,
            i + 2
        )?;
        writeln!(
            f,
            "    macros::run_cn_comb!(run_c{}_comb, run_comb{});",
            i,
            i + 2
        )?;
    }

    writeln!(f, "}}")?;

    // Implementation mapping function
    write!(
        f,
        r#"
impl Comb {{
    pub(super) fn implementation(&self) -> fn(&mut TurnerEngine) -> Option<CellPtr> {{
        match self {{
            Comb::S => TurnerEngine::run_s_comb,
            Comb::K => TurnerEngine::run_k_comb,
            Comb::I => TurnerEngine::run_i_comb,
            Comb::B => TurnerEngine::run_b_comb,
            Comb::C => TurnerEngine::run_c_comb,
            Comb::Plus => TurnerEngine::run_plus_comb,
            Comb::Minus => TurnerEngine::run_minus_comb,
            Comb::Times => TurnerEngine::run_times_comb,
            Comb::Cond => TurnerEngine::run_cond_comb,
            Comb::Eq => TurnerEngine::run_eq_comb,
            Comb::Lt => TurnerEngine::run_lt_comb,
            Comb::Abort => TurnerEngine::run_abort_comb,
        "#
    )?;
    for i in 2..=BULK_COMB_MAX_N {
        write!(
            f,
            r#"
            Comb::Sn({i}) => TurnerEngine::run_s{i}_comb,
            Comb::Bn({i}) => TurnerEngine::run_b{i}_comb,
            Comb::Cn({i}) => TurnerEngine::run_c{i}_comb,
            "#
        )?;
    }
    write!(
        f,
        r#"
            _ => todo!("No implementation for combinator {{:?}}", self),
        }}
        "#
    )?;
    writeln!(f, "    }}")?;
    writeln!(f, "}}")?;
    Ok(())
}
