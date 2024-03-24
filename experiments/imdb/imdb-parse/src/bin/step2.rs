use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
};

use flate2::bufread::GzDecoder;

#[derive(Debug, serde::Deserialize)]
struct Record {
    tconst: String,
    ordering: i32,
    nconst: String,
    category: String,
    job: String,
    characters: String,
}

impl Record {
    fn characters(&self) -> Vec<String> {
        serde_json::from_str(&self.characters).unwrap_or_default()
    }
}

fn main() -> std::io::Result<()> {
    // Parse args.
    let mut args = std::env::args();
    args.next().expect("No path to original executable");
    let path = args.next().expect("Expected path as argument");
    if args.next().is_some() {
        panic!("Expected only 1 argument");
    }

    // Add decompressor.
    let file = File::open(path)?;
    let file_reader = BufReader::new(file);
    let decompressed_file = GzDecoder::new(file_reader);

    // Configure reader for TSV.
    let mut tsv_reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_reader(decompressed_file);

    let mut characters_per_name: HashMap<String, HashSet<String>> = HashMap::new();

    for (i, record) in tsv_reader.deserialize().enumerate() {
        if i > 0 && i % 10_000 == 0 {
            println!("Processed {i} lines");
        }

        let record: Record = record?;
        let entry = characters_per_name
            .entry(record.nconst.clone())
            .or_default();
        entry.extend(record.characters());
    }

    // Final answer
    let mut max_char = 0usize;
    let mut max_name: Option<String> = None;

    for (name, chars) in characters_per_name.into_iter() {
        if chars.len() > max_char {
            max_char = chars.len();
            max_name = Some(name);
        }
    }

    println!(
        "Final answer: {} ({} characters)",
        max_name.expect("No max found"),
        max_char
    );

    Ok(())
}
