use flate2::bufread;
use std::fs::File;
use std::io::BufReader;
use xml::ParserConfig;

fn main() {
    let mut args = std::env::args();
    let _binary = args.next().expect("No path to current binary");
    let src_path = args.next().expect("No source path");
    assert_eq!(args.next(), None, "Too many arguments");
    let src_file = File::open(src_path).expect("Error opening source path");
    let gzip_decoder = bufread::GzDecoder::new(BufReader::new(src_file));

    let mut parser_config = ParserConfig::new();
    parser_config = configure_entities(parser_config);
    let parser = parser_config.create_reader(gzip_decoder);

    for event in parser {
        let _ = event.unwrap();
    }
}

fn configure_entities(mut config: ParserConfig) -> ParserConfig {
    for (name, value) in dblp_rs::dblp_mapping() {
        config = config.add_entity(name, value);
    }
    config
}
