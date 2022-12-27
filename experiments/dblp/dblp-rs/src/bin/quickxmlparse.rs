use flate2::bufread;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs::File;
use std::io::BufReader;

fn main() {
    let mut args = std::env::args();
    let _binary = args.next().expect("No path to current binary");
    let src_path = args.next().expect("No source path");
    assert_eq!(args.next(), None, "Too many arguments");
    let src_file = File::open(src_path).expect("Error opening source path");
    let gzip_decoder = bufread::GzDecoder::new(BufReader::new(src_file));

    let mut reader = Reader::from_reader(BufReader::new(gzip_decoder));
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => panic!("Parse error: {}", e),
            Ok(Event::Eof) => break,
            Ok(_) => {}
        }
    }
}
