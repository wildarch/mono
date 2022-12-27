use flate2::bufread;
use std::fs::File;
use std::io::BufReader;

fn main() {
    let mut args = std::env::args();
    let _binary = args.next().expect("No path to current binary");
    let src_path = args.next().expect("No source path");
    assert_eq!(args.next(), None, "Too many arguments");
    let src_file = File::open(src_path).expect("Error opening source path");
    let mut gzip_decoder = bufread::GzDecoder::new(BufReader::new(src_file));

    let mut dst_file = File::create("/dev/null").unwrap();

    std::io::copy(&mut gzip_decoder, &mut dst_file).unwrap();
}
