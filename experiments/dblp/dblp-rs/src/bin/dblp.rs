use dblp_rs::entities::DblpUnescaper;
use dblp_rs::record::{Record, RecordType};
use flate2::bufread;
use quick_xml::events::Event;
use quick_xml::Reader;
use rusqlite::Connection;
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
    let mut record = Record::new();
    let mut state = ParseState::Initial;
    let unescaper = DblpUnescaper::new();

    let conn = Connection::open_in_memory().unwrap();

    conn.execute(
        r#"
        CREATE TABLE Dblp(
            record_type TEXT,
            key TEXT,
            title TEXT
        );
    "#,
        (),
    )
    .unwrap();
    let mut insert_stmt = conn.prepare("INSERT INTO Dblp VALUES (?, ?, ?)").unwrap();

    let mut do_insert = |record: &Record| {
        let record_type = match record.record_type {
            RecordType::InProceedings => "inproceedings",
            RecordType::Article => "article",
            RecordType::Book => "book",
            RecordType::Www => "www",
            RecordType::Proceedings => "proceedings",
            RecordType::Incollection => "incollection",
            RecordType::MastersThesis => "mastersthesis",
            RecordType::PhdThesis => "phdthesis",
        };
        insert_stmt
            .execute((record_type, &record.key, &record.title))
            .unwrap();
    };

    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => panic!("parse error: {e:?}"),
            Ok(Event::Decl(_)) => { /* ignored */ }
            Ok(Event::Eof) => break,
            Ok(Event::Text(e)) => match state {
                ParseState::Initial | ParseState::Dblp | ParseState::Record => {}
                ParseState::Title => {
                    let text = e.unescape_with(|s| unescaper.unescape(s)).unwrap();
                    record.title.push_str(&text);
                }
                ParseState::Author => {
                    let text = e.unescape_with(|s| unescaper.unescape(s)).unwrap();
                    record.author.last_mut().unwrap().push_str(&text);
                }
                ParseState::IgnoredField => {}
                ParseState::End => {}
            },
            Ok(Event::DocType(_)) => match state {
                ParseState::Initial => {}
                s => unreachable!("doctype in {s:?}"),
            },
            Ok(Event::Start(e)) => match state {
                ParseState::Initial => match e.name().as_ref() {
                    b"dblp" => {
                        state = ParseState::Dblp;
                    }
                    _ => unreachable!(),
                },
                ParseState::Dblp => {
                    let record_type = match e.name().as_ref() {
                        b"article" => RecordType::Article,
                        b"phdthesis" => RecordType::PhdThesis,
                        b"mastersthesis" => RecordType::MastersThesis,
                        b"book" => RecordType::Book,
                        b"incollection" => RecordType::Incollection,
                        b"proceedings" => RecordType::Proceedings,
                        b"www" => RecordType::Www,
                        b"inproceedings" => RecordType::InProceedings,
                        n => unimplemented!("start {:?}", std::str::from_utf8(n).unwrap()),
                    };
                    state = ParseState::Record;
                    record.record_type = record_type;

                    for attr in e.attributes() {
                        let attr = attr.unwrap();
                        match attr.key.as_ref() {
                            b"key" => {
                                record
                                    .key
                                    .push_str(std::str::from_utf8(&attr.value).unwrap());
                            }
                            _ => {}
                        }
                    }
                }
                ParseState::Record => match e.name().as_ref() {
                    b"title" => {
                        state = ParseState::Title;
                    }
                    b"author" => {
                        state = ParseState::Author;
                        record.author.push(String::new());
                    }
                    _ => {
                        state = ParseState::IgnoredField;
                    }
                },
                ParseState::Title => {
                    let tag = match e.name().as_ref() {
                        b"i" => "<i>",
                        b"sup" => "<sup>",
                        b"sub" => "<sub>",
                        b"tt" => "<tt>",
                        n => unreachable!("{:?} in title", std::str::from_utf8(n).unwrap()),
                    };
                    record.title.push_str(tag);
                }
                s => unimplemented!("start in {s:?}"),
            },
            Ok(Event::End(e)) => match state {
                ParseState::Title => {
                    if e.name().as_ref() == b"title" {
                        state = ParseState::Record;
                    } else {
                        let tag = match e.name().as_ref() {
                            b"i" => "</i>",
                            b"sup" => "</sup>",
                            b"sub" => "</sub>",
                            b"tt" => "</tt>",
                            n => unreachable!("{:?} in title", std::str::from_utf8(n).unwrap()),
                        };
                        record.title.push_str(tag);
                    }
                }
                ParseState::Record => {
                    match e.name().as_ref() {
                        b"article" | b"phdthesis" | b"mastersthesis" | b"book"
                        | b"incollection" | b"proceedings" | b"www" | b"inproceedings" => {}
                        n => unreachable!("end {:?} from record", std::str::from_utf8(n).unwrap()),
                    };
                    assert!(!record.key.is_empty());
                    do_insert(&record);
                    record.clear();
                    state = ParseState::Dblp;
                }
                ParseState::Author => {
                    state = ParseState::Record;
                }
                ParseState::IgnoredField => {
                    state = ParseState::Record;
                }
                ParseState::Dblp => {
                    state = ParseState::End;
                }
                s => todo!("end {s:?}"),
            },
            Ok(e) => todo!("{e:?}"),
        };
    }
    assert!(state == ParseState::End);

    std::mem::drop(insert_stmt);
    conn.close().unwrap();
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum ParseState {
    Initial,
    Dblp,
    Record,
    Title,
    Author,
    End,
    IgnoredField,
}
