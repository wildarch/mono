# Parsing DBLP, fast.
DBLP is a massive database of published works in Computer Science.
The data is exported as a 4GiB+ giant XML document, which is updated every month.

For my use case, I want fast access to individual records, so this is not a convenient format. 
The default DBLP parser parses the document into memory and calls it a day, but that seems like a waste of RAM to me.
Instead, let's make a tool to parse the XML and store it in an SQLite database.
Oh, and let's try and do it *as fast as possible*, because why not.

## Setting expectations 
I don't know how fast we should be able to do this, so let us determine some theoretical limits we hit if we can parse the file infinitely fast.

### Network
Our first theoretical limit is how quickly we can fetch the export file from the remote server. On a VPS with a fast connection, I measured 22 seconds to download the 712MB file (about 32MB/s).

### GZIP Decompression
The DBLP files come gzipped, so let us check how long plain gzip takes to decompress the file:

```shell
$ time gzip --decompress --stdout dblp-2022-12-01.xml.gz > /dev/null

real	0m16.765s
user	0m16.667s
sys	0m0.096s
```

That is about 42MB/s, I am not surprised the network is the bottleneck here.
As far as I understand it, plain gzip cannot be decompressed in parallel, decompression is single-threaded and sequential.

### SQLite insert
Let us not concern ourselves with the data format too much, and just look at raw SQLite insert speed for a simple schema:

```sql
CREATE TABLE Test(
    key TEXT,
    title TEXT,
);
```

We will use an in-memory database to prevent the disk from becoming a bottleneck.
According to the website, there are 6,444,017 publications in DBLP at the time of writing this, so let's see how fast SQLite can import a dataset that large.

This little python script creates a dummy CSV file with the right number of records:

```python
RECORDS = 6_444_017
print("key,title")
for i in range(RECORDS):
    print(f"key{i},title{i}")
```

We can import it in SQLite like so:

```shell
$ time echo '.quit' | sqlite3 -cmd ".import /tmp/dbdblp_dummy.csv test"

real    0m6.469s
user    0m6.334s
sys     0m0.134s
```

Quite an impressive score! 
It seems unlikely SQLite will be a bottleneck in the process.

## Format
There is good documentation out there for the [DBLP format](https://dblp.org/xml/docu/dblpxml.pdf).
Here are the first few lines of the XML dump:

```xml
<?xml version="1.0" encoding="ISO-8859-1"?>
<!DOCTYPE dblp SYSTEM "dblp-2019-11-22.dtd">
<dblp>
<article mdate="2017-06-08" key="dblpnote/error" publtype="informal">
<title>(error)</title>
</article><article mdate="2017-06-08" key="dblpnote/ellipsis" publtype="informal">
<title>&#8230;</title>
</article><article mdate="2017-06-08" key="dblpnote/neverpublished" publtype="informal">
<title>(was never published)</title>
</article><phdthesis mdate="2002-01-03" key="phd/Turpin92">
<author>Russell Turpin</author>
<title>Programming Data Structures in Logic.</title>
<year>1992</year>
<school>University of Texas, Austin</school>
</phdthesis>
```

The list of entries begins with a `<dblp>` tag, after which we find all the records in the database: various types of publications, `www` for personal websites, `person` for people and `data` (not sure what that is for). 

A record has fields such as `title` and `year`, one or more `author`s etc.

There are also attributes like `key` and `mdate` (modification date). 
Attributes vary, but fields are the same for all document types.

The `url` and `ee` fields are useful for finding more information about publications.
Some are local, in which case we should prefix `https://dblp.uni-trier.de/` to get a full URL.
`ee` is the more interesting of the two, pointing to an electronically available edition of the paper. `url` is mostly used to point to a table of contents for a journal or conference on DBLP.

The `<dblp>` element is closed again at the end of the document.
The [DTD file](https://dblp.org/xml/dblp.dtd) linked on the second line of the XML describes the structure of the elements inside the `dblp` tag.

### Schema analysis
To figure out a suitable mapping to SQL, we will write a script to analyse the XML file and see what columns will be in our schema.
Another important thing to see is what fields occur multiple times within a publication.

Python has a built-in XML parser, let's try that first:

```python
#!/usr/bin/env python3
import sys
import gzip
import xml.etree.ElementTree as ET

with gzip.open(sys.argv[1], 'rb') as gzf:
    for event, elem in ET.iterparse(gzf, events=("end",)):
        # Remember to clear or the parser will leak memory
        elem.clear()
```

Unfortunately, it fails to run:

```shell
./experiments/dblp/analyse_schema.py ~/Downloads/dblp-2022-11-02.xml.gz
Traceback (most recent call last):
  File "/home/daan/workspace/mono/./experiments/dblp/analyse_schema.py", line 7, in <module>
    for event, elem in ET.iterparse(gzf, events=("end",)):
  File "/usr/lib/python3.9/xml/etree/ElementTree.py", line 1254, in iterator
    yield from pullparser.read_events()
  File "/usr/lib/python3.9/xml/etree/ElementTree.py", line 1329, in read_events
    raise event
  File "/usr/lib/python3.9/xml/etree/ElementTree.py", line 1301, in feed
    self._parser.feed(data)
xml.etree.ElementTree.ParseError: undefined entity &ograve;: line 244, column 22
```

Definitions for this entity and others are in the DTD file that comes with the export.
It appears the built-in XML parser based on expat does not support them. 
I found the easiest solution is to use the parser based on lxml.
That library allows loading the DTD with a simple `load_dtd` parameter:

```python
#!/usr/bin/env python3
import sys
import gzip
from lxml import etree

with gzip.open(sys.argv[1], 'rb') as gzf:
    for event, elem in etree.iterparse(gzf, events=("end",), load_dtd=True):
        # Remember to clear or the parser will leak memory
        elem.clear()
```

Much better! 
The full script for the schema analysis is available at `analyse_schema.py`.
Important take-aways from that are:
* Performance of the python script is quite terrible, it takes a few minutes to complete.
* Record types are 'inproceedings', 'article', 'book', 'www', 'proceedings', 'incollection', 'mastersthesis', 'phdthesis'
* Fields are: 'school', 'author', 'journal', 'isbn', 'editor', 'url', 'publisher', 'chapter', 'booktitle', 'note', 'ee', 'cdrom', 'cite', 'volume', 'pages', 'publnr', 'month', 'crossref', 'year', 'number', 'series', 'address', 'title'
* Top-level attributes: 'publtype', 'cdate', 'key', 'mdate'
* Some fields are often repeated: 'ee': 1236657, 'school': 903, 'author': 13894347, 'isbn': 9542, 'note': 40924, 'editor': 90333, 'cite': 164365, 'url': 187470, 'cdrom': 408
* Some fields are seen repeated, but rarely: 'pages': 7, 'title': 1, 'publisher': 3, 'crossref': 1, 'series': 1, 'year': 1. It seems likely to me these are mistakes, so we'll ignore them.

### A fast parser (attempt 1)
I figured a Rust implementation would be pretty fast, so I built a little program to read in the gzipped file using `flate2`:

```rust
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

```

This is essentially identical to our initial plain gzip test, and it turns out it is even slightly faster:

```shell
 time cargo run --bin gzipread --release ~/Downloads/dblp-2022-11-02.xml.gz
   Compiling dblp-rs v0.1.0 (/home/daan/workspace/mono/experiments/dblp/dblp-rs)
    Finished release [optimized] target(s) in 0.50s
     Running `/home/daan/workspace/mono/target/release/gzipread /home/daan/Downloads/dblp-2022-11-02.xml.gz`

real    0m7.297s
user    0m7.517s
sys     0m0.427s
```

Next, we plug in an XML parser, it seems `xml-rs` is quite popular:

```rust
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
```

DBLP uses a set of custom entities for non-ASCII characters. I didn't see any Rust libraries with support for loading the DTD as we did in python, so I had to hardcode their mappings. 
The mappings seem fairly stable, so I am okay with that.
Sadly, performance takes a massive hit:

```shell
$ time cargo run --bin xmlparse --release ~/Downloads/dblp-2022-11-02.xml.gz
    Finished release [optimized] target(s) in 0.02s
     Running `/home/daan/workspace/mono/target/release/xmlparse /home/daan/Downloads/dblp-2022-11-02.xml.gz`

real    5m43.686s
user    5m43.335s
sys     0m0.312s
```

### A fast parser (attempt 2)
Maybe another library will be faster, let's so if `quick-xml` lives up to its reputation:

```rust
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
```

Much better:

```shell
$ time cargo run --bin quickxmlparse --release ~/Downloads/dblp-2022-11-02.xml.gz
    Finished release [optimized] target(s) in 0.03s
     Running `/home/daan/workspace/mono/target/release/quickxmlparse /home/daan/Downloads/dblp-2022-11-02.xml.gz`

real    0m17.372s
user    0m16.311s
sys     0m1.054s
```

That is more than fast enough to keep up with network delay, and all of this is still running on a single thread!
One caveat is that we still need to decode the custom entities, but that should hopefully not add too much to the running time.