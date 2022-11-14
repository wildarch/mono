# DBLP to sqlite
The DBLP database is exported as a 4GiB+ giant XML document.
This is not a great format to operate on directly, so we'll parse it out to a sqlite database instead.

## Format
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

The `<dblp>` element is closed again all the way at the end of the document.
The [DTD file](https://dblp.org/xml/dblp.dtd) linked on the second line of the XML describes the structure of the elements inside the `dblp` tag.

Elements have attributes like key, modification date as well as fields (child elements). Attributes vary, but fields are the same for all document types.

There are also person types `author` and `editor`.

The `url` and `ee` fields are useful for finding more information about publications.
Some are local, in which case we should prefix `https://dblp.uni-trier.de/` to get a full URL.
`ee` is the more interesting of the two, pointing to an electronically available edition of the paper. `url` is mostly used to point to a table of contents for a journal or conference on DBLP.

## Python prototype
I built a python prototype in `parse.py`. 
It extracts and dumps the rows in 4m43s.
The resulting sqlite DB is 2.1G.

TODO: Can have multiple author, editor, cite children,

## Performance analysis
We first check how long it takes to decompress the file:

```shell
time gunzip -c ~/Downloads/dblp-2022-11-02.xml.gz > /dev/null
real    0m13.623s
user    0m13.338s
sys     0m0.281s
```

If we tack on a simple expat parser that does almost nothing, we get:

```shell
time gunzip -c ~/Downloads/dblp-2022-11-02.xml.gz | bazel-bin/experiments/dblp/parse_expat

real    0m25.421s
user    0m41.390s
sys     0m1.727s
```

A simple Java parser runs slower still:

```shell
time bazel run -c opt //src/main/java/dev/wildarch/experiments/dblp:DblpParser ~/Downloads/dblp-2022-11-02.xml.gz ~/Downloads/dblp-2019-11-22.dtd
INFO: Analyzed target //src/main/java/dev/wildarch/experiments/dblp:DblpParser (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //src/main/java/dev/wildarch/experiments/dblp:DblpParser up-to-date:
  bazel-bin/src/main/java/dev/wildarch/experiments/dblp/DblpParser.jar
  bazel-bin/src/main/java/dev/wildarch/experiments/dblp/DblpParser
INFO: Elapsed time: 0.142s, Critical Path: 0.00s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
INFO: Running command line: bazel-bin/src/main/java/dev/wildarch/experiments/dblp/DblpParser /home/daan/Downloads/dblp-2022
INFO: Build completed successfully, 1 total action
Hello, world!

real    0m41.991s
user    0m44.495s
sys     0m0.830s
```

This program handles Gzip decompression itself, so it might be doing decompression and parsing on the same thread, which is not a completely fair comparison.
It also correctly handles entity resolution and DTD parsing, which the C version does not.

Let's try again with gzip decompression handled separately:
```shell
time gunzip -c ~/Downloads/dblp-2022-11-02.xml.gz | bazel-bin/src/main/java/dev/wildarch/experiments/dblp/DblpParser ~/Downloads/dblp-2019-11-22.dtd 
Hello, world!

real    0m33.187s
user    0m57.688s
sys     0m2.188s
```

Not bad at all, only a little slower than the C version!

Next we'll try using the woodstox parser for Java, which supposedly is better?

```shell
time gunzip -c ~/Downloads/dblp-2022-11-02.xml.gz | bazel-bin/src/main/java/dev/wildarch/experiments/dblp/DblpParser ~/Downloads/dblp-2019-11-22.dtd 

real    0m34.704s
user    0m59.942s
sys     0m3.275s
```

Basically the same..

## Notes
Trying to build a fast DBLP XML parser.

Will use Expat, seems to be a very fast SAX parser.
Getting started: https://www.xml.com/pub/1999/09/expat/index.html.

Need external entity resolver (DTD). See here https://libexpat.github.io/doc/api/latest/#XML_ExternalEntityParserCreate.

A simple example: https://github.com/libexpat/libexpat/blob/master/expat/examples/elements.c

### State machine
START -> ARTICLE -> FIELD