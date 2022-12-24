# Parsing DBLP, fast.
DBLP is a massive database of published works in Computer Science.
The data is exported as a 4GiB+ giant XML document, which is updated every month.

For my use case, I want fast access to individual records, so this is not a convenient format. 
The default DBLP parser parses the document into memory and calls it a day, but that seems like a waste of RAM to me.
Instead, let's make a tool to parse the XML and store it in an SQLite database.
Oh, and let's try and do it *as fast as possible*, because why not.

## Setting expectations 
I don't know how fast we should be able to do this, so let us put some bounds by building a really slow but correct version, and determine some theoretical limits we hit of we can parse the file infinitely fast.

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

The list of entries begins with a `<dblp>` tag, after which we find all the records in the database: various types of publications, `www` for personal websites, `person` for people and `data` (not sure what that is for). 

A record has fields such as `title` and `year`, one or more `author`s etc.

There are also attributes like `key` and `mdate` (modification date). 
Attributes vary, but fields are the same for all document types.

The `url` and `ee` fields are useful for finding more information about publications.
Some are local, in which case we should prefix `https://dblp.uni-trier.de/` to get a full URL.
`ee` is the more interesting of the two, pointing to an electronically available edition of the paper. `url` is mostly used to point to a table of contents for a journal or conference on DBLP.

The `<dblp>` element is closed again at the end of the document.
The [DTD file](https://dblp.org/xml/dblp.dtd) linked on the second line of the XML describes the structure of the elements inside the `dblp` tag.