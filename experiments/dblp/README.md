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

## Python prototype

## Notes
Trying to build a fast DBLP XML parser.

Will use Expat, seems to be a very fast SAX parser.
Getting started: https://www.xml.com/pub/1999/09/expat/index.html.

Need external entity resolver (DTD). See here https://libexpat.github.io/doc/api/latest/#XML_ExternalEntityParserCreate.

A simple example: https://github.com/libexpat/libexpat/blob/master/expat/examples/elements.c

### State machine
START -> ARTICLE -> FIELD