# Seekable gzip files
Regular gzip files do not allow for seeking into some part and decoding only that part: files must be decoded sequentially. 
Why is that?

## Background
gzip uses the [DEFLATE algorithm](https://www.zlib.net/feldspar.html) for compression.
The gzip format is essentially just a header, a DEFLATE compressed payload, and a trailer.

A nice intro to the zlib library is here: https://www.zlib.net/zlib_how.html.