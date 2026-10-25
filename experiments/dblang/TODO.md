- 0x1.5 lexes as INT 0x1 then FLOAT .5, and x.5 as IDENT x then FLOAT .5. 
  The parser will reject both, but the error messages may be confusing.
- Columns count bytes, so locations after a non-ASCII character are shifted by the extra bytes.