use std::fmt;

use crate::ir::{InFilePos, Loc, SourceFileRef};

macro_rules! token_kinds {
    (
        other { $($kind:ident => $name:literal,)* }
        keywords { $($kw:ident => $kw_name:literal, $kw_str:literal,)* }
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum TokenKind {
            $($kind,)*
            $($kw,)*
        }

        impl TokenKind {
            /// Name of the token kind, as used in debug and test output.
            pub fn name(self) -> &'static str {
                match self {
                    $(TokenKind::$kind => $name,)*
                    $(TokenKind::$kw => $kw_name,)*
                }
            }

            /// The keyword token kind for `s`, if `s` is a keyword.
            pub fn keyword(s: &str) -> Option<TokenKind> {
                match s {
                    $($kw_str => Some(TokenKind::$kw),)*
                    _ => None,
                }
            }
        }
    };
}

token_kinds! {
    other {
        Ident => "IDENT",
        Int => "INT",
        Float => "FLOAT",
        String => "STRING",
        Char => "CHAR",
        // Punctuation
        LParen => "LPAREN",         // (
        RParen => "RPAREN",         // )
        LBrace => "LBRACE",         // {
        RBrace => "RBRACE",         // }
        LSBracket => "LSBRACKET",   // [
        RSBracket => "RSBRACKET",   // ]
        LAngle => "LANGLE",         // <
        RAngle => "RANGLE",         // >
        Colon => "COLON",           // :
        DColon => "DCOLON",         // ::
        Comma => "COMMA",           // ,
        Dot => "DOT",               // .
        Semi => "SEMI",             // ;
        // Operators
        Plus => "PLUS",             // +
        Minus => "MINUS",           // -
        Asterisk => "ASTERISK",     // *
        Slash => "SLASH",           // /
        Percent => "PERCENT",       // %
        Exclamation => "EXCLAMATION", // !
        Tilde => "TILDE",           // ~
        Ampersand => "AMPERSAND",   // &
        Pipe => "PIPE",             // |
        Caret => "CARET",           // ^
        QMark => "QMARK",           // ?
        Arrow => "ARROW",           // ->
        DPlus => "DPLUS",           // ++
        DMinus => "DMINUS",         // --
        DAmpersand => "DAMPERSAND", // &&
        DPipe => "DPIPE",           // ||
        Equal => "EQUAL",           // ==
        NotEqual => "NOT_EQUAL",    // !=
        Leq => "LEQ",               // <=
        Geq => "GEQ",               // >=
        LShift => "LSHIFT",         // <<
        RShift => "RSHIFT",         // >>
        // Assignment
        Assign => "ASSIGN",                    // =
        AsteriskAssign => "ASTERISK_ASSIGN",   // *=
        SlashAssign => "SLASH_ASSIGN",         // /=
        PercentAssign => "PERCENT_ASSIGN",     // %=
        PlusAssign => "PLUS_ASSIGN",           // +=
        MinusAssign => "MINUS_ASSIGN",         // -=
        LShiftAssign => "LSHIFT_ASSIGN",       // <<=
        RShiftAssign => "RSHIFT_ASSIGN",       // >>=
        AmpersandAssign => "AMPERSAND_ASSIGN", // &=
        CaretAssign => "CARET_ASSIGN",         // ^=
        PipeAssign => "PIPE_ASSIGN",           // |=
    }
    keywords {
        Alias => "ALIAS", "alias",
        Alignof => "ALIGNOF", "alignof",
        Bool => "BOOL", "bool",
        Break => "BREAK", "break",
        Case => "CASE", "case",
        Cast => "CAST", "cast",
        CharKw => "CHAR_KW", "char",
        Const => "CONST", "const",
        Continue => "CONTINUE", "continue",
        Def => "DEF", "def",
        Default => "DEFAULT", "default",
        Do => "DO", "do",
        Else => "ELSE", "else",
        Enum => "ENUM", "enum",
        False => "FALSE", "false",
        For => "FOR", "for",
        Func => "FUNC", "func",
        Global => "GLOBAL", "global",
        I8 => "I8", "i8",
        I16 => "I16", "i16",
        I32 => "I32", "i32",
        I64 => "I64", "i64",
        I128 => "I128", "i128",
        If => "IF", "if",
        Import => "IMPORT", "import",
        Isize => "ISIZE", "isize",
        Let => "LET", "let",
        Return => "RETURN", "return",
        Sizeof => "SIZEOF", "sizeof",
        Struct => "STRUCT", "struct",
        Switch => "SWITCH", "switch",
        True => "TRUE", "true",
        U8 => "U8", "u8",
        U16 => "U16", "u16",
        U32 => "U32", "u32",
        U64 => "U64", "u64",
        U128 => "U128", "u128",
        Usize => "USIZE", "usize",
        Void => "VOID", "void",
        While => "WHILE", "while",
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Token<'a> {
    pub loc: Loc,
    pub kind: TokenKind,
    /// The source text of the token. For string and character literals this
    /// excludes the quotes (escape sequences are not processed).
    pub body: &'a str,
}

/// An error encountered while lexing.
#[derive(Clone, Debug)]
pub struct LexError {
    pub loc: Loc,
    pub message: String,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LexError {}

type LexResult<T> = Result<T, LexError>;

/// Lexes `source`, appending tokens to `tokens`. `start` is the position of
/// the first character of `source` within the source file.
///
/// Lexing stops at the first error; tokens lexed before it remain in `tokens`.
pub fn lex<'a>(
    source: &'a str,
    source_file: SourceFileRef,
    start: InFilePos,
    tokens: &mut Vec<Token<'a>>,
) -> LexResult<()> {
    let mut lexer = Lexer {
        source,
        bytes: source.as_bytes(),
        source_file,
        offset: 0,
        pos: start,
    };

    while let Some(token) = lexer.next_token()? {
        tokens.push(token);
    }

    Ok(())
}

fn is_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}
fn is_digit(c: u8) -> bool {
    c.is_ascii_digit()
}
fn is_hex(c: u8) -> bool {
    c.is_ascii_hexdigit()
}
fn is_binary(c: u8) -> bool {
    c == b'0' || c == b'1'
}
fn is_identifier_start(c: u8) -> bool {
    is_alpha(c) || c == b'_'
}
fn is_identifier_continue(c: u8) -> bool {
    is_identifier_start(c) || is_digit(c)
}

/// Prints a character for use in an error message.
fn describe_char(c: u8) -> String {
    if (0x20..0x7F).contains(&c) {
        format!("'{}'", c as char)
    } else {
        format!("byte 0x{c:02X}")
    }
}

const THREE_CHAR_PUNCT: &[(&str, TokenKind)] = &[
    ("<<=", TokenKind::LShiftAssign),
    (">>=", TokenKind::RShiftAssign),
];

const TWO_CHAR_PUNCT: &[(&str, TokenKind)] = &[
    ("::", TokenKind::DColon),
    ("->", TokenKind::Arrow),
    ("++", TokenKind::DPlus),
    ("--", TokenKind::DMinus),
    ("&&", TokenKind::DAmpersand),
    ("||", TokenKind::DPipe),
    ("==", TokenKind::Equal),
    ("!=", TokenKind::NotEqual),
    ("<=", TokenKind::Leq),
    (">=", TokenKind::Geq),
    ("<<", TokenKind::LShift),
    (">>", TokenKind::RShift),
    ("*=", TokenKind::AsteriskAssign),
    ("/=", TokenKind::SlashAssign),
    ("%=", TokenKind::PercentAssign),
    ("+=", TokenKind::PlusAssign),
    ("-=", TokenKind::MinusAssign),
    ("&=", TokenKind::AmpersandAssign),
    ("^=", TokenKind::CaretAssign),
    ("|=", TokenKind::PipeAssign),
];

fn one_char_punct(c: u8) -> Option<TokenKind> {
    Some(match c {
        b'(' => TokenKind::LParen,
        b')' => TokenKind::RParen,
        b'{' => TokenKind::LBrace,
        b'}' => TokenKind::RBrace,
        b'[' => TokenKind::LSBracket,
        b']' => TokenKind::RSBracket,
        b'<' => TokenKind::LAngle,
        b'>' => TokenKind::RAngle,
        b':' => TokenKind::Colon,
        b',' => TokenKind::Comma,
        b'.' => TokenKind::Dot,
        b';' => TokenKind::Semi,
        b'+' => TokenKind::Plus,
        b'-' => TokenKind::Minus,
        b'*' => TokenKind::Asterisk,
        b'/' => TokenKind::Slash,
        b'%' => TokenKind::Percent,
        b'=' => TokenKind::Assign,
        b'!' => TokenKind::Exclamation,
        b'~' => TokenKind::Tilde,
        b'&' => TokenKind::Ampersand,
        b'|' => TokenKind::Pipe,
        b'^' => TokenKind::Caret,
        b'?' => TokenKind::QMark,
        _ => return None,
    })
}

struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    source_file: SourceFileRef,
    offset: usize,
    pos: InFilePos,
}

impl<'a> Lexer<'a> {
    fn cur(&self) -> Option<u8> {
        self.bytes.get(self.offset).copied()
    }

    /// The character n positions ahead of the current one.
    fn lookahead(&self, n: usize) -> Option<u8> {
        self.bytes.get(self.offset + n).copied()
    }

    fn starts_with(&self, s: &str) -> bool {
        self.bytes[self.offset..].starts_with(s.as_bytes())
    }

    fn eat(&mut self) {
        if self.cur() == Some(b'\n') {
            self.pos.column = 1;
            self.pos.line += 1;
        } else {
            self.pos.column += 1;
        }

        self.offset += 1;
    }

    fn try_eat(&mut self, c: u8) -> bool {
        if self.cur() == Some(c) {
            self.eat();
            true
        } else {
            false
        }
    }

    fn loc_from(&self, start: InFilePos) -> Loc {
        Loc {
            source_file: self.source_file,
            start,
            end: self.pos,
        }
    }

    fn error<T>(&self, start: InFilePos, message: impl Into<String>) -> LexResult<T> {
        Err(LexError {
            loc: self.loc_from(start),
            message: message.into(),
        })
    }

    fn token(&self, start: InFilePos, start_offset: usize, kind: TokenKind) -> Token<'a> {
        Token {
            loc: self.loc_from(start),
            kind,
            body: &self.source[start_offset..self.offset],
        }
    }

    fn eat_whitespace(&mut self) -> LexResult<()> {
        while let Some(c) = self.cur() {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.eat();
            } else if self.starts_with("//") {
                // line comment
                while self.cur().is_some_and(|c| c != b'\n') {
                    self.eat();
                }
            } else if self.starts_with("/*") {
                /* multi-line comment */
                let start = self.pos;
                self.eat();
                self.eat();
                loop {
                    if self.cur().is_none() {
                        return self.error(start, "unterminated block comment");
                    } else if self.starts_with("*/") {
                        self.eat();
                        self.eat();
                        break;
                    } else {
                        self.eat();
                    }
                }
            } else {
                // No (more) whitespace
                break;
            }
        }

        Ok(())
    }

    /// Returns the next token, or `None` at the end of the input.
    fn next_token(&mut self) -> LexResult<Option<Token<'a>>> {
        self.eat_whitespace()?;

        let Some(c) = self.cur() else {
            return Ok(None);
        };

        if is_identifier_start(c) {
            return Ok(Some(self.lex_ident()));
        } else if is_digit(c) || (c == b'.' && self.lookahead(1).is_some_and(is_digit)) {
            return self.lex_number().map(Some);
        } else if c == b'\'' {
            return self.lex_char().map(Some);
        } else if c == b'"' {
            return self.lex_string().map(Some);
        }

        // Punctuators are matched longest first.
        let start = self.pos;
        let start_offset = self.offset;
        let multi = THREE_CHAR_PUNCT.iter().chain(TWO_CHAR_PUNCT);
        for &(punct, kind) in multi {
            if self.starts_with(punct) {
                for _ in 0..punct.len() {
                    self.eat();
                }

                return Ok(Some(self.token(start, start_offset, kind)));
            }
        }

        if let Some(kind) = one_char_punct(c) {
            self.eat();
            return Ok(Some(self.token(start, start_offset, kind)));
        }

        // Report the location of just the offending character.
        Err(LexError {
            loc: Loc {
                source_file: self.source_file,
                start,
                end: InFilePos {
                    line: start.line,
                    column: start.column + 1,
                },
            },
            message: format!("unrecognized input: {}", describe_char(c)),
        })
    }

    fn lex_ident(&mut self) -> Token<'a> {
        debug_assert!(self.cur().is_some_and(is_identifier_start));
        let start = self.pos;
        let start_offset = self.offset;
        while self.cur().is_some_and(is_identifier_continue) {
            self.eat();
        }

        let body = &self.source[start_offset..self.offset];
        // Check if we have a keyword instead of an identifier.
        let kind = TokenKind::keyword(body).unwrap_or(TokenKind::Ident);
        self.token(start, start_offset, kind)
    }

    /// Eats a sequence of digits, optionally separated by single underscores.
    /// Returns false if there is no digit at the current position.
    fn eat_digit_seq(&mut self, is_digit_fn: fn(u8) -> bool) -> bool {
        if !self.cur().is_some_and(is_digit_fn) {
            return false;
        }

        self.eat();
        while let Some(c) = self.cur() {
            if is_digit_fn(c) {
                self.eat();
            } else if c == b'_' && self.lookahead(1).is_some_and(is_digit_fn) {
                self.eat(); // the '_'
                self.eat();
            } else {
                break;
            }
        }

        true
    }

    fn lex_number(&mut self) -> LexResult<Token<'a>> {
        let start = self.pos;
        let start_offset = self.offset;

        let mut kind = TokenKind::Int;
        if self.starts_with("0x") || self.starts_with("0X") {
            // hex notation, e.g. 0xDEADBEEF
            self.eat();
            self.eat();
            if !self.eat_digit_seq(is_hex) {
                return self.error(start, "expected hex digit after '0x'");
            }
        } else if self.starts_with("0b") || self.starts_with("0B") {
            // binary notation, e.g. 0b1010
            self.eat();
            self.eat();
            if !self.eat_digit_seq(is_binary) {
                return self.error(start, "expected binary digit after '0b'");
            }
        } else {
            let has_int_part = self.eat_digit_seq(is_digit);
            if self.try_eat(b'.') {
                // float notation, e.g. 3.14, 1. or .5
                kind = TokenKind::Float;
                let has_frac_part = self.eat_digit_seq(is_digit);
                debug_assert!(has_int_part || has_frac_part);
            }

            if matches!(self.cur(), Some(b'e' | b'E')) {
                // exponent, e.g. 1e9
                kind = TokenKind::Float;
                self.eat();
                if matches!(self.cur(), Some(b'+' | b'-')) {
                    self.eat();
                }

                if !self.eat_digit_seq(is_digit) {
                    return self.error(start, "expected digit in exponent");
                }
            }

            if kind == TokenKind::Int
                && self.bytes[start_offset] == b'0'
                && self.offset - start_offset > 1
            {
                return self.error(start, "decimal integer literal cannot start with '0'");
            }
        }

        if let Some(c) = self.cur().filter(|&c| is_identifier_continue(c)) {
            self.eat();
            return self.error(
                start,
                format!("invalid character {} in numeric literal", describe_char(c)),
            );
        }

        Ok(self.token(start, start_offset, kind))
    }

    fn eat_escape_seq(&mut self) -> LexResult<()> {
        debug_assert_eq!(self.cur(), Some(b'\\'));
        let start = self.pos;
        self.eat();
        let c = match self.cur() {
            // Reported by the caller as unterminated.
            None | Some(b'\n') => return Ok(()),
            Some(c) => c,
        };

        match c {
            b'\'' | b'"' | b'?' | b'\\' | b'a' | b'b' | b'f' | b'n' | b'r' | b't' | b'v' => {
                self.eat();
                Ok(())
            }
            b'x' => {
                self.eat();
                if !self.cur().is_some_and(is_hex) {
                    return self.error(start, "expected hex digit in escape sequence");
                }

                while self.cur().is_some_and(is_hex) {
                    self.eat();
                }

                Ok(())
            }
            _ => {
                self.eat();
                self.error(
                    start,
                    format!("invalid escape sequence: {}", describe_char(c)),
                )
            }
        }
    }

    fn lex_char(&mut self) -> LexResult<Token<'a>> {
        debug_assert_eq!(self.cur(), Some(b'\''));
        let start = self.pos;
        self.eat();
        let body_start = self.offset;

        match self.cur() {
            None | Some(b'\n') => {
                return self.error(start, "unterminated character literal");
            }
            Some(b'\'') => {
                self.eat();
                return self.error(start, "empty character literal");
            }
            Some(b'\\') => self.eat_escape_seq()?,
            Some(c) if !c.is_ascii() => {
                self.eat();
                return self.error(start, "non-ASCII character in character literal");
            }
            Some(_) => self.eat(),
        }

        let body_end = self.offset;
        if !self.try_eat(b'\'') {
            if matches!(self.cur(), None | Some(b'\n')) {
                return self.error(start, "unterminated character literal");
            }

            self.eat();
            return self.error(
                start,
                "character literal must contain exactly one character",
            );
        }

        Ok(Token {
            loc: self.loc_from(start),
            kind: TokenKind::Char,
            body: &self.source[body_start..body_end],
        })
    }

    fn lex_string(&mut self) -> LexResult<Token<'a>> {
        debug_assert_eq!(self.cur(), Some(b'"'));
        let start = self.pos;
        self.eat();
        let body_start = self.offset;
        let body_end = loop {
            match self.cur() {
                None | Some(b'\n') => {
                    return self.error(start, "unterminated string literal");
                }
                Some(b'"') => {
                    let end = self.offset;
                    self.eat();
                    break end;
                }
                Some(b'\\') => self.eat_escape_seq()?,
                Some(_) => self.eat(),
            }
        };

        Ok(Token {
            loc: self.loc_from(start),
            kind: TokenKind::String,
            body: &self.source[body_start..body_end],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Program;

    const TESTS_PATH: &str = "experiments/dblang/parse/Lexer.test";

    fn fmt_loc(filename: &str, loc: &Loc) -> String {
        let mut s = filename.to_string();
        if loc.start.line != 0 {
            if loc.start.line == loc.end.line {
                s += &format!(
                    ":{}:{}-{}",
                    loc.start.line, loc.start.column, loc.end.column
                );
            } else {
                s += &format!(
                    " lines {}-{} characters {}-{}",
                    loc.start.line, loc.end.line, loc.start.column, loc.end.column
                );
            }
        }
        s
    }

    fn run(input: &str) -> String {
        let mut program = Program::new();
        let file = program.add_source_file(TESTS_PATH);
        let mut tokens = Vec::new();
        let result = lex(input, file, InFilePos::start_of_file(), &mut tokens);

        let mut out = String::new();
        for token in &tokens {
            out += &format!(
                "{}: {} '{}'\n",
                fmt_loc(TESTS_PATH, &token.loc),
                token.kind,
                token.body
            );
        }

        if let Err(e) = result {
            out += &format!("{}: error: {}\n", fmt_loc(TESTS_PATH, &e.loc), e.message);
        }

        out
    }

    /// Runs the golden tests shared with the C++ implementation.
    #[test]
    fn golden() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../dblang/parse/Lexer.test");
        let content = std::fs::read_to_string(path).unwrap();
        let mid_sep = format!("{}\n", "-".repeat(80));
        let test_sep = format!("{}\n", "=".repeat(80));

        let mut failures = Vec::new();
        let mut count = 0;
        let mut rest = content.as_str();
        while let Some(end) = rest.find(&test_sep) {
            // Separators must be at the start of a line.
            assert!(end == 0 || rest.as_bytes()[end - 1] == b'\n');
            let case = &rest[..end];
            rest = &rest[end + test_sep.len()..];

            let mid = case
                .match_indices(&mid_sep)
                .find(|&(i, _)| i == 0 || case.as_bytes()[i - 1] == b'\n')
                .expect("missing input/output separator")
                .0;
            let input = &case[..mid];
            let expected = &case[mid + mid_sep.len()..];
            let actual = run(input);
            count += 1;
            if actual != expected {
                failures.push(format!(
                    "input:\n{input}\nexpected:\n{expected}\nactual:\n{actual}"
                ));
            }
        }

        assert!(rest.trim().is_empty(), "trailing content after last test");
        assert!(count > 0);
        assert!(
            failures.is_empty(),
            "{}/{count} lexer tests failed:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
