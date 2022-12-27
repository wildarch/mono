use std::collections::HashMap;

/// Mapping between DBLP entity names and unicode characters.
pub fn dblp_mapping() -> Vec<(&'static str, char)> {
    let mut map = Vec::from(CUSTOM_ENTITIES);
    for (idx, name) in LATIN1_NAMES.iter().enumerate() {
        let value = char::from_u32(192 + idx as u32).unwrap();
        map.push((name, value));
    }
    map
}

// Taken from dblp-2019-11-22
const CUSTOM_ENTITIES: &'static [(&'static str, char)] =
    &[("reg", '®'), ("micro", 'µ'), ("times", '×')];

const LATIN1_NAMES: &'static [&str] = &[
    "Agrave",       // capital A, grave accent (unicode 192)
    "Aacute",       // capital A, acute accent
    "Acirc",        // capital A, circumflex accent
    "Atilde",       // capital A, tilde
    "Auml",         // capital A, dieresis or umlaut mark
    "Aring",        // capital A, ring
    "AElig",        // capital AE diphthong (ligature)
    "Ccedil",       // capital C, cedilla
    "Egrave",       // capital E, grave accent
    "Eacute",       // capital E, acute accent
    "Ecirc",        // capital E, circumflex accent
    "Euml",         // capital E, dieresis or umlaut mark
    "Igrave",       // capital I, grave accent
    "Iacute",       // capital I, acute accent
    "Icirc",        // capital I, circumflex accent
    "Iuml",         // capital I, dieresis or umlaut mark
    "ETH",          // capital Eth, Icelandic
    "Ntilde",       // capital N, tilde
    "Ograve",       // capital O, grave accent
    "Oacute",       // capital O, acute accent
    "Ocirc",        // capital O, circumflex accent
    "Otilde",       // capital O, tilde
    "Ouml",         // capital O, dieresis or umlaut mark
    "UNMAPPED_215", // 215 is not mapped
    "Oslash",       // capital O, slash
    "Ugrave",       // capital U, grave accent
    "Uacute",       // capital U, acute accent
    "Ucirc",        // capital U, circumflex accent
    "Uuml",         // capital U, dieresis or umlaut mark
    "Yacute",       // capital Y, acute accent
    "THORN",        // capital THORN, Icelandic
    "szlig",        // small sharp s, German (sz ligature)
    "agrave",       // small a, grave accent
    "aacute",       // small a, acute accent
    "acirc",        // small a, circumflex accent
    "atilde",       // small a, tilde
    "auml",         // small a, dieresis or umlaut mark
    "aring",        // small a, ring
    "aelig",        // small ae diphthong (ligature)
    "ccedil",       // small c, cedilla
    "egrave",       // small e, grave accent
    "eacute",       // small e, acute accent
    "ecirc",        // small e, circumflex accent
    "euml",         // small e, dieresis or umlaut mark
    "igrave",       // small i, grave accent
    "iacute",       // small i, acute accent
    "icirc",        // small i, circumflex accent
    "iuml",         // small i, dieresis or umlaut mark
    "eth",          // small eth, Icelandic
    "ntilde",       // small n, tilde
    "ograve",       // small o, grave accent
    "oacute",       // small o, acute accent
    "ocirc",        // small o, circumflex accent
    "otilde",       // small o, tilde
    "ouml",         // small o, dieresis or umlaut mark
    "UNMAPPED_247", // 247 is not mapped
    "oslash",       // small o, slash
    "ugrave",       // small u, grave accent
    "uacute",       // small u, acute accent
    "ucirc",        // small u, circumflex accent
    "uuml",         // small u, dieresis or umlaut mark
    "yacute",       // small y, acute accent
    "thorn",        // small thorn, Icelandic
    "yuml",         // small y, dieresis or umlaut mark
];

pub struct DblpUnescaper {
    mapping: HashMap<&'static str, String>,
}

impl DblpUnescaper {
    pub fn new() -> Self {
        let mapping = dblp_mapping()
            .into_iter()
            .map(|(s, c)| (s, c.to_string()))
            .collect();
        Self { mapping }
    }

    pub fn unescape<'a>(&'a self, s: &str) -> Option<&'a str> {
        self.mapping.get(s).map(|x| &**x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(non_snake_case)]
    #[test]
    fn Agrave() {
        assert_eq!(unescape("Agrave"), "À");
    }

    #[test]
    fn ograve() {
        assert_eq!(unescape("ograve"), "ò");
    }

    #[test]
    fn yuml() {
        assert_eq!(unescape("yuml"), "ÿ");
    }

    fn unescape(s: &str) -> &str {
        let escaper = Box::leak(Box::new(DblpUnescaper::new()));
        escaper.unescape(s).unwrap()
    }
}
