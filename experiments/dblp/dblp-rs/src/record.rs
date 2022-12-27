#[derive(Debug, Clone)]
pub struct Record {
    pub record_type: RecordType,
    pub key: String,
    // TODO: publtype
    // TODO: cdate
    // TODO: mdate
    pub title: String,
    // TODO: pages
    // TODO: publisher
    // TODO: crossref
    // TODO: series
    pub year: u16,

    // Repeated fields
    pub ee: Vec<String>,
    pub author: Vec<String>,
    pub editor: Vec<String>,
    pub cite: Vec<String>,
    // TODO: school
    // TODO: isbn
    // TODO: note
    // TODO: url
    // TODO: cdrom

    // Not sure
    // TODO: journal
    // TODO: chapter
    // TODO: booktitle
    // TODO: volume
    // TODO: publnl
    // TODO: month
    // TODO: number
    // TODO: address
}

impl Record {
    pub fn new() -> Self {
        Self {
            record_type: RecordType::Article,
            key: String::new(),
            title: String::new(),
            year: 0,
            ee: Vec::new(),
            author: Vec::new(),
            editor: Vec::new(),
            cite: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.record_type = RecordType::Article;
        self.key.clear();
        self.title.clear();
        self.year = 0;
        self.ee.clear();
        self.author.clear();
        self.editor.clear();
        self.cite.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecordType {
    InProceedings,
    Article,
    Book,
    Www,
    Proceedings,
    Incollection,
    MastersThesis,
    PhdThesis,
}
