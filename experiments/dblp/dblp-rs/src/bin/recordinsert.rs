use rusqlite::Connection;

fn main() {
    let conn = Connection::open_in_memory().unwrap();

    conn.execute(
        r#"
        CREATE TABLE Dblp(
            key TEXT,
            record_type TEXT,
            title TEXT
        );
    "#,
        (),
    )
    .unwrap();

    {
        let mut insert_stmt = conn.prepare("INSERT INTO Dblp VALUES (?, ?, ?)").unwrap();

        let record_type = "article";
        let title = String::from("Some fancy title");

        for i in 0..9_000_000 {
            let key = format!("key{i}");
            insert_stmt.execute((&key, record_type, &title)).unwrap();
        }
    }

    conn.close().unwrap();
}
