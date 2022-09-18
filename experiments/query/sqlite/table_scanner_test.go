package sqlite_test

import (
	"database/sql"
	"fmt"
	"reflect"
	"testing"

	"github.com/wildarch/mono/experiments/query/sqlite"
	_ "modernc.org/sqlite"
)

func TestScanSchema(t *testing.T) {
	tempDir := t.TempDir()
	dbFilePath := tempDir + "/db.sqlite3"
	db, err := sql.Open("sqlite", dbFilePath)
	defer db.Close()
	if err != nil {
		t.Fatalf("failed to open database file %s: %v", dbFilePath, err)
	}

	personSql :=
		`CREATE TABLE Person(
			id INTEGER PRIMARY KEY,
			name TEXT,
			age INT
		)`
	bookSql :=
		`CREATE TABLE Book(
			isbn INTEGER PRIMARY KEY,
			title TEXT,
			description TEXT
		)`
	lovedBookSql :=
		`CREATE TABLE LovedBook(
			person_id INTEGER,
			isbn INTEGER,

			FOREIGN KEY(person_id) REFERENCES Person(id),
			FOREIGN KEY(isbn) REFERENCES Book(isbn)
		)`

	// Create some tables
	_, err = db.Exec(fmt.Sprintf("%s;%s;%s;", personSql, bookSql, lovedBookSql))
	if err != nil {
		t.Fatalf("failed to create tables: %v", err)
	}

	// Now check that our scanner can read them
	conn, err := sqlite.Open(dbFilePath)
	if err != nil {
		t.Fatalf("failed to open database file %s: %v", dbFilePath, err)
	}
	defer conn.Close()

	scan, err := conn.ScanTable("sqlite_schema")
	if err != nil {
		t.Fatalf("failed to open schema table for scanning: %s", err.Error())
	}

	// type | name | tbl_name | rootpage | sql

	personRow := assertReadRow(t, scan)
	assertEqual(t, personRow.Values[0], "table")
	assertEqual(t, personRow.Values[1], "Person")
	assertEqual(t, personRow.Values[4], personSql)

	bookRow := assertReadRow(t, scan)
	assertEqual(t, bookRow.Values[0], "table")
	assertEqual(t, bookRow.Values[1], "Book")
	assertEqual(t, bookRow.Values[4], bookSql)

	lovedBookRow := assertReadRow(t, scan)
	assertEqual(t, lovedBookRow.Values[0], "table")
	assertEqual(t, lovedBookRow.Values[1], "LovedBook")
	assertEqual(t, lovedBookRow.Values[4], lovedBookSql)

}

func TestReadTable(t *testing.T) {
	tempDir := t.TempDir()
	dbFilePath := tempDir + "/db.sqlite3"
	db, err := sql.Open("sqlite", dbFilePath)
	defer db.Close()
	if err != nil {
		t.Fatalf("failed to open database file %s: %v", dbFilePath, err)
	}

	// Create a table and fill it with some rows
	_, err = db.Exec(`
		CREATE TABLE Person(
			name TEXT,
			age INT
		);
		INSERT INTO Person VALUES ("Alice", 99);
		INSERT INTO Person VALUES ("Bob", 88);
		INSERT INTO Person VALUES ("Gandalf", 1000);
	`)
	if err != nil {
		t.Fatalf("failed to create person table: %v", err)
	}

	// Now check that our scanner can read them
	conn, err := sqlite.Open(dbFilePath)
	if err != nil {
		t.Fatalf("failed to open database file %s: %v", dbFilePath, err)
	}
	defer conn.Close()

	scan, err := conn.ScanTable("Person")
	if err != nil {
		t.Fatalf("failed to open Person table for scanning: %s", err.Error())
	}

	aliceRow := assertReadRow(t, scan)
	assertEqual(t, aliceRow.Values[0], "Alice")
	assertEqual(t, aliceRow.Values[1], int64(99))
	bobRow := assertReadRow(t, scan)
	assertEqual(t, bobRow.Values[0], "Bob")
	assertEqual(t, bobRow.Values[1], int64(88))
	gandalfRow := assertReadRow(t, scan)
	assertEqual(t, gandalfRow.Values[0], "Gandalf")
	assertEqual(t, gandalfRow.Values[1], int64(1000))
}

func assertEqual(t *testing.T, a interface{}, b interface{}) {
	tya := reflect.TypeOf(a)
	tyb := reflect.TypeOf(b)
	if tya != tyb {
		t.Errorf("a and b have different types (%v != %v)", tya, tyb)
		return
	}
	if a != b {
		t.Errorf("expected a == b\na = %v\nb = %v", a, b)
	}
}

func assertReadRow(t *testing.T, scan *sqlite.TableScanner) *sqlite.Row {
	row, err := scan.ReadRow()
	if err != nil {
		t.Errorf("failed to read row: %s", err.Error())
	}
	return row
}
