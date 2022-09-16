package main

import (
	"flag"
	"log"

	"github.com/wildarch/mono/experiments/query/sqlite"
)

func main() {
	dbFilePath := flag.String("db", "", "path to sqlite database file")

	flag.Parse()

	if *dbFilePath == "" {
		log.Fatal("missing argument -db")
	}

	conn, err := sqlite.Open(*dbFilePath)
	if err != nil {
		log.Fatalf("failed to open database: %s", err.Error())
	}
	defer conn.Close()

	scan, err := conn.ScanTable("sqlite_schema")
	if err != nil {
		log.Fatalf("failed to open schema table for scanning: %s", err.Error())
	}

	for {
		row, err := scan.ReadRow()
		if err != nil {
			log.Fatalf("failed to read row: %s", err.Error())
		}
		if row == nil {
			// No more rows left
			break
		}

		log.Printf("row: %+v", row)
	}
}
