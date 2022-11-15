package main

import (
	"database/sql"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"

	_ "embed"

	_ "modernc.org/sqlite"
)

var dblpDbPath = flag.String("dblp-db-path", "", "path to the sqlite database file storing the dblp table")
var reviewDbPath = flag.String("review-db-path", "", "path to the sqlite database file storing the PaperReview table")
var db *sql.DB = nil

//go:embed templates.html
var rawTemplates string
var templates *template.Template

func parseTemplates() {
	templates = template.Must(template.New("templates").Parse(rawTemplates))
}

func main() {
	flag.Parse()
	parseTemplates()

	// Open review database
	if *reviewDbPath == "" {
		log.Fatalf("Missing required argument: --review-db-path")
	}
	reviewDb, err := sql.Open("sqlite", *reviewDbPath)
	if err != nil {
		log.Fatalf("error opening review DB at %s: %s", *reviewDbPath, err.Error())
	}
	db = reviewDb
	defer db.Close()

	// Attach the DBPL database
	if *dblpDbPath == "" {
		log.Fatalf("Missing required argument: --dblp-db-path")
	}
	_, err = db.Exec("attach database ? as dblp;", *dblpDbPath)
	if err != nil {
		log.Fatalf("error attaching DBLP database: %s", err.Error())
	}

	http.HandleFunc("/", handleShowPaper)
	http.HandleFunc("/approve", handleApprove)
	http.HandleFunc("/reject", handleReject)
	http.ListenAndServe(":8080", nil)
}

type ShowPaper struct {
	Title    string
	Abstract string
	Key      string
}

func handleShowPaper(w http.ResponseWriter, r *http.Request) {
	// Get a paper with an abstract that is unrated
	row := db.QueryRow(`
		SELECT key, title, abstract
		FROM PaperReview
		JOIN dblp ON (key = dblp_key)
			  -- Has an abstract
		WHERE abstract IS NOT NULL
		  	  -- Not yet reviewed
		  AND interesting IS NULL
		LIMIT 1;
	`)
	var showPaper ShowPaper
	err := row.Scan(&showPaper.Key, &showPaper.Title, &showPaper.Abstract)
	if err != nil {
		log.Printf("error fetching a paper to show: %s", err.Error())
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	err = templates.Lookup("ShowPaper").Execute(w, showPaper)
	if err != nil {
		log.Printf("error rendering paper: %s", err.Error())
		return
	}
}

func handleApprove(w http.ResponseWriter, r *http.Request) {
	handleApproveOrReject(w, r, true)
}

func handleReject(w http.ResponseWriter, r *http.Request) {
	handleApproveOrReject(w, r, false)
}

func handleApproveOrReject(w http.ResponseWriter, r *http.Request, interested bool) {
	key := r.URL.Query().Get("key")
	if key == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "Missing 'key' query parameter")
		return
	}

	res, err := db.Exec(`
		UPDATE PaperReview
		   SET interesting = ?
		 WHERE dblp_key = ?;
	`, interested, key)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		log.Printf("error writing paper evaluation to the database")
		return
	}
	modified, err := res.RowsAffected()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		log.Printf("error retrieving number of rows affected")
		return
	}
	if modified == 0 {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "No paper found for key '%s'", key)
		return
	}
	if modified > 1 {
		w.WriteHeader(http.StatusInternalServerError)
		log.Printf("expect to change one row, but modified %d", modified)
		return
	}
	// Sanity check
	if modified != 1 {
		panic("expect to modify one paper")
	}

	// Redirect back to main screen
	http.Redirect(w, r, "/", http.StatusTemporaryRedirect)
}
