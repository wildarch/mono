package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"time"

	"github.com/andybalholm/cascadia"
	"golang.org/x/net/html"

	_ "modernc.org/sqlite"
)

var dblpDbPath = flag.String("dblp-db-path", "", "path to the sqlite database file storing the dblp table")
var reviewDbPath = flag.String("review-db-path", "", "path to the sqlite database file storing the PaperReview table")
var testUrl = flag.String("test-url", "", "test abstract fetching on the given url")
var prefetchLimit = flag.Int("prefetch-limit", 100, "number of abstracts to prefetch")

func main() {
	flag.Parse()

	if *testUrl != "" {
		_, abstract, err := fetchAbstract(*testUrl)
		if err != nil {
			log.Fatalf("error fetching abstract: %s", err.Error())
		}
		log.Printf("found abstract: %s", abstract)
		return
	}

	// Open review database
	if *reviewDbPath == "" {
		log.Fatalf("Missing required argument: --review-db-path")
	}
	reviewDb, err := sql.Open("sqlite", *reviewDbPath)
	if err != nil {
		log.Fatalf("error opening review DB at %s: %s", *reviewDbPath, err.Error())
	}
	defer reviewDb.Close()

	// Attach the DBPL database
	if *dblpDbPath == "" {
		log.Fatalf("Missing required argument: --dblp-db-path")
	}
	_, err = reviewDb.Exec("attach database ? as dblp;", *dblpDbPath)
	if err != nil {
		log.Fatalf("error attaching DBLP database: %s", err.Error())
	}

	runFetcher(reviewDb)
}

func runFetcher(db *sql.DB) {
	for {
		// Check if we have reached our limit for prefetched abstracts
		r := db.QueryRow(`
			SELECT COUNT(*) 
			FROM PaperReview 
				  -- We have attempted to fetch the abstract 
			WHERE resolved_ee IS NOT NULL 
				  -- But the user has not seen it yet
			  AND interesting IS NULL`)
		var cnt int
		err := r.Scan(&cnt)
		if err != nil {
			log.Fatalf("error checking for prefetched abstracts: %s", err.Error())
		}

		if cnt >= *prefetchLimit {
			log.Printf("We are limited to prefetching %d abstracts, but already have %d. Waiting...", *prefetchLimit, cnt)
			time.Sleep(time.Minute)
			continue
		}

		// Find our new candidate
		r = db.QueryRow(`
			SELECT key, ee
			FROM dblp 
			LEFT JOIN PaperReview ON (key = dblp_key) 
				  -- We have not yet attempted to prefetch the abstract
			WHERE resolved_ee IS NULL 
				  -- An electronic edition is available
			  AND ee IS NOT NULL 
			-- Randomly take one
			ORDER BY RANDOM()
			LIMIT 1;
		`)
		var key string
		var ee string
		err = r.Scan(&key, &ee)
		if err != nil {
			log.Fatalf("error looking for a paper abstract to prefetch: %s", err.Error())
		}

		// Try to fetch the abstract
		log.Printf("Looking for abstract for key %s, url %s", key, ee)
		lastUrl, abstract, err := fetchAbstract(ee)
		// Save result in the database
		if err != nil {
			log.Printf("Error retrieving abstract: %s", err.Error())

			// Set resolved_ee but not abstract
			_, err := db.Exec(`
				INSERT INTO PaperReview (dblp_key, resolved_ee) VALUES (?, ?);
			`, key, lastUrl)
			if err != nil {
				log.Fatalf("error inserting failed abstract fetch record: %s", err.Error())
			}
		} else {
			// Set resolved_ee and abstract
			_, err := db.Exec(`
				INSERT INTO PaperReview (dblp_key, resolved_ee, abstract) VALUES (?, ?, ?);
			`, key, lastUrl, abstract)
			if err != nil {
				log.Fatalf("error inserting abstract fetch record: %s", err.Error())
			}
		}
		// sleep a bit to avoid making library servers angry
		time.Sleep(2 * time.Second)
	}
}

func fetchAbstract(urlS string) (string, string, error) {
	lastUrl, err := url.Parse(urlS)
	if err != nil {
		return lastUrl.String(), "", fmt.Errorf("url to fetch is not valid: %w", err)
	}
	jar, err := cookiejar.New(nil)
	if err != nil {
		return lastUrl.String(), "", fmt.Errorf("error making cookie jar: %w", err)
	}
	client := http.Client{
		Jar: jar,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) == 10 {
				return fmt.Errorf("too many redirects (%d)", len(via))
			}
			lastUrl = req.URL
			return nil
		},
	}
	resp, err := client.Get(urlS)
	if err != nil {
		return lastUrl.String(), "", fmt.Errorf("request to %s failed: %s", urlS, err)
	}
	var abs string
	switch host := lastUrl.Hostname(); host {
	case "link.springer.com":
		abs, err = fetchSpringerAbstract(resp)
	case "ieeexplore.ieee.org":
		abs, err = fetchIEEEXploreAbstract(resp)
	case "dl.acm.org":
		abs, err = fetchACMAbstract(resp)
	default:
		abs, err = "", fmt.Errorf("unsupported host %s", host)
	}
	return lastUrl.String(), abs, err
}

func fetchSpringerAbstract(resp *http.Response) (string, error) {
	doc, err := parseResponse(resp)
	if err != nil {
		return "", err
	}

	return getSelectorInnerText(doc, "div#Abs1-content p")
}

func fetchIEEEXploreAbstract(resp *http.Response) (string, error) {
	doc, err := parseResponse(resp)
	if err != nil {
		return "", err
	}

	sel := cascadia.MustCompile("meta[property=\"og:description\"]")
	n := cascadia.Query(doc, sel)
	if n == nil {
		return "", fmt.Errorf("cannot find abstract")
	}
	for _, a := range n.Attr {
		if a.Key == "content" {
			return a.Val, nil
		}
	}
	return "", fmt.Errorf("content key not found")
}

func fetchACMAbstract(resp *http.Response) (string, error) {
	doc, err := parseResponse(resp)
	if err != nil {
		return "", err
	}

	return getSelectorInnerText(doc, "div.abstractInFull p")
}

func parseResponse(resp *http.Response) (*html.Node, error) {
	// Parse the HTML
	doc, err := html.Parse(resp.Body)
	if err != nil {
		resp.Body.Close()
		return nil, fmt.Errorf("error parsing html response: %w", err)
	}
	err = resp.Body.Close()
	if err != nil {
		return nil, fmt.Errorf("error closing response body: %w", err)
	}

	return doc, nil
}

func getSelectorInnerText(doc *html.Node, s string) (string, error) {
	sel := cascadia.MustCompile(s)
	n := cascadia.Query(doc, sel)
	if n == nil {
		return "", fmt.Errorf("cannot find abstract")
	}

	text := ""
	c := n.FirstChild
	for c != nil {
		if c.Type == html.TextNode {
			text += c.Data
		} else {
			text += "<?>"
		}
		c = c.NextSibling
	}
	return text, nil
}
