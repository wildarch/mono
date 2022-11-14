package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/url"

	"github.com/andybalholm/cascadia"
	"golang.org/x/net/html"

	_ "modernc.org/sqlite"
)

var dblpDbPath = flag.String("dblp-db-path", "", "path to the sqlite database file storing the dblp table")
var reviewDbPath = flag.String("review-db-path", "", "path to the sqlite database file storing the PaperReview table")
var testUrl = flag.String("test-url", "", "test abstract fetching on the given url")

func main() {
	flag.Parse()

	if *testUrl != "" {
		abstract, err := fetchAbstract(*testUrl)
		if err != nil {
			log.Fatalf("error fetching abstract: %s", err.Error())
		}
		log.Printf("found abstract: %s", abstract)
		return
	}

	if *dblpDbPath == "" {
		log.Fatalf("Missing required argument: --dblp-db-path")
	}
	dblpDb, err := sql.Open("sqlite", *dblpDbPath)
	if err != nil {
		log.Fatalf("error opening dblp DB at %s: %s", *dblpDbPath, err.Error())
	}
	defer dblpDb.Close()
}

func fetchAbstract(urlS string) (string, error) {
	lastUrl, err := url.Parse(urlS)
	if err != nil {
		return "", fmt.Errorf("url to fetch is not valid: %w", err)
	}
	client := http.Client{
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
		return "", fmt.Errorf("request to %s failed: %s", urlS, err)
	}
	switch host := lastUrl.Hostname(); host {
	case "link.springer.com":
		return fetchSpringerAbstract(resp)
	case "ieeexplore.ieee.org":
		return fetchIEEEXploreAbstract(resp)
	default:
		return "", fmt.Errorf("unsupported host %s", host)
	}
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
