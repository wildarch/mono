# Paper tinder - Paper discovery for Generation Z
For better or worse, there are many moments in a typical day when I have access to my phone and a few minutes to kill.
I try to stay away from social media and infinite scrolling apps as much as possible these days, so instead, I find myself browsing news articles or reading [Hacker](https://news.ycombinator.com) News](https://news.ycombinator.com).
The content there is arguably a less harmful and better use of my time than social media, but I still think it's not the most productive.
Meanwhile, the scientific community puts out a ton of interesting papers (especially for a potential future Ph.D. student), but as far as I know, there is no easy way to keep up to date with the latest publications in a field aside from following specific conferences.

My idea then is to use this downtime in my day to skim publication titles and abstracts to determine if they are of interest to me and if so add them to my reading list for later.
This would be similar to how people casually check Tinder during the day to find people to chat with later: you are shown the title and abstract of a paper and can decide to either discard it or mark it as an interesting read for later. 
If I build a simple web interface, I can open it from my phone when I have some time, and start skimming.

## Data sources
First of all, we need a database of candidate papers to read.
The obvious candidate is [DBLP](https://dblp.org), a huge library of computer science publications that is regularly updated with publications for (as far as I can tell) all the prominent conferences.
DBLP is available as [one big XML file](https://dblp.org/xml/) which is a bit inconvenient, but we can parse it and push the data into a database without too much trouble.
DBLP already gives us:
- Paper title
- Authors
- Publication date
- Often, a link to the authoritative source for the publication, often a [DOI](https://www.doi.org/).

### Obtaining abstracts
The main complicating factor is that DBLP does not store abstracts, as this would [violate copyright](https://dblp.org/faq/Why+are+there+no+abstracts+in+dblp.html).
My proposed solution to this is to follow the URL stored in DBPL, and attempt to extract the abstract from the web page.
Take for example the paper [`Making a fast curry`](https://dblp.org/rec/conf/icfp/MarlowJ04.xml).
It lists a URL `https://doi.org/10.1145/1016848.1016856` which forwards to the ACM digital library website and presents us with an abstract.
A simple python script is enough for us to scrape the abstract:

```python
from bs4 import BeautifulSoup
import requests

r = requests.get('https://doi.org/10.1145/1016848.1016856')
soup = BeautifulSoup(r.text, features='lxml')
abstracts = soup.select("div.abstractInFull p")
print(abstracts[0].text)
```

In my test the HTTP request takes about 2 seconds, so we probably want to prefetch abstracts in the background.

## Design
The design consists of three parts:
1. A DBLP downloader and parser, storing relevant data in an SQLite database.
2. An abstract prefetcher that maintains a buffer of abstracts for unread papers. 
3. A web interface that displays random unread papers, and allows the user to either mark them as read or put them on a reading list.

### Database schema
The application runs on a single machine and has just one user (me!), so SQLite should be more than good enough.
The database of available papers will be rebuilt from scratch at every sync, so we create a separate database file for it.
If we ever add backups to the database, we easily backup up just the state that can we cannot rebuild. 
The schema for the DBLP export will be:

```sql
CREATE TABLE dblp(
    key TEXT PRIMARY KEY,
    -- Publication type. 
    -- * 'article' for a journal article
    -- * 'inproceedings' for a conference paper
    pub_type TEXT NOT NULL,
    -- Title of the publication.
    -- In DBLP titles can have style elements like <it>,
    -- but in parsing we remove those and use flat text only.
    title TEXT NOT NULL,
    -- Stored as a JSON array. 
    -- Not all publications list an author!
    authors TEXT NOT NULL,
    -- Optional but usually present.
    year INT,
    -- Link to the authoritative source.
    -- Often a DOI.
    ee TEXT
);
```

Our application keeps track of which papers seem interesting, and their abstracts.

```sql
CREATE TABLE PaperReview(
    -- References 'key' in the 'dblp' table
    dblp_key TEXT,
    -- If this field is set, the paper was reviewed.
    interesting BOOLEAN,
    -- The 'ee' field in 'dblp' is often a doi link that redirects to the actual page hosting the publication.
    -- We store it here because the final domain usually tells us who hosts the publication (for example, the ACM).
    -- If we fail to fetch the abstract, the resolved url can help us debug for what domains we should add fetch support.
    resolved_ee TEXT,
    abstract TEXT
);
```

### DBLP parser
For the initial version, we can download a recent export manually, and write a python script to parse it.
See the adjacent `parse.py` script. 
This script takes a few minutes to run, attempts to make it faster are under `../fastparse`.

### Abstract scraping
As we saw earlier, CSS Selectors are a flexible and compact way to extract text from web pages.
We could use python for the tool, but python is harder to deploy to servers, so I'll opt for go instead.
The [cascadia](https://pkg.go.dev/github.com/andybalholm/cascadia) package appears to be widely used, let's go with that.

Our scraper runs the following in a loop:
1. Find a publication from `dblp` with an `ee` but no entry in `PaperReview`. The protocol for `ee` must be `http` or `https`.
2. Figure out where the `ee` link redirects to, and store it in `resolved_ee`.
3. Retrieve the HTML page. 
   We bake in a few rules to parse the abstracts of common websites.
   For example, if we see that the domain is `https://dl.acm.org`, we can use the selector `div.abstractInFull p` to get the abstract.
4. If we find an abstract on the page, store it in `abstract`. 
5. Wait for a few seconds, to avoid getting banned.

We need to store URLs we are redirected to, which we do with a custom [`CheckRedirect`](https://pkg.go.dev/net/http#Client) function.

Selectors:
```
https://dl.acm.org                  div.abstractInFull p                text
https://link.springer.com           div#Abs1-content p                  text
https://ieeexplore.ieee.org         meta[property="og:description"]     content
```

For IEEE content, things are more involved.
`ee` should redirect to a link similar to
`https://www.computer.org/csdl/proceedings-article/coopis/1997/00613823/1dUnbgpYLKM`.
The last component `1dUnbgpYLKM` is the internal article ID.

To get the abstract we need to send a POST request to a graphql endpoint.
The request must look exactly like the one below, just with a different article ID (with some of the fields removed, the query fails with `Must provide query string`).

```js
fetch("https://www.computer.org/csdl/api/v1/graphql", {
  "body": "{\"variables\":{\"articleId\":\"1dUnbgpYLKM\"},\"query\":\"query ($articleId: String!) {\\n  proceeding: proceedingByArticleId(articleId: $articleId) {\\n    id\\n    title\\n    acronym\\n    groupId\\n    volume\\n    displayVolume\\n    year\\n    __typename\\n  }\\n  article: articleById(articleId: $articleId) {\\n    id\\n    doi\\n    title\\n    abstract\\n    abstracts {\\n      abstractType\\n      content\\n      __typename\\n    }\\n    fno\\n    authors {\\n      affiliation\\n      fullName\\n      givenName\\n      surname\\n      __typename\\n    }\\n    idPrefix\\n    isOpenAccess\\n    showRecommendedArticles\\n    showBuyMe\\n    hasPdf\\n    pubDate\\n    pubType\\n    pages\\n    year\\n    issn\\n    isbn\\n    notes\\n    notesType\\n    __typename\\n  }\\n  webExtras: webExtrasByArticleId(articleId: $articleId) {\\n    id\\n    name\\n    size\\n    location\\n    __typename\\n  }\\n  adjacentArticles: adjacentArticles(articleId: $articleId) {\\n    previous {\\n      fno\\n      articleId\\n      __typename\\n    }\\n    next {\\n      fno\\n      articleId\\n      __typename\\n    }\\n    __typename\\n  }\\n  recommendedArticles: recommendedArticlesById(articleId: $articleId) {\\n    id\\n    title\\n    doi\\n    abstractUrl\\n    parentPublication {\\n      id\\n      title\\n      __typename\\n    }\\n    __typename\\n  }\\n  articleVideos: videosByArticleId(articleId: $articleId) {\\n    id\\n    videoExt\\n    videoType {\\n      featured\\n      recommended\\n      sponsored\\n      __typename\\n    }\\n    article {\\n      id\\n      fno\\n      issueNum\\n      pubType\\n      volume\\n      year\\n      idPrefix\\n      doi\\n      title\\n      __typename\\n    }\\n    channel {\\n      id\\n      title\\n      status\\n      featured\\n      defaultVideoId\\n      category {\\n        id\\n        title\\n        type\\n        __typename\\n      }\\n      __typename\\n    }\\n    year\\n    title\\n    description\\n    keywords {\\n      id\\n      title\\n      status\\n      __typename\\n    }\\n    speakers {\\n      firstName\\n      lastName\\n      affiliation\\n      __typename\\n    }\\n    created\\n    updated\\n    imageThumbnailUrl\\n    runningTime\\n    aspectRatio\\n    metrics {\\n      views\\n      likes\\n      __typename\\n    }\\n    notShowInVideoLib\\n    __typename\\n  }\\n}\"}",
  "method": "POST",
});
```

Alternatively, we can render the page with JS enabled, and then extract the text for CSS selector `div.article-content`.

Here is a helpful query that will tell you what sites cause the most failures:
```sql
SELECT 
    -- Cut the part between // and next /, which should be the host part of the URL
    SUBSTR(SUBSTR(resolved_ee, INSTR(resolved_ee, '//') + 2), 0, INSTR(SUBSTR(resolved_ee, INSTR(resolved_ee, '//') + 2), '/')) AS site, 
    COUNT(*) 
FROM PaperReview 
      -- we followed the redirects
WHERE resolved_ee IS NOT NULL
      -- but were unable to parse the abstract
  AND abstract IS NULL 
GROUP BY 1
ORDER BY 2 DESC;
```

Scraping Elsevier does not seem to work very well.
It seems Cloudflare is interfering, or maybe their bot detection is good.