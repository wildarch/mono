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
1. A DBLP downloader and parser, storing relevant data in a SQLite database.
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