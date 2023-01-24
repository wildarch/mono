#!/usr/bin/env python3
import os
from lxml import etree
from dataclasses import dataclass

PEOPLE_DIR='people/'

@dataclass
class Paper:
    key: str
    title: str
    authors: list[str]
    year: int
    ee: str

author_names = set()
papers = {}

for xml_filename in os.listdir(PEOPLE_DIR):
    xml_path = PEOPLE_DIR + xml_filename
    tree = etree.parse(xml_path)

    for name in tree.iterfind('./person/author'):
        author_names.add(name.text)

    for r in tree.iterfind('//r/*'):
        title = etree.tostring(r.find('.//title')).decode('utf-8')
        title = title.strip().replace('<title>', '').replace('</title>', '')
        paper = Paper(
            key = r.get('key'),
            title = title,
            authors = [a.text for a in r.iterfind('.//author')],
            year = int(r.findtext('.//year')),
            ee = r.findtext('.//ee'),
        )

        papers[paper.key] = paper

print(len(papers))

papers = list(papers.values())
papers.sort(key=lambda p: p.year, reverse=True)

with open('papers.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<meta charset="UTF-8">
<title>Papers</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<body>
<h1>Papers</h1>

    ''')

    year = papers[0].year
    f.write(f'<h2>{year}</h2>')
    for paper in papers:
        if paper.year != year:
            year = paper.year
            f.write(f'<h2>{year}</h2>')
        
        authors = [f'<b>{n}</b>' if n in author_names else n for n in paper.authors]
        
        f.write('<div>')
        f.write(f'<a href="{paper.ee}">{paper.title}</a> {", ".join(authors)}')
        f.write('<br />')
        f.write('<br />')
        f.write('</div>')

    f.write('''
</body>
</html>
    ''')