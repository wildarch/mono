#!/usr/bin/env python3
from bs4 import BeautifulSoup
import requests

r = requests.get('https://doi.org/10.1145/1016848.1016856')
soup = BeautifulSoup(r.text, features='lxml')
abstracts = soup.select("div.abstractInFull p")
print(abstracts[0].text)