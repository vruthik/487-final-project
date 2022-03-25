import requests
from bs4 import BeautifulSoup

def scrape_allsides():
    r = requests.get('https://www.allsides.com/unbiased-balanced-news')
    soup = BeautifulSoup(r.text, "html.parser")
    


scrape_allsides()