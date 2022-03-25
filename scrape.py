import requests
from bs4 import BeautifulSoup
import sys
import pandas as pd

def scrape():
    # df = pd.read_csv('data/AllSides News Source Political Leanings.csv')
    r = requests.get('https://www.axios.com/us-europe-liquefied-natural-gas-russia-d9f94ffe-1559-4307-8b13-28b4526beb02.html')
    soup = BeautifulSoup(r.text, 'html.parser')
    print(soup.text)

if __name__ == "__main__":
    # scrape(sys.argv[1])
    scrape()