import requests
from bs4 import BeautifulSoup
import sys
import pandas as pd
import pprint
import time
import os
import numpy as np
import re

pp = pprint.PrettyPrinter()

files = ['data/right.txt', 'data/left.txt', 'data/center.txt']

for file in files:
    with open(file, 'r') as f:
        url_counts = {}
        label = file[5:].replace(".txt", "")

        lines = f.readlines()
        lines = [line.strip().replace('\n', '') for line in lines]
        df = pd.DataFrame({'urls':lines, 'label': [label]*len(lines)})
        urls = list(df['urls'])
        for url in urls:
            
            domain = re.findall(r"\bhttps://\b(.*?)\b/\b", url)
            
            if len(domain) > 0:
                if domain[0] in url_counts:
                    url_counts[domain[0]] += 1
                else:
                    url_counts[domain[0]] = 1

            else:
                domain = re.findall(r"\bhttp://\b(.*?)\b/\b", url)
                if len(domain) > 0:
                    if domain[0] in url_counts:
                        url_counts[domain[0]] += 1
                    else:
                        url_counts[domain[0]] = 1
                else:
                    print(url)
                    print(domain)
    # largest = []
    # while len(largest) < 5:
    sorted_counts = sorted(url_counts.items(), key=lambda x: x[1], reverse=True)
    print(file)
    pp.pprint(sorted_counts)
    
    # print(file)
    # print(largest)

df = pd.read_csv("data/MBIC/labeled_dataset.csv")
print(df.head())