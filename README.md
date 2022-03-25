# 487-final-project

## TODO
- [X] Implement Crawler and Scraper in ```scrape.py```
- [ ] Implement Naives Bayes as baseline model in ```naivebayes.py```
- [ ] Implement evaluation metric calculations in ```eval.py```
- [ ] Implement deep learning model

## data


## scrape.py

<ins>**scrape_allsides_internal_links()**</ins>

Scrapes the internal allsides links for each of the topics on the website and writes them to internal_links.txt
Scraping output is written to internal_links.txt everytime a topic is complete to prevent restarting
```last_topic```: topic to restart from in case of a crash (command line argument, see below)

```python scrape.py internal```  
Runs scrape.py to gather internal allsides.com links to webpages that have the link to the external article as well as the media bias score. This command will scrape the links from all of the topics on allsides.com

```python scrape.py internal last_topic```  
Runs scrape.py like above, except it starts scraping link from last_topic onwards (inclusive). Meant to be used in case the scraper crashes midway.

<ins>**scrape_news_article_links()**</ins>  
Gathers the external article link from the allsides article along with the media bias rating of the website 
Based on the media bias rating, the gathered article links are written to either right.txt, left.txt or center.txt
Scraping output is written to right.txt, left.txt and center.txt every 60 iterations (approx every 10 minutes) in case of a crash
```filename```: File that contains internal allsides.com article links (default is internal_links.txt but must be specified on the command line)
```start_link_number```: link number to restart from in case of a crash (command line argument, see below)

```python scrape.py articles```   
Runs scrape.py to gather external article links from the file which contains allsides.com links gathered from the command(s). The default filename is internal_links.txt.

```python scrape.py articles filename start_link_number```   
Runs scrape.py like above, except it starts scraping from start_link_number (inclusive). Meant to be used in case the scraper crashes midway.


## naivebayes.py


## eval.py


