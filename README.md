# 487-final-project

## TODO
- [ ] Implement Crawler and Scraper in ```scrape.py```
- [ ] Implement Naives Bayes as baseline model in ```naivebayes.py```
- [ ] Implement evaluation metric calculations in ```eval.py```

## data


## scrape.py
```python scrape.py internal```  
Runs scrape.py to gather internal allsides.com links to webpages that have the link to the external article as well as the media bias score. This command will scrape the links from all of the topics on allsides.com

```python scrape.py internal last_topic``` 
Runs scrape.py like above except starts scraping link from last_topic onwards. Meant to be used in case the scraper crashes midway.

```python scrape.py articles internal_links.txt```   
Runs scrape.py to gather external article links from internal_links.txt which contains allsides.com links gathered from the command(s).


## naivebayes.py


## eval.py


