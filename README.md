# 487-final-project

## TODO
- [X] Implement crawler and scraper  
- [ ] Implement Preprocessor  
- [ ] Implement baseline Naive Bayes model  
- [ ] Implement evaluation metrics   

## scrape.py

```
scrape_allsides_internal_links(last_topic=None)
  Scrapes the internal allsides links for each of the topics on the website and writes them to internal_links.txt
  Scraping output is written to internal_links.txt everytime a topic is complete to prevent restarting
  
  last_topic: In case of a program crash, to prevent restarting from the first topic, the program will resume from last_topic onwards (inclusive)
```

## naivebayes.py

## eval.py
