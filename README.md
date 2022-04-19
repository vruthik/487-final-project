# 487-final-project

## TODO
- [X] Implement crawler and scraper  
- [X] Implement baseline Naive Bayes model  
- [X] Implement evaluation metrics  
- [X] Implement Bert based linear model 
- [X] Hyperparameter tuning
- [X] Implement twitter models
- [X] Evaluate

## model.py

Contains all of the code for the Bert based linear model. The training and hyperparameter tuning code is in ```train_model.ipynb``` and ```train_tweet_model.ipynb```.

## naivebayes.py
Contains the code for the baseline naivebayes model. Training and testing code is in ```nb.ipynb``` 

## twitter.py
Contains the code to scrape recent tweets of all 100 US Senators and process them. 

## scrape.py

```
scrape_allsides_internal_links(last_topic=None)
```
Scrapes the internal allsides links for each of the topics on the website and writes them to internal_links.txt. Scraping output is written to internal_links.txt everytime a topic is complete to prevent restarting

last_topic: In case of a program crash, to prevent restarting from the first topic, the program will resume from last_topic onwards (inclusive). If last_topic is not specified, it will scrape from the beginning.

Usage:
```python scrape.py internal```
```python scrape.py internal last_topic```


```
scrape_news_article_links(filename, start_link_number=None)
```

Gathers the external article link from the allsides article along with the media bias rating of the website 
Based on the media bias rating, the gathered article links are written to either right.txt, left.txt or center.txt
Scraping output is written to right.txt, left.txt and center.txt every 60 iterations (approx every 10 minutes) in case of a crash

filename: the name of the text file that contains all of the allsides internal links from scrape_allsides_internal_links()
start_link_number: In case of a crash, the program can be restarted at start_link_number (inclusive) to avoid having to rerun the entire program

Usage:
```python scrape.py article filename```
```python scrape.py article filename start_link_number```


```
scrape_raw_text()
```
Scrapes the raw text from the article links gathered from allsides.com in ```scrape_news_article_links()```. It also balances the dataset to include the same number of urls for each class.

It utilizes a vectorized scraping function so if program crashes it has to be restarted.  

Usage:
```python scrape.py text```

## eval.py
Contains an eval class to evaluate accuracy, macro and micro F1, recall and precision given predictions and labels. 


