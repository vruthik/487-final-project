import requests
from bs4 import BeautifulSoup
import sys
import pandas as pd
import pprint
import time
import os
import numpy as np
import re

def scrape_allsides_internal_links(last_topic=None):
    """
    Scrapes the internal allsides links for each of the topics on the website and writes them to internal_links.txt
    Scraping output is written to internal_links.txt everytime a topic is complete to prevent restarting
    
    last_topic: In case of a program crash, to prevent restarting from the first topic, the program will resume from last_topic onwards (inclusive)
    """

    pprinter = pprint.PrettyPrinter()
    
    topics = ["2022 Elections", "Abortion", "Africa", "Agriculture", "Animal Welfare", "Arts and Entertainment", "Asia", "Australia", "Banking and Finance", "Bridging Divides", "Business", "Campaign Finance", "Campaign Rhetoric", "Capital Punishment and Death Penalty", "China", "CIA", "Civil Rights", "Coronavirus", "Criminal Justice", "Culture", "Cybersecurity", "DEA", "Defense and Security", "Defense Department", "Democratic Party", "Disaster", "Domestic Policy", "Economic Policy", "Economy and Jobs", "Education", "Elections", "Energy", "Environment", "EPA", "Ethnicity and Heritage", "Europe", "Fact Checking", "Fake News", "Family and Marriage", "FBI", "FDA", "Federal Budget", "Food", "Foreign Policy", "Free Speech", "General News", "George Floyd Protests", "Great Britain", "Gun Control and Gun Rights", "Healthcare", "History of Media Bias", "Holidays", "Homeland Security", "Housing and Homelessness", "Humor and Satire", "Immigration", "Impeachment", "Inequality", "ISIS", "Israel", "Justice", "Justice Department", "Kamala Harris", "Labor", "LGBTQ Issues", "Marijuana Legalization", "Media Bias", "Media Industry", "Mexico", "Middle East", "National Defense", "National Security", "North Korea", "NSA", "Nuclear Weapons", "Opioid Crisis", "Palestine", "People and Profit", "Polarization", "Politics", "Privacy", "Public Health", "Race and Racism", "Religion and Faith", "Republican Party", "Role of Government", "Russia", "Science", "Sexual Misconduct", "Social Security", "South Korea", "Sports", "State Department", "Supreme Court", "Sustainability", "Taxes", "Tea Party", "Technology", "Terrorism", "The Americas", "Trade", "Transportation", "Treasury", "US Census", "US Congress", "US Constitution", "US House", "US Military", "US Senate", "Veterans Affairs", "Violence in America", "Voting Rights and Voter Fraud", "Welfare", "White House", "Women's Issues", "World"]
    
    if last_topic is not None:
        if last_topic in topics:
            topics = topics[topics.index(last_topic):]
    
    # Format topic names for allsides urls
    topics_url = {}
    for topic in topics:
        topic = topic.split(" ")
        if len(topic) > 1:
            topics_url['-'.join(topic)] = "%20".join(topic)
        else:
            topics_url[topic[0]] = topic[0]

    print("----- Scraping Internal allsides.com links from", len(topics_url.keys()), "topics -----")

    for i, key in enumerate(topics_url.keys()):
        allsides_links = set()
        print('Topic', str(i)  + ":", key)

        with open ('internal_links.txt', 'a') as file:
            url = 'https://www.allsides.com/topics/' + key + '?search=' + topics_url[key]

            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')

            for link in soup.find_all('a'):
                for div in link.find_all("div", {"class": "news-title"}):
                    if 'www.allsides.com' in link.get('href') and 'news-source' not in link.get('href'):
                        allsides_links.add(link.get('href'))
            
            allsides_links = list(allsides_links)
            file.writelines(["%s\n" % item  for item in allsides_links])
        
        # Required in allsides.com robots.txt
        time.sleep(10)

def scrape_news_article_links(filename, start_link_number=None):

    """
    Gathers the external article link from the allsides article along with the media bias rating of the website 
    Based on the media bias rating, the gathered article links are written to either right.txt, left.txt or center.txt
    Scraping output is written to right.txt, left.txt and center.txt every 60 iterations (approx every 10 minutes) in case of a crash

    filename: the name of the text file that contains all of the allsides internal links from scrape_allsides_internal_links()
    start_link_number: In case of a crash, the program can be restarted at start_link_number (inclusive) to avoid having to rerun the entire program
    """

    pprinter = pprint.PrettyPrinter()

    allsides_links = set()

    with open(os.path.join("data", filename), 'r') as file:
        links = file.readlines()
        for link in links:
            allsides_links.add(link.strip().replace("\n", ""))

    allsides_links = list(allsides_links)
    
    if start_link_number is not None:
        assert(int(start_link_number) < len(allsides_links))
        allsides_links = allsides_links[int(start_link_number):]

    data_links = {'right': set(), 'left': set(), 'center': set()}

    print("----- Scraping news article links from", len(allsides_links), "internal links -----")
    for i, url in enumerate(allsides_links):
        if start_link_number is not None:
            print(i + int(start_link_number),":", url)
        else:
            print(i,":", url)

        r_ = requests.get(url)
        soup_ = BeautifulSoup(r_.text, 'html.parser')

        for div in soup_.find_all("div", {"class": "article-bias"}):
            for img in div.find_all("img", alt=True):
                rating = img['alt']

        for div in soup_.find_all("div", {"class": "read-more-story"}):
            for link in div.find_all("a"):
                if rating == 'AllSides Media Bias Rating: Lean Right' or rating == 'AllSides Media Bias Rating: Right':
                    data_links['right'].add(link.get('href'))
                elif rating == 'AllSides Media Bias Rating: Lean Left' or rating == 'AllSides Media Bias Rating: Left':
                    data_links['left'].add(link.get('href'))
                elif rating == 'AllSides Media Bias Rating: Center':
                    data_links['center'].add(link.get('href'))

        if i % 60 == 0:
            files = ['data/right.txt', 'data/left.txt', 'data/center.txt']
            for file in files:
                with open(file, 'r') as f:
                    label = file[5:].replace(".txt", "")
                    f.writelines(["%s\n" % item  for item in list(data_links[label])])
            data_links = {'right': set(), 'left': set(), 'center': set()}
            

        # Required in allsides.com robots.txt
        # time.sleep(10)
    
    files = ['data/right.txt', 'data/left.txt', 'data/center.txt']
    for file in files:
        with open(file, 'r') as f:
            label = file[5:].replace(".txt", "")
            f.writelines(["%s\n" % item  for item in list(data_links[label])])


def scrape_single_text(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup.text

scrape_vec = np.vectorize(scrape_single_text)

def scrape_raw_text():
    files = ['data/right.txt', 'data/left.txt', 'data/center.txt']
    frames = []
    for file in files:
        with open(file, 'r') as f:
            label = file[5:].replace(".txt", "")

            lines = f.readlines()
            lines = [line.strip().replace('\n', '') for line in lines]
            df = pd.DataFrame({'urls':lines, 'label': [label]*len(lines)})
            frames.append(df)
    
    shortest_length = min(frames[0].shape[0], frames[1].shape[0], frames[2].shape[0])

    whitelisted = [['www.foxnews.com', 'www.washingtonexaminer.com', 'www.nationalreview.com', 'nypost.com', 'www.washingtontimes.com'], ['www.nytimes.com', 'www.cnn.com', 'www.politico.com', 'www.nbcnews.com', 'www.npr.org'], ['www.axios.com', 'www.wsj.com', 'www.csmonitor.com', 'apnews.com', 'www.pewresearch.org']]
    
    urls = [[], [], []]
    for i, df in enumerate(frames):
        for url in df['urls']:
            domain = re.findall(r"\bhttps://\b(.*?)\b/\b", url)
            if len(domain) == 0:
                domain = re.findall(r"\bhttp://\b(.*?)\b/\b", url)
            
            if len(domain) > 0:
                if domain[0] in whitelisted[i]:
                    urls[i].append(url)
    
    labels_names = ['right', 'center']
    urls = [urls[0], urls[2]]
    for label_, url_list in enumerate(urls):

        labels = [labels_names[label_]] * len(url_list)
        text = []
        print(labels_names[label_])
        print(len(url_list))

        for i, url in enumerate(url_list):
            print(i, url)
            try:
                r = requests.get(url)
                
            except requests.exceptions.RequestException:
                text.append('error')
                continue

            if r.status_code != 404 and r.status_code != 403:
                soup = BeautifulSoup(r.text, 'html.parser')
                text.append(str(soup.text).rstrip().replace('\n', ' '))
            else:
                # print('error')
                text.append('error')
        
        sub_df = pd.DataFrame({'urls': url_list, 'label': labels, 'text': text})
        sub_df.to_csv("data/mini_allsides/" + labels_names[label_] + "_mini.csv", index=False)


    
    # for label_, df in enumerate(balanced_frames[1:]):
    #     print(df['label'].head())
    #     label_ = label_ + 1
    #     print(labels[label_])
    #     text = []
    #     num_error = 0
    #     df = df.sample(n=500)
    #     start = time.time()
    #     for i, url in enumerate(df['urls']):
    #         # if i % 50 == 0:
    #         #     print('iteration ' + str(i) + ': ' + str((i/df.shape[0]) *100) + " percent done in " + str(time.time() - start) + " seconds")
    #         #     start = time.time()

    #         if 'www.washingtonpost.com' not in url and 'www.miamiherald.com' not in url and 'www.usnews.com' not in url:
    #             print(i, url, time.time() - start)
    #             start = time.time()            
    #             try:
    #                 r = requests.get(url)
                
    #             except requests.exceptions.RequestException:
    #                 text.append('error')
    #                 continue

    #             if r.status_code != 404 and r.status_code != 403:
    #                 soup = BeautifulSoup(r.text, 'html.parser')
    #                 text.append(str(soup.text).rstrip().replace('\n', ' '))
    #             else:
    #                 # print('error')
    #                 num_error += 1
    #                 text.append('error')
            
    #         else:
    #             print(i, url, time.time() - start)
    #             start = time.time()
    #             text.append("error")

    #     print(num_error / df.shape[0])
    #     df['text'] = text
    #     df.to_csv("data/" + labels[label_] + ".csv", index=False)


if __name__ == "__main__":
    
    # scrape_allsides_internal_links()
    if sys.argv[1] == 'internal':
        if len(sys.argv) > 2:
            scrape_allsides_internal_links(sys.argv[2])
        else:
            scrape_allsides_internal_links()

    # scrape_news_article_links()
    elif sys.argv[1] == 'articles':
        if len(sys.argv) > 3:
            scrape_news_article_links(sys.argv[2], sys.argv[3])
        else:
            scrape_news_article_links(sys.argv[2])
    
    # scrape_raw_text()
    elif sys.argv[1] == 'text': 
        scrape_raw_text()