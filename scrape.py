import requests
from bs4 import BeautifulSoup
import sys
import pandas as pd
import pprint
import time

def scrape():
    pprinter = pprint.PrettyPrinter()
    # TODO: Add functionality to scrape allsides.com

    # topics = {
    #     '2022-elections': '2022%20elections'
    # }

    topics = ["2022 Elections", "Abortion", "Africa", "Agriculture", "Animal Welfare", "Arts and Entertainment", "Asia", "Australia", "Banking and Finance", "Bridging Divides", "Business", "Campaign Finance", "Campaign Rhetoric", "Capital Punishment and Death Penalty", "China", "CIA", "Civil Rights", "Coronavirus", "Criminal Justice", "Culture", "Cybersecurity", "DEA", "Defense and Security", "Defense Department", "Democratic Party", "Disaster", "Domestic Policy", "Economic Policy", "Economy and Jobs", "Education", "Elections", "Energy", "Environment", "EPA", "Ethnicity and Heritage", "Europe", "Fact Checking", "Fake News", "Family and Marriage", "FBI", "FDA", "Federal Budget", "Food", "Foreign Policy", "Free Speech", "General News", "George Floyd Protests", "Great Britain", "Gun Control and Gun Rights", "Healthcare", "History of Media Bias", "Holidays", "Homeland Security", "Housing and Homelessness", "Humor and Satire", "Immigration", "Impeachment", "Inequality", "ISIS", "Israel", "Justice", "Justice Department", "Kamala Harris", "Labor", "LGBTQ Issues", "Marijuana Legalization", "Media Bias", "Media Industry", "Mexico", "Middle East", "National Defense", "National Security", "North Korea", "NSA", "Nuclear Weapons", "Opioid Crisis", "Palestine", "People and Profit", "Polarization", "Politics", "Privacy", "Public Health", "Race and Racism", "Religion and Faith", "Republican Party", "Role of Government", "Russia", "Science", "Sexual Misconduct", "Social Security", "South Korea", "Sports", "State Department", "Supreme Court", "Sustainability", "Taxes", "Tea Party", "Technology", "Terrorism", "The Americas", "Trade", "Transportation", "Treasury", "US Census", "US Congress", "US Constitution", "US House", "US Military", "US Senate", "Veterans Affairs", "Violence in America", "Voting Rights and Voter Fraud", "Welfare", "White House", "Women's Issues", "World"]
    topics_url = {}
    for topic in topics:
        print(topic)
        topic = topic.split(" ")
        if len(topic) > 1:
            topics_url['-'.join(topic)] = "%20".join(topic)
        else:
            topics_url[topic[0]] = topic[0]

    # pprinter.pprint(topics_url)
    allsides_links = set()
    data_links = {'right': set(), 'left': set(), 'center': set()}

    for i, key in enumerate(topics_url.keys()):
        if i == 0:
            url = 'https://www.allsides.com/topics/' + key + '?search=' + topics_url[key]
            print(url)
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')

            for link in soup.find_all('a'):
                for div in link.find_all("div", {"class": "news-title"}):
                    if 'www.allsides.com' in link.get('href') and 'news-source' not in link.get('href'):
                        allsides_links.add(link.get('href'))

    allsides_links = list(allsides_links)
    print(len(allsides_links))
    for i, url in enumerate(allsides_links[:10]):
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

        time.sleep(10)

    print(data_links)


    # print(soup_)
            
                # for img in link.find_all('img', alt=True):
                #     rating = img['alt']
                #     print(rating, print(link))
                #     print("-------------")

                    # if rating != "" and 'AllSides Media Bias Rating' in rating:
                    #     if rating == 'AllSides Media Bias Rating: Lean Right' or rating == 'AllSides Media Bias Rating: Right':
                    #         allsides_links['right'].add(link.get('href'))
                    #     elif rating == 'AllSides Media Bias Rating: Lean Left' or rating == 'AllSides Media Bias Rating: Left':
                    #         allsides_links['left'].add(link.get('href'))
                    #     elif rating == 'AllSides Media Bias Rating: Center':
                    #         allsides_links['center'].add(link.get('href'))

            # pprinter.pprint(allsides_links)
                #     print()
                #     # for img in div.find_all('img', alt=True):
                #     #     print(img)
                #     print(div)
                # if 'AllSides Media Bias Rating: Center' in link['img alt']:
                #     print(link.image)
                #     print('-----------------------')

if __name__ == "__main__":
    # scrape(sys.argv[1])
    scrape()