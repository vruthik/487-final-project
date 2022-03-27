import pandas as pd
import re


def get_domain(url):
    domain = re.findall(r"\bhttps://\b(.*?)\b/\b", url)
            
    if len(domain) == 0:
        domain = re.findall(r"\bhttp://\b(.*?)\b/\b", url)
    
    if len(domain) > 0:
        return domain[0]
    
    else:
        return ''


def ny_times_preprocess(text):
    starts = re.findall(r'Updated', text)
    if len(starts) == 0:
        starts = re.findall(r'Supported byContinue reading the main story', text)
        print(starts)
        # print(text[:500])
    
    end_indexes = []
    for match in re.finditer(r'—', text):
        end_indexes.append(match.start())
    if len(end_indexes) > 0:
        text = text[:end_indexes[-1]]
    
    else:
        for match in re.finditer(r'AdvertisementContinue', text):
            end_indexes.append(match.start())
        text = text[:end_indexes[-1]]
    return text

def apnews_preprocess(text, url):
    starts = []
    print(text[:1500])
    for match in re.finditer(re.escape(url), text):
        starts.append(match.end())
    
    print(starts)
    print(text[starts[0]:500])
    print("-----------------------------")

df = pd.read_csv("data/mini_allsides/center_mini.csv")
count = 0
for row in df.iterrows():
    if get_domain(row[1][0]) == 'apnews.com':
        output = apnews_preprocess(row[1][2], row[1][0])

