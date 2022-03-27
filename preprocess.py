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
    # print(text[-1000:])
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

df = pd.read_csv("data/mini_allsides/left_mini.csv")
count = 0
for row in df.iterrows():
    if get_domain(row[1][0]) == 'www.nytimes.com':
        output = ny_times_preprocess(row[1][2])
        print(output[-500:])


