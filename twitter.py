from credentials import *
import tweepy
import pandas as pd
import csv
import re

print(tweepy.__version__)


def get_senator_usernames():
    df = pd.read_csv("/Users/vruthikthakkar/Desktop/487-final-project/data/govtrack-stats-2020-senate-ideology.csv")

    # 40-20-40 Split
    df['label'] = ''
    df[:40]['label'] = 'left'
    df[40:60]['label'] = 'center'
    df[60:]['label'] = 'right'
    df.drop(columns=['district', 'id', 'bioguide_id'], inplace=True)
    df['name'] = df['name'].apply(lambda x: x.replace("b'", ""))
    df['name'] = df['name'].apply(lambda x: x.replace("'", ""))

    usernames = ["@lisamurkowski @SenDanSullivan @SenDougJones @SenShelby @JohnBoozman @SenTomCotton @SenMarkKelly @SenatorSinema @SenFeinstein @KamalaHarris @SenatorBennet @SenCoryGardner @SenBlumenthal @ChrisMurphyCT @SenatorCarper @ChrisCoons @marcorubio @SenRickScott @KLoeffler @sendavidperdue @maziehirono @SenBrianSchatz @SenJoniErnst @ChuckGrassley @MikeCrapo @SenatorRisch @SenDuckworth @SenatorDurbin @SenatorBraun @SenToddYoung @JerryMoran @SenPatRoberts @LeaderMcConnell @RandPaul @BillCassidy @SenJohnKennedy @SenMarkey @SenWarren @SenatorCardin @ChrisVanHollen @SenatorCollins @SenAngusKing @SenGaryPeters @SenStabenow @amyklobuchar @SenTinaSmith @RoyBlunt @SenHawleyPress @SenHydeSmith @SenatorWicker", "@SteveDaines @SenatorTester @SenatorBurr @SenThomTillis @SenKevinCramer @SenJohnHoeven @SenatorFischer @SenSasse @SenatorHassan @SenatorShaheen @CoryBooker @SenatorMenendez @MartinHeinrich @SenatorTomUdall @SenCortezMasto @SenJackyRosen @gillibrandny @SenSchumer @SenSherrodBrown @senrobportman @JimInhofe @SenatorLankford @SenJeffMerkley @RonWyden @SenBobCasey @SenToomey @SenJackReed @SenWhitehouse @LindseyGrahamSC @SenatorTimScott @SenatorRounds @SenJohnThune @SenAlexander @MarshaBlackburn @JohnCornyn @SenTedCruz @SenMikeLee @SenatorRomney @timkaine @MarkWarner @SenatorLeahy @SenSanders @SenatorCantwell @PattyMurray @SenatorBaldwin @SenRonJohnson @SenCapito @Sen_JoeManchin @SenJohnBarrasso @SenatorEnzi"]
    names = ["AK Murkowski", "AK Sullivan", "AL Jones", "AL Shelby", "AR Boozman", "AR Cotton", "AZ Kelly", "AZ Sinema", "CA Feinstein", "CA Harris", "CO Bennet", "CO Gardner", "CT Blumenthal", "CT Murphy", "DE Carper", "DE Coons", "FL Rubio", "FL Scott", "GA Loeffler", "GA Perdue", "HI Hirono", "HI Schatz", "IA Ernst", "IA Grassley", "ID Crapo", "ID Risch", "IL Duckworth", "IL Durbin", "IN Braun", "IN Young", "KS Moran", "KS Roberts", "KY McConnell", "KY Paul", "LA Cassidy", "LA Kennedy", "MA Markey", "MA Warren", "MD Cardin", "MD Van Hollen", "ME Collins", "ME King", "MI Peters", "MI Stabenow", "MN Klobuchar", "MN Smith", "MO Blunt", "MO Hawley", "MS Hyde-Smith", "MS Wicker","MT Daines", "MT Tester", "NC Burr", "NC Tillis", "ND Cramer", "ND Hoeven", "NE Fischer", "NE Sasse", "NH Hassan", "NH Shaheen", "NJ Booker", "NJ Menendez", "NM Heinrich", "NM Udall", "NV Masto", "NV Rosen", "NY Gillibrand", "NY Schumer", "OH Brown", "OH Portman", "OK Inhofe", "OK Lankford", "OR Merkley", "OR Wyden", "PA Casey", "PA Toomey", "RI Reed", "RI Whitehouse", "SC Graham", "SC Scott", "SD Rounds", "SD Thune", "TN Alexander", "TN Blackburn", "TX Cornyn", "TX Cruz", "UT Lee", "UT Romney", "VA Kaine", "VA Warner", "VT Leahy", "VT Sanders", "WA Cantwell", "WA Murray", "WI Baldwin", "WI Johnson", "WV Capito", "WV Manchin", "WY Barrasso", "WY Enzi"]
    usernames_cleaned = []
    for string in usernames:
        usernames_ = string.split("@")
        usernames_.remove('')
        usernames_ = [username.strip() for username in usernames_]
        usernames_cleaned.extend(usernames_)

    frames = []
    for i, name in enumerate(names):
        name = name.strip()
        name = name.split(" ")

        if name[1] == 'Van':
            name[1] = name[1] + " Hollen"

        username = usernames_cleaned[i]
        sub_df = df[(df['name'] == name[1]) & (df['state'] == name[0])]
        sub_df['username'] = username
        frames.append(sub_df)

    df = pd.concat(frames)

    return df

def get_user_id(username):
    client = tweepy.Client(bearer_token=get_bearer_token())
    user = client.get_user(username=username)
    return user.data.id

def get_user_tweets(username):

    client = tweepy.Client(bearer_token=get_bearer_token())
    
    # for tweets in tweepy.Cursor(client.get_users_tweets, id=get_user_id(username), tweet_fields=['context_annotations','created_at','geo'], count=3200):
    #     print(tweets.data)
    tweets = client.get_users_tweets(id=get_user_id(username), tweet_fields=['context_annotations','created_at','geo'], max_results=100, exclude=['retweets', 'replies'])
    tweets_text = []

    for tweet_text in tweets.data:
        tweets_text.append(tweet_text.text)

    return tweets_text
    
if __name__ == '__main__':
    senators = get_senator_usernames()
    with open('/Users/vruthikthakkar/Desktop/487-final-project/data/tweets.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['senator_name', 'senator_username', 'label', 'tweet'])
        i = 0
        for row in senators.iterrows():
            print(i)
            i += 1
            row_items = row[1]
            text = get_user_tweets(row_items['username'])
            
            for tweet in text:
                tweet = re.sub(' +', ' ', tweet)
                tweet = re.sub(r"http\S+", "", tweet)
                tweet = re.sub(r"www.\S+", "", tweet)
                tweet = re.sub("@[A-Za-z0-9_]+","", tweet)
                tweet = re.sub("#[A-Za-z0-9_]+","", tweet)

                writer.writerow([row_items['name'], row_items['username'], row_items['label'], tweet])
