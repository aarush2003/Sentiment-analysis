from urllib.request import urlopen, Request 
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

main_url = "https://finviz.com/quote.ashx?t="
stocks = ["AMZN", "FB", "AMC", "GME"]
articles = {}
parsed_html = []

for stock in stocks:
    stock_url = main_url + stock
    request = Request(url= stock_url, headers={"user-agent": "app"})
    request_response = urlopen(request) 
    
    raw_html = BeautifulSoup(request_response, 'html')
    
    loa = raw_html.find(id='news-table')
    articles[stock] = loa


for stock, loa in articles.items(): 
    for tr in loa.findAll('tr'):
        article_title = tr.a.get_text()
        dates = tr.td.get_text().split(' ')

        if(len(dates) > 1):
            article_date = dates[0]
            article_time = dates[1]
        else:
            article_time = dates[0]

        parsed_html.append([stock, article_date, article_time, article_title]) 

pandas_data = pd.DataFrame(parsed_html, columns=['stock', 'date','time', 'title'])

analyzer = SentimentIntensityAnalyzer()

sentiment_calculator = lambda x: analyzer.polarity_scores(x)['compound']
pandas_data['score'] = pandas_data['title'].apply(sentiment_calculator)
pandas_data['date'] = pd.to_datetime(pandas_data.date).dt.date

plt.figure(figsize=(10,8))
mean = pandas_data.groupby(['stock', 'date']).mean().unstack()
mean = mean.xs('score', axis="columns")
mean.plot(kind='bar')
plt.show()
