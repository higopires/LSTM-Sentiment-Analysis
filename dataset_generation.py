############################################ GERAÇÃO DO DATASET ######################################################

from finpie import NewsData

# VALE
vale_news = NewsData('VALE', 'vale')
vale_news.filterz = [ 'vale', 'mining', 'energy' ]
vale_barrons = vale_news.barrons(datestop = '2020-11-23')
vale_barrons = vale_barrons.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
vale_cnbc = vale_news.cnbc(datestop = '2020-11-23')
vale_cnbc = vale_cnbc.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
vale_ft = vale_news.ft(datestop = '2020-11-23')
vale_ft = vale_ft.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
vale_nyt = vale_news.nyt(datestop = '2020-11-23')
vale_nyt = vale_nyt.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
vale_reuters = vale_news.reuters()
vale_reuters = vale_reuters.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
vale_sa = vale_news.seeking_alpha(datestop = '2020-11-23')
vale_sa = vale_sa.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
vale_wsj = vale_news.wsj(datestop = '2020-11-23')
vale_wsj = vale_wsj.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)

# PBR
pbr_news = NewsData('PBR', 'petrobras')
pbr_news.filterz = [ 'petrobras', 'oil', 'energy' ]
pbr_barrons = pbr_news.barrons(datestop = '2020-11-23')
pbr_barrons = pbr_barrons.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
pbr_cnbc = pbr_news.cnbc(datestop = '2020-11-23')
pbr_cnbc = pbr_cnbc.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
pbr_ft = pbr_news.ft(datestop = '2020-11-23')
pbr_ft = pbr_ft.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
pbr_nyt = pbr_news.nyt(datestop = '2020-11-23')
pbr_nyt = pbr_nyt.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
pbr_reuters = pbr_news.reuters()
pbr_reuters = pbr_reuters.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
pbr_sa = pbr_news.seeking_alpha(datestop = '2020-11-23')
pbr_sa = pbr_sa.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
pbr_wsj = pbr_news.wsj(datestop = '2020-11-23')
pbr_wsj = pbr_wsj.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)

# ITUB
itub_news = NewsData('ITUB', 'itau')
itub_news.filterz = [ 'itau', 'banking' ]
itub_barrons = itub_news.barrons(datestop = '2020-11-23')
itub_barrons = itub_barrons.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
itub_cnbc = itub_news.cnbc(datestop = '2020-11-23')
itub_cnbc = itub_cnbc.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
itub_ft = itub_news.ft(datestop = '2020-11-23')
itub_ft = itub_ft.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
itub_nyt = itub_news.nyt(datestop = '2020-11-23')
itub_nyt = itub_nyt.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
itub_reuters = itub_news.reuters()
itub_reuters = itub_reuters.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
itub_sa = itub_news.seeking_alpha(datestop = '2020-11-23')
itub_sa = itub_sa.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
itub_wsj = itub_news.wsj(datestop = '2020-11-23')
itub_wsj = itub_wsj.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)

# ABEV
abev_news = NewsData('ABEV', 'ambev')
abev_news.filterz = [ 'ambev', 'beverage', 'brewing' ]
abev_barrons = abev_news.barrons(datestop = '2020-11-23')
abev_barrons = abev_barrons.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
abev_cnbc = abev_news.cnbc(datestop = '2020-11-23')
abev_cnbc = abev_cnbc.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
abev_ft = abev_news.ft(datestop = '2020-11-23')
abev_ft = abev_ft.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
abev_nyt = abev_news.nyt(datestop = '2020-11-23')
abev_nyt = abev_nyt.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
abev_reuters = abev_news.reuters()
abev_reuters = abev_reuters.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
abev_sa = abev_news.seeking_alpha(datestop = '2020-11-23')
#abev_sa = abev_sa.drop(['link','description','tag','author','date_retrieved','comments',
#                            'newspaper','search_term','id','source'], axis=1)
abev_wsj = abev_news.wsj(datestop = '2020-11-23')
abev_wsj = abev_wsj.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)

# BBD
bbd_news = NewsData('BBD', 'bradesco')
bbd_news.filterz = [ 'bradesco', 'banking' ]
bbd_barrons = bbd_news.barrons(datestop = '2020-11-23')
bbd_barrons = bbd_barrons.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
bbd_cnbc = bbd_news.cnbc(datestop = '2020-11-23')
bbd_cnbc = bbd_cnbc.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
bbd_ft = bbd_news.ft(datestop = '2020-11-23')
bbd_ft = bbd_ft.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
bbd_nyt = bbd_news.nyt(datestop = '2020-11-23')
bbd_nyt = bbd_nyt.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
bbd_reuters = bbd_news.reuters()
bbd_reuters = bbd_reuters.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
bbd_sa = bbd_news.seeking_alpha(datestop = '2020-11-23')
bbd_sa = bbd_sa.drop(['link','description','tag','author','date_retrieved','comments',
                            'newspaper','search_term','id','source'], axis=1)
bbd_wsj = bbd_news.wsj(datestop = '2020-11-23')
#bbd_wsj = bbd_wsj.drop(['link','description','tag','author','date_retrieved','comments',
#                            'newspaper','search_term','id','source'], axis=1)

dataframes = [abev_barrons,abev_cnbc,abev_ft,abev_nyt,abev_reuters,abev_wsj,bbd_barrons,bbd_cnbc,
              bbd_ft,bbd_nyt,bbd_reuters,bbd_sa,itub_barrons,itub_cnbc,itub_ft,itub_reuters,itub_sa,
              itub_wsj,pbr_barrons,pbr_cnbc,pbr_ft,pbr_nyt,pbr_reuters,pbr_sa,pbr_wsj,vale_barrons,
              vale_cnbc,vale_ft,vale_nyt,vale_reuters,vale_sa,vale_wsj]

import pandas as pd

result = pd.concat(dataframes)
result  = result.sort_values(by=['date'],ascending=False)
result = result.drop_duplicates(subset=['headline'])
result = result.reset_index()

############################################ GERAÇÃO DOS LABELS DE SENTIMENTO DAS HEADLINES ##########################


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

vader = SentimentIntensityAnalyzer()

import csv

# Source: https://towardsdatascience.com/https-towardsdatascience-com-algorithmic-trading-using-sentiment-analysis-on-news-articles-83db77966704

# # stock market lexicon
stock_lex = pd.read_csv('stock_lex.csv')
stock_lex['sentiment'] = (stock_lex['Aff_Score'] + stock_lex['Neg_Score'])/2
stock_lex = dict(zip(stock_lex.Item, stock_lex.sentiment))
stock_lex = {k:v for k,v in stock_lex.items() if len(k.split(' '))==1}
stock_lex_scaled = {}
for k, v in stock_lex.items():
     if v > 0:
         stock_lex_scaled[k] = v / max(stock_lex.values()) * 4
     else:
         stock_lex_scaled[k] = v / min(stock_lex.values()) * -4

# # # Loughran and McDonald
positive = []
with open('lm_positive.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        positive.append(row[0].strip())
    
negative = []
with open('lm_negative.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        entry = row[0].strip().split(" ")
        if len(entry) > 1:
            negative.extend(entry)
        else:
            negative.append(entry[0])

final_lex = {}
final_lex.update({word:2.0 for word in positive})
final_lex.update({word:-2.0 for word in negative})
final_lex.update(stock_lex_scaled)
final_lex.update(vader.lexicon)
vader.lexicon = final_lex

scores = result['headline'].apply(vader.polarity_scores).tolist()
scores_df = pd.DataFrame(scores)
result = result.join(scores_df, rsuffix='_right')
result = result.drop(['neg', 'neu', 'pos', 'date', 'ticker'], axis=1)

result.loc[result.compound <= -0.05, "compound"] = 'Negative' # negative
result.loc[result.compound >= 0.05, "compound"] = 'Positive' # positive
result.loc[(result.compound != 1) & (result.compound != 0), "compound"] = 'Neutral' # neutral
result = result.rename(columns={"compound": "label"})
result.label = result.label.astype(string)

with open('dataset.txt', 'w') as f: result.to_string(f, col_space=4,index=None)

# Após esse passo, corrigir no editor de texto a estrutura do texto do dataset, de modo que ele fique com a seguinte estrutura: headline -> tabulação -> label da headline (pos, neg, neu)

######################################################################################################################