#BBC_text_data_news_categories
#pip install nltk
import pandas as pd
import pickle as pk
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.write("BBC data to news classification")

load_model = pk.load(open("bbc_text_data_news_classify.pickle", 'rb'))

nltk.download('stopwords')
words = stopwords.words("english")
stemmer = PorterStemmer()

news = st.text_area("Enter your news:--")

if st.button("predict"):
   # df = pd.DataFrame({
   #    'cleaned':[text]
   #    })  we can write the code of 3 lines above and continue from else: line  or simply use if else condtion to modify it
      #  sentiment = input("Enter text = ") which is already given just before the dataframe
   if news.strip() == "":
      st.write("⚠️ Please enter some text")
   else:
      # Put text in dataframe
      news_data = {'predict_news':[news]}
      news_data_df = pd.DataFrame(news_data)

      # Clean text
      news_data_df['predict_news'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), news_data_df['predict_news']))
      news_data_df['predict_news'] = news_data_df['predict_news'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

      # Predicticting result 
      # predict_news_cat = load_model.predict(sentiment_data_df['predict_sentiments'])
      result = load_model.predict(news_data_df['predict_news'])

      # Show result
      #  st.write("Predicted sentiment category = ",predict_news_cat[0])
      st.write("Predicted news category = ",result[0])




st.write("This project is done by Mahesh Thapa")