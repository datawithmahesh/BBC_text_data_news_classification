# BBC Text Data News Classification
# pip install nltk
import pandas as pd
import pickle as pk
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Page configuration
st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="centered",
)

# Title section with styling
st.markdown(
    """
    <div style="background-color:#4CAF50;padding:20px;border-radius:10px">
    <h1 style="color:white;text-align:center;">📰 BBC News Classification</h1>
    <p style="color:white;text-align:center;">Predict the category of your news article using NLP</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load model
load_model = pk.load(open("bbc_text_data_news_classify.pickle", 'rb'))

# NLTK setup
nltk.download('stopwords')
words = stopwords.words("english")
stemmer = PorterStemmer()

# Input section with box
st.markdown(
    """
    <div style="background-color:#f2f2f2;padding:20px;border-radius:10px;">
    <h3 style="color:#333;">Enter your news text below:</h3>
    </div>
    """,
    unsafe_allow_html=True
)

news = st.text_area("", height=150, placeholder="Type or paste your news article here...")

# Button in center
if st.button("Predict 🟢"):
    if news.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Put text in dataframe
        news_data = {'predict_news':[news]}
        news_data_df = pd.DataFrame(news_data)

        # Clean text
        news_data_df['predict_news'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), news_data_df['predict_news']))
        news_data_df['predict_news'] = news_data_df['predict_news'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

        # Predict result
        result = load_model.predict(news_data_df['predict_news'])

        # Show result in fancy box
        st.markdown(
            f"""
            <div style="background-color:#4CAF50;padding:20px;border-radius:10px;margin-top:20px">
            <h2 style="color:white;text-align:center;">Predicted News Category:</h2>
            <h1 style="color:white;text-align:center;">{result[0]}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer section
st.markdown(
    """
    <hr>
    <p style="text-align:center;color:gray;">This project is done by <b>Mahesh Thapa</b></p>
    """,
    unsafe_allow_html=True
)





# #BBC_text_data_news_categories
# #pip install nltk
# import pandas as pd
# import pickle as pk
# import streamlit as st
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# st.write("BBC data to news classification")

# load_model = pk.load(open("bbc_text_data_news_classify.pickle", 'rb'))

# nltk.download('stopwords')
# words = stopwords.words("english")
# stemmer = PorterStemmer()

# news = st.text_area("Enter your news:--")

# if st.button("predict"):
#    # df = pd.DataFrame({
#    #    'cleaned':[text]
#    #    })  we can write the code of 3 lines above and continue from else: line  or simply use if else condtion to modify it
#       #  sentiment = input("Enter text = ") which is already given just before the dataframe
#    if news.strip() == "":
#       st.write("⚠️ Please enter some text")
#    else:
#       # Put text in dataframe
#       news_data = {'predict_news':[news]}
#       news_data_df = pd.DataFrame(news_data)

#       # Clean text
#       news_data_df['predict_news'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), news_data_df['predict_news']))
#       news_data_df['predict_news'] = news_data_df['predict_news'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

#       # Predicticting result 
#       # predict_news_cat = load_model.predict(sentiment_data_df['predict_sentiments'])
#       result = load_model.predict(news_data_df['predict_news'])

#       # Show result
#       #  st.write("Predicted sentiment category = ",predict_news_cat[0])
#       st.write("Predicted news category = ",result[0])




# st.write("This project is done by Mahesh Thapa")