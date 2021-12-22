import streamlit as st
# Base packages
import pandas as pd
import numpy as np
import datetime
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from pprint import pprint
from gensim.models import LdaModel
import pyLDAvis
from pyLDAvis import gensim
from streamlit import components
from streamlit import caching
import pickle
from wordcloud import WordCloud
# Command to launch Ngrok: ./ngrok http 8501

st.set_page_config(layout='wide')

st.header("Sentiment Analyzer")
st.sidebar.markdown("*Last update: 11/12/2021*")
st.sidebar.markdown("---")
st.sidebar.slider('Slide', min_value=3, max_value=12)
#st.sidebar.header("Ressources utiles")
#st.sidebar.markdown("Numéro d'urgence 1: **78 172 10 81**")
# I. Dataframe
df = "../data/reviews_tgtg.pkl" #path to be written
with open(df, 'rb') as f:
  df = pickle.load(f)
# II. Summary of the number of cases

with st.form('my_form'): #reduire la taille en centrant
    st.text_input('Type and test your tweet review')

    uploaded_file = st.file_uploader('Or choose a file', type=["csv","txt"])
    if uploaded_file is not None:
        file = pd.read_csv(uploaded_file)
        prepro_file = preprocessing(file.iloc[:,0])
        predictions = 

    button = st.form_submit_button('Predict')
    #if button:
        #st.st.markdown("""# Sentiment Analyzer {value_predict} :star: """)
        #st.metric(....)

st.markdown('### Estimator result : ')
st.container()

st.markdown("---")
col1,col2 = st.columns(2)
with col1:
    st.subheader("Reviews Distribution per Platform")
    plt.figure(figsize=(24,16))
    plt.pie(df['source'].value_counts(), labels = ['Google','Apple','Trustpilot'],autopct='%1.0f%%')
    plt.title('Reviews Distribution per Platform')
    st.pyplot(plt)
with col2:
    st.subheader("Reviews Distribution per Rating")
    plt.figure(figsize=(24,16))
    sns.set_theme(style="whitegrid")
    sns.countplot(x='rating', data=df)
    plt.title('Reviews Distribution per Rating')
    st.pyplot(plt)
# III. Interactive map
st.markdown("---")

def preprocessing(text):
    text=text.lower()
    tokens = word_tokenize(text)
    tokens_no_punctuation = [t for t in tokens if t.isalpha()]
    stop_words = stopwords.words('english')
    tokens_no_stop = [t for t in tokens_no_punctuation if t not in stop_words]
    stemmer = PorterStemmer()
    token_stem = [stemmer.stem(t) for t in tokens_no_stop]
    return token_stem
# preprocessing applied on the column concerned + output stored in new column
df['preprocessed_review']= df['review_content'].apply(preprocessing)



def wordcloud(text):

    text =' '.join([str(item) for item in text])
    wordcloud_words = " ".join(text)
    wordcloud = WordCloud(
        height=300, width=500, background_color="black", random_state=100,
    ).generate(wordcloud_words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("cloud.jpg")
    img = Image.open("cloud.jpg")
    return img

# calling the function to create the word cloud
img = wordcloud(df['preprocessed_review'])
st.success(
    "Word Cloud"
)
st.image(img)

st.markdown("---")
# Create a corpus
corpus = df['preprocessed_review']
# Compute the dictionary: this is a dictionary mapping words and their corresponding numbers for later visualisation
id2word = Dictionary(corpus)
# Create a BOW
bow = [id2word.doc2bow(line) for line in corpus]  # convert corpus to BoW format
# Instanciate a TF-IDF
tfidf_model = TfidfModel(bow)
# Compute the TF-IDF
tf_idf_gensim = tfidf_model[bow]
# compute LDA
lda1 = LdaModel(corpus=tf_idf_gensim, num_topics=4, id2word=id2word, passes=10, random_state=0)
# Visualize the different topics
vis = pyLDAvis.gensim.prepare(topic_model=lda1, corpus=bow, dictionary=id2word)
html_string = pyLDAvis.prepared_data_to_html(vis)
#from streamlit import components
"""@st.cache
button = components.v1.html(html_string, width=1300, height=800)
if button:"""
