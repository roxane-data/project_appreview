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
import os
import pickle
from wordcloud import WordCloud
from PIL import Image
import PIL
# Command to launch Ngrok: ./ngrok http 8501

st.set_page_config(layout='wide')


# Part 0 => Dataframe

ORIGINAL_DF_PATH = "../data/reviews_tgtg.pkl"
PROCESSED_DF_PATH = "../data/reviews_tgtg_processed.pkl"

def get_data(path):
    with open(path, 'rb') as f:
        df = pickle.load(f)
    return df

def save_data(path):
    with open(path, 'wb') as f:
        pickle.dump(unprocessed_df, f)

def preprocessing(text):
    text=text.lower()
    tokens = word_tokenize(text)
    tokens_no_punctuation = [t for t in tokens if t.isalpha()]
    stop_words = stopwords.words('english')
    tokens_no_stop = [t for t in tokens_no_punctuation if t not in stop_words]
    stemmer = PorterStemmer()
    token_stem = [stemmer.stem(t) for t in tokens_no_stop]
    return token_stem

preprocess_button = st.button('Preprocess')
if preprocess_button:
    unprocessed_df = get_data(ORIGINAL_DF_PATH)
    unprocessed_df['preprocessed_review'] = unprocessed_df['review_content'].apply(preprocessing)
    save_data(PROCESSED_DF_PATH)

if not os.path.isfile(PROCESSED_DF_PATH):
    st.warning("Please preprocess the reviews !")
    st.stop()

df = get_data(PROCESSED_DF_PATH)



channels = df['source'].drop_duplicates()
make_choice = st.sidebar.selectbox('Select channel:', channels)
period = df['date'].loc[df["source"] == make_choice]
period_choice = st.sidebar.selectbox('Select time range', period)
rating = df['rating'].loc[(df["source"] == make_choice) & (df["date"] == period_choice)]
rating_choice = st.sidebar.selectbox('Select rating', rating)

st.sidebar.markdown("*Last update: 11/12/2021*")
st.sidebar.markdown("---")
#st.sidebar.slider('Slide', min_value=3, max_value=12)




# Part 1 => Tweet scoring estimator based on classification
st.markdown("# Tweet Scorer")

# tweet to be typed down or downloaded
with st.form('my_form'): #reduire la taille en centrant
    st.markdown('### Type and test your tweet')
    st.text_input('Write a review in the field below')

    uploaded_file = st.file_uploader('Or choose a file', type=["csv","txt"])
    if uploaded_file is not None:
        file = pd.read_csv(uploaded_file)
        prepro_file = preprocessing(file.iloc[:,0])
        #predictions =

    button = st.form_submit_button('Predict')
    #if button:
        #st.st.markdown("""# Sentiment Analyzer {value_predict} :star: """)
        #st.me

st.markdown('### Estimated score : ')

st.markdown("---")


# Part 2 => Data viz
st.markdown("# Data viz")

# channel name depending on the channel selected in the sidebar
st.markdown("## Channel selected")
time_source_df = df.loc[(df["source"] == make_choice) & (df["date"] == period_choice)]
col1,col2 = st.columns(2)
with col1:
    st.metric(label="% Total review", value="'{}'".format(time_source_df['source']/len(df)*100)) # metric 1 of the channel selected
    st.metric(label="Avg Score", value="'{}'".format(time_source_df['rating'].mean()))# metric 2 of the channel selected

with col2:
    st.subheader("Reviews Distribution per Rating") # distribution chart of the channel selected
    plt.figure(figsize=(24,16))
    sns.set_theme(style="whitegrid")
    sns.countplot(x='rating', data=time_source_df)
    st.pyplot(plt)

st.markdown("---")



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
st.subheader('Wordcloud')
img = wordcloud(df['preprocessed_review'])
st.image(img)
st.markdown("---")




# creating room for pyLDAvis
@st.cache # to save time

# 2 functions to create pyLDAvis chart into html format and display it

def pyLDAvis_get(number): # to create pylDavis and define the nb of topics to show
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
    # compute LDA according to nb of topics selected
    lda1 = LdaModel(corpus=tf_idf_gensim, num_topics=number, id2word=id2word, passes=10, random_state=0)
    # Visualize the different topics
    vis = pyLDAvis.gensim.prepare(topic_model=lda1, corpus=bow, dictionary=id2word)
    html_string = pyLDAvis.prepared_data_to_html(vis) #to convert vis inot html format to display in streamlit
    return html_string

 #to display pyLDAvis chart


with st.expander("More ..."):
    st.subheader("Key insights to improve product")
    st.markdown("#### Most frequent associated words used in reviews")
    # tfidf n_gram

    st.markdown("#### Main trend topics")
    button = st.button('pyLDAvis')
    if button:
        input = st.number_input(label = 'Type down the number of topics you would like to look @',step=1, min_value=3) #to get nb of topics to display on chart
        components.v1.html(pyLDAvis_get(input), width=1300, height=800)
