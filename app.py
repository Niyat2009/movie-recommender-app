import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies_dataset.csv")

# Prepare data
movies['overview'] = movies['overview'].fillna('')
movies['combined'] = movies['genre'] + ' ' + movies['overview'] + ' ' + movies['actors']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend(title, n=5):
    title = title.strip().title()
    if title not in movies['title'].values:
        return ["⚠️ Movie or Series not found in database."]
    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    recs = [movies.iloc[i[0]]['title'] for i in scores]
    return recs

# Streamlit UI
st.set_page_config(page_title="🎬 Movie & Series Recommender", page_icon="🎥")
st.title("🎬 IMDb-style Movie & Series Recommender")
st.write("Discover similar movies and series from Hollywood and Bollywood!")

movie_input = st.text_input("Enter a movie or series name (e.g., Inception, Dangal, Sacred Games):")

if st.button("Recommend"):
    with st.spinner("Finding similar titles..."):
        results = recommend(movie_input)
        st.subheader("🎞️ Recommendations:")
        for r in results:
            st.write("👉", r)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & scikit-learn | Dataset inspired by IMDb")
