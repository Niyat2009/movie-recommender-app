import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies_dataset_with_images.csv")
movies['overview'] = movies['overview'].fillna('')
movies['combined'] = movies['genre'] + ' ' + movies['overview'] + ' ' + movies['actors']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title, n=5):
    title = title.strip().title()
    if title not in movies['title'].values:
        return []
    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    recs = [movies.iloc[i[0]]['title'] for i in scores]
    return recs

st.set_page_config(page_title="🎬 Movie & Series Recommender", page_icon="🎥", layout="wide")
st.title("🎬 IMDb-style Movie & Series Recommender")
st.write("Search a movie or series to see poster, rating, and recommendations!")

movie_input = st.text_input("Enter a movie or series name:")

if st.button("Show Movie & Recommendations"):
    title_search = movie_input.strip().title()
    if title_search not in movies['title'].values:
        st.error("⚠️ Movie or Series not found!")
    else:
        movie_row = movies[movies['title'] == title_search].iloc[0]
        st.image(movie_row['image_url'], width=300)
        st.markdown(f"**Title:** {movie_row['title']}")
        st.markdown(f"**Rating:** {'⭐'*int(movie_row['rating'])} ({movie_row['rating']}/10)")
        st.markdown(f"**Genre:** {movie_row['genre']}")
        st.markdown(f"**Overview:** {movie_row['overview']}")
        st.markdown(f"**Actors:** {movie_row['actors']}")

        st.subheader("🎞️ Recommended Titles")
        recs = recommend(title_search)
        cols = st.columns(len(recs))
        for i, r in enumerate(recs):
            rec_row = movies[movies['title'] == r].iloc[0]
            with cols[i]:
                st.image(rec_row['image_url'], width=150)
                st.markdown(f"**{rec_row['title']}**")
                st.markdown(f"⭐ {rec_row['rating']}/10")
