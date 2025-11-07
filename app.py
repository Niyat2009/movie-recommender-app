
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dark theme CSS
dark_css = """
<style>
body { background-color: #0e1117; color: #e6eef8; }
.stButton>button { background-color: #1f6feb; color: white; }
.css-1d391kg { color: #e6eef8; }  /* streamlit class may vary */
h1, h2, h3, h4, h5, h6 { color: #e6eef8; }
.stSelectbox, .stTextInput { background-color: #10161a; color: #e6eef8; }
</style>
"""

st.set_page_config(page_title="🎬 Movie Recommender (Dark)", layout="wide")
st.markdown(dark_css, unsafe_allow_html=True)

movies = pd.read_csv("movies_dataset.csv")
movies['overview'] = movies['overview'].fillna('')
movies['combined'] = movies['genre'] + ' ' + movies['overview'] + ' ' + movies['actors']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title, n=5):
    try:
        idx = movies[movies['title'].str.lower()==title.lower()].index[0]
    except IndexError:
        return []
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return [movies.iloc[i[0]]['title'] for i in scores]

st.title("🎬 Movie & Series Recommender (Dark Mode)")
st.write("Search a movie (type name) or choose from the list. Click 'Show' to display poster and rating.")

col1, col2 = st.columns([2,1])
with col1:
    query = st.text_input("Search by title (case-insensitive):")
    if st.button("Show"):
        if not query.strip():
            st.warning("Please enter a movie or series title.")
        else:
            matches = movies[movies['title'].str.lower()==query.strip().lower()]
            if matches.empty:
                st.error("Movie/Series not found. Try selecting from the list below.")
            else:
                movie_row = matches.iloc[0]
                st.image(movie_row['image_url'], width=350)
                st.markdown(f"### {movie_row['title']}  ({movie_row['year']})")
                st.markdown(f"**Rating:** {movie_row['rating']}/10")
                st.markdown(f"**Genre:** {movie_row['genre']}")
                st.markdown(f"**Actors:** {movie_row['actors']}")
                st.markdown(f"**Overview:** {movie_row['overview']}")

with col2:
    st.subheader("Or pick from list")
    selected = st.selectbox("Choose a title:", options=sorted(movies['title'].unique().tolist()))
    if st.button("Show Selected"):
        movie_row = movies[movies['title']==selected].iloc[0]
        st.image(movie_row['image_url'], width=250)
        st.markdown(f"### {movie_row['title']}  ({movie_row['year']})")
        st.markdown(f"**Rating:** {movie_row['rating']}/10")
        st.markdown(f"**Genre:** {movie_row['genre']}")
        st.markdown(f"**Actors:** {movie_row['actors']}")
        st.markdown(f"**Overview:** {movie_row['overview']}")

st.markdown("---")
st.subheader("Recommendations (for the shown title)")
active_title = None
if query.strip() and any(movies['title'].str.lower()==query.strip().lower()):
    active_title = query.strip()
else:
    active_title = selected if selected else None

if active_title:
    recs = recommend(active_title, n=5)
    if recs:
        cols = st.columns(len(recs))
        for i,r in enumerate(recs):
            rec_row = movies[movies['title']==r].iloc[0]
            with cols[i]:
                st.image(rec_row['image_url'], width=150)
                st.markdown(f"**{rec_row['title']}**")
                st.markdown(f"{rec_row['rating']}/10")
