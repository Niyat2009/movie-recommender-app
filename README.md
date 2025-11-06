# 🎬 Movie & Series Recommender App

An **IMDb-style Movie & Series Recommendation System** built with **Streamlit** and **scikit-learn**.

## ✨ Features
- Dataset of 1000+ movies and series (Hollywood + Bollywood)
- Content-based recommendations using TF-IDF + Cosine Similarity
- Simple Streamlit web interface
- Ready for free deployment on [Streamlit Cloud](https://share.streamlit.io)

## 🧠 How it Works
The app analyses movie genres, overviews, and actors to find similar titles using cosine similarity.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501)

## ☁️ Deploy on Streamlit Cloud
1. Push these files to GitHub.
2. Visit [https://share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → choose your repo → select `app.py`.
4. Click **Deploy**.

Your app will be live at:  
`https://movie-recommender-yourusername.streamlit.app`

---

👨‍💻 Built by [Your Name] with ❤️
