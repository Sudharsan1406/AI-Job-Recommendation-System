import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sentence_transformers import SentenceTransformer, models

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"   # force safe CPU loading
)

df = pd.read_pickle("processed_jobs.pkl")

# Stack embeddings into array
job_embeddings = np.vstack(df['embedding'].values)


st.title("🔍 AI-Powered Job Recommendation System")
st.write("Find jobs that match your skills using NLP + Embeddings")

user_text = st.text_area(
    "Enter your skills:",
    placeholder="e.g., python, machine learning, sql, deep learning"
)


if st.button("Recommend Jobs"):
    if user_text.strip() == "":
        st.warning("Please enter your skills.")
    else:

        # Convert user text -> embedding
        user_embedding = model.encode(
            user_text,
            normalize_embeddings=True
        )

        # Similarity calculation
        similarity = cosine_similarity(
            user_embedding.reshape(1, -1),
            job_embeddings
        ).flatten()

        df['similarity'] = similarity

        # Ranking formula
        df['ranking_score'] = (
            0.7 * df['similarity'] +
            0.2 * df['exp_norm'] +
            0.1 * df['salary_norm']
        )

        top_jobs = df.sort_values("ranking_score", ascending=False).head(10)

        st.subheader("Top Job Recommendations")
        for _, row in top_jobs.iterrows():
            st.markdown(f"""
                **{row['title']}**  
                🌍 Location: {row['location']}  
                🧩 Experience: {row['formatted_experience_level']}  
                💰 Salary: {row['normalized_salary']}  
                🎯 Match Score: **{row['ranking_score']:.3f}**
                ---
            """)