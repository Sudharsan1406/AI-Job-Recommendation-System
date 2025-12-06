# AI-Job-Recommendation-System


🧠 AI Job Recommendation System

An intelligent job-matching application powered by Sentence-BERT embeddings, cosine similarity, and a custom ranking engine.
The app analyzes user-entered skills and recommends the most relevant job roles based on semantic similarity, experience fit, and salary normalization.

Built using Streamlit, Sentence Transformers, Scikit-Learn, and Pandas.



🚀 Features


✔ Skill → Embedding Conversion

User inputs (skills, tech stack, domain interest) are converted into sentence embeddings using:
all-MiniLM-L6-v2 (Sentence-BERT)



✔ Job Embedding Search

Each job role from the dataset already has a precomputed embedding.
The app compares user embeddings with job embeddings using cosine similarity.



✔ Custom Ranking Score

Final score =
70% semantic similarity
20% experience-level compatibility
10% normalized salary score

This approach gives more realistic and balanced job recommendations.



✔ Top-10 Recommendations

Clean UI showing:
Job title
Location
Experience level
Salary
Match score



✔ Streamlit App (Interactive & Lightweight)

Instant recommendations with a simple text input.



🧠 Tech Stack

Component	Technology
Frontend	Streamlit
NLP Model	Sentence-BERT (all-MiniLM-L6-v2)
Vector Search	Cosine Similarity (Sklearn)
Data Processing	Pandas, NumPy
Scoring Logic	Custom ranking formula
Files Used	processed_jobs.pkl (preprocessed dataset)



📂 Project Structure
.

├── app.py                  # Streamlit frontend + recommendation logic

├── processed_jobs.pkl      # Pickle file containing job data + embeddings

├── README.md               # Documentation

└── requirements.txt        # Dependencies



📝 Requirements File Example

streamlit

pandas

numpy

scikit-learn

sentence-transformers==2.6.0

transformers==4.36.2

huggingface-hub==0.20.3




📌 Data File: processed_jobs.pkl



Your .pkl contains:

job title
location
experience level
salary
normalized fields (exp_norm, salary_norm)
embedding vector (as NumPy array)




👨‍💻 Author

Sudharsan M S

AI/ML Developer

NLP | Machine Learning | Streamlit | Embedding Models
