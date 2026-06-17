import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD

print("1. DATA PREPROCESSING - Pandas")
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', usecols=[0,1], names=['movie_id', 'title'])
print(f"Data loaded: {ratings.shape[0]} ratings, {movies.shape[0]} movies")

print("\n2. CONTENT-BASED - TF-IDF + Cosine Similarity")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['title'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_content_recs(title, top_n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

print(f"Content-based for 'Star Wars (1977)': {get_content_recs('Star Wars (1977)')}")

print("\n3. COLLABORATIVE - SVD Matrix Factorization")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD(n_factors=50, random_state=42)
algo.fit(trainset)

def get_collab_recs(user_id, top_n=5):
    movie_ids = ratings['movie_id'].unique()
    predictions = [algo.predict(user_id, iid) for iid in movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_movies = predictions[:top_n]
    return [movies[movies['movie_id']==int(p.iid)]['title'].values[0] for p in top_movies]

print(f"Collaborative for User 50: {get_collab_recs(50)}")

print("\n4. HYBRID RECOMMENDER")
def hybrid_recommend(user_id, movie_title, top_n=5):
    content_recs = get_content_recs(movie_title, top_n*2)
    content_movies = movies[movies['title'].isin(content_recs)]
    collab_scores = [(row['movie_id'], algo.predict(user_id, row['movie_id']).est)
                     for _, row in content_movies.iterrows()]
    collab_scores.sort(key=lambda x: x[1], reverse=True)
    top_ids = [str(mid) for mid, _ in collab_scores[:top_n]]
    return movies[movies['movie_id'].astype(str).isin(top_ids)]['title'].tolist()

print(f"HYBRID for User 50 + Star Wars: {hybrid_recommend(50, 'Star Wars (1977)')}")
print("\nProject complete: Hybrid system built with Pandas, TF-IDF, Cosine Similarity, SVD")