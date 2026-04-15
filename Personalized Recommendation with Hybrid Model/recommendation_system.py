import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("Hybrid Recommendation System")

data = pd.read_csv(
    "Personalized Recommendation with Hybrid Model/recommendation_dataset.csv"
)

user_item_matrix = data.pivot_table(
    index="user_id", columns="item_id", values="rating"
).fillna(0)

user_similarity = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(
    user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
)

data["content"] = data["category"] + " " + data["description"]
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["content"])
content_similarity = cosine_similarity(tfidf_matrix)


def recommend(user_id, top_n=5):
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:]
    cf_scores = user_item_matrix.loc[similar_users.index].mean()

    user_items = data[data["user_id"] == user_id]["item_id"]
    content_scores = pd.Series(0.0, index=data["item_id"].unique())

    for item in user_items:
        idx_list = data[data["item_id"] == item].index
        for i in idx_list:
            sim_scores = list(enumerate(content_similarity[i]))
            for j, score in sim_scores:
                item_j = data.iloc[j]["item_id"]
                content_scores[item_j] += float(score)

    hybrid_scores = cf_scores.add(content_scores, fill_value=0)
    recommendations = hybrid_scores.sort_values(ascending=False).head(top_n)

    return recommendations.index.tolist()


user_ids = sorted(data["user_id"].unique(), key=lambda x: int(x[1:]))

user_input = st.selectbox("Select User", user_ids)

if st.button("Recommend"):
    recs = recommend(user_input)
    st.write("Recommended Items:", recs)
