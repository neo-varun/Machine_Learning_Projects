import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("Recommendation System")


@st.cache_data
def load_data():
    df = pd.read_csv("Recommendation System with User Behavior/product_data.csv")
    return df


df = load_data()

user_item_matrix = df.pivot_table(
    index="user_id", columns="item_id", values="rating"
).fillna(0)

st.subheader("User Item Matrix")
st.write(user_item_matrix)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
)

st.subheader("User Similarity Matrix")
st.write(user_similarity_df)


def recommend_items(user_id, top_n=3):
    if user_id not in user_item_matrix.index:
        return []

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]

    weighted_sum = pd.Series(0, index=user_item_matrix.columns)
    similarity_sum = 0

    for sim_user, sim_score in similar_users.items():
        weighted_sum += sim_score * user_item_matrix.loc[sim_user]
        similarity_sum += sim_score

    predicted_ratings = weighted_sum / similarity_sum

    already_rated = user_item_matrix.loc[user_id]
    recommendations = predicted_ratings[already_rated == 0]

    return recommendations.sort_values(ascending=False).head(top_n)


st.subheader("Get Recommendations")

user_id = st.selectbox("Select User ID", user_item_matrix.index)
top_n = st.slider("Number of Recommendations", 1, 5, 3)

if st.button("Recommend"):
    recs = recommend_items(user_id, top_n)

    if len(recs) == 0:
        st.write("No recommendations available")
    else:
        st.write(recs)
