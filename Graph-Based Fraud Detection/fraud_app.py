import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("Graph-Based Fraud Detection/fraud_data.csv")

G = nx.Graph()

for _, row in df.iterrows():
    G.add_edge(row["user_id"], row["transaction_id"])
    G.add_edge(row["user_id"], row["device_id"])
    G.add_edge(row["user_id"], row["ip_address"])

degree_dict = dict(G.degree())
df["degree"] = df["user_id"].map(degree_dict)

centrality = nx.degree_centrality(G)
df["centrality"] = df["user_id"].map(centrality)

st.subheader("Graph Visualization")

sample_nodes = list(G.nodes)[:50]
H = G.subgraph(sample_nodes)

pos = nx.spring_layout(H)

plt.figure(figsize=(8, 6))
nx.draw(H, pos, with_labels=True, node_size=300, font_size=8)

st.pyplot(plt)

X = df[["amount", "degree", "centrality"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

st.title("Graph-Based Fraud Detection")

st.subheader("Model Evaluation")
st.dataframe(pd.DataFrame(report).transpose())

st.subheader("Transaction Prediction")

amount = st.number_input("Transaction Amount", 0, 2000)
degree = st.number_input("Node Degree", 0, 50)
centrality = st.number_input("Centrality", 0.0, 1.0)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[amount, degree, centrality]], columns=["amount", "degree", "centrality"]
    )
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("Fraud Detected")
    else:
        st.write("Legitimate Transaction")
