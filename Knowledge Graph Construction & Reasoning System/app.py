import streamlit as st
import spacy
import networkx as nx
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Knowledge Graph System", layout="wide")

st.markdown(
    """
<style>
.main {
    padding: 2rem;
}
h1, h2, h3 {
    color: #2c3e50;
}
.block-container {
    padding-top: 2rem;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
}
.stTextInput>div>div>input {
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Knowledge Graph Construction and Reasoning System")

try:
    with open("Knowledge Graph Construction & Reasoning System/data.txt", "r") as f:
        text = f.read()
except FileNotFoundError:
    st.error("data.txt file not found")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Data")
    st.text_area("Dataset", text, height=400, label_visibility="collapsed")

doc = nlp(text)

entities = [(ent.text, ent.label_) for ent in doc.ents]


def extract_relations(doc):
    relations = []
    for sent in doc.sents:
        words = [token.text.lower() for token in sent]
        ents = [ent.text for ent in sent.ents]

        if "works" in words and "at" in words and len(ents) >= 2:
            relations.append((ents[0], "works_at", ents[1]))

        if "lives" in words and len(ents) >= 2:
            relations.append((ents[0], "lives_in", ents[1]))

        if "knows" in words and len(ents) >= 2:
            relations.append((ents[0], "knows", ents[1]))

        if "located" in words and len(ents) >= 2:
            relations.append((ents[0], "located_in", ents[1]))

    return relations


relations = extract_relations(doc)

G = nx.Graph()
for subj, rel, obj in relations:
    G.add_node(subj)
    G.add_node(obj)
    G.add_edge(subj, obj, label=rel)

with col2:
    st.subheader("Knowledge Graph")
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, k=0.3, seed=42)

        fig, ax = plt.subplots(figsize=(4, 3))

        nx.draw(G, pos, with_labels=True, node_size=100, font_size=2, width=0.4, ax=ax)

        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=2, ax=ax
        )

        ax.set_axis_off()
        st.pyplot(fig)

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("Relationship Query")
    node1 = st.text_input("Entity A")
    node2 = st.text_input("Entity B")

    if st.button("Find Relationship"):
        if node1 in G and node2 in G:
            try:
                path = nx.shortest_path(G, node1, node2)
                st.write("Path:", " → ".join(path))

                explanation = []
                for i in range(len(path) - 1):
                    rel = G[path[i]][path[i + 1]]["label"]
                    explanation.append(f"{path[i]} --({rel})--> {path[i+1]}")

                st.text("\n".join(explanation))

            except nx.NetworkXNoPath:
                st.error("No relationship found")
        else:
            st.warning("Entity not found")

with c2:
    st.subheader("Connections")
    entity = st.text_input("Entity")

    if st.button("Get Connections"):
        if entity in G:
            neighbors = list(G.neighbors(entity))
            st.write("Direct:", neighbors)

            indirect = set()
            for n in neighbors:
                indirect.update(G.neighbors(n))

            indirect.discard(entity)
            st.write("Indirect:", list(indirect))
        else:
            st.warning("Entity not found")

st.divider()

st.subheader("Entities and Relationships")

c3, c4 = st.columns(2)

with c3:
    st.write("Entities")
    st.dataframe(entities, width="stretch")

with c4:
    st.write("Relationships")
    st.dataframe(relations, width="stretch")
