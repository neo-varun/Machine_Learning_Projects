import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")

data = pd.read_csv("FAQ Chatbot System/faq_dataset.csv")


def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    tokens = word_tokenize(text)

    text = " ".join(tokens)

    return text


data["processed_question"] = data["Question"].apply(preprocess_text)

model = SentenceTransformer("all-MiniLM-L6-v2")

question_embeddings = model.encode(data["processed_question"].tolist())


def chatbot_response(user_question):

    processed_input = preprocess_text(user_question)

    user_embedding = model.encode([processed_input])

    similarity_scores = cosine_similarity(user_embedding, question_embeddings)

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.3:
        return "Sorry, I could not understand your question."

    return data["Answer"][best_match_index]


print("FAQ Chatbot (type 'exit' to quit)")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = chatbot_response(user_input)

    print("Chatbot:", response)
