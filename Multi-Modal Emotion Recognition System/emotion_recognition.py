import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras.models import load_model

labels = ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]

data = pd.read_csv(
    "Multi-Modal Emotion Recognition System/emotion_data.csv", names=["text", "label"]
)
data["label"] = data["label"].str.strip()
data = data[data["label"].isin(labels)]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

text_model = LogisticRegression(max_iter=200)
text_model.fit(X, y)

label_order = list(text_model.classes_)


def build_and_train_image_model():
    classes = ["angry", "fear", "happy", "sad", "surprise", "neutral"]

    gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_data = gen.flow_from_directory(
        "Multi-Modal Emotion Recognition System/dataset/train",
        target_size=(48, 48),
        color_mode="grayscale",
        classes=classes,
        class_mode="categorical",
        batch_size=32,
        subset="training",
    )

    val_data = gen.flow_from_directory(
        "Multi-Modal Emotion Recognition System/dataset/train",
        target_size=(48, 48),
        color_mode="grayscale",
        classes=classes,
        class_mode="categorical",
        batch_size=32,
        subset="validation",
    )

    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(6, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(train_data, validation_data=val_data, epochs=5)
    model.save("Multi-Modal Emotion Recognition System/emotion_model.h5")
    return model


if os.path.exists("Multi-Modal Emotion Recognition System/emotion_model.h5"):
    image_model = load_model("Multi-Modal Emotion Recognition System/emotion_model.h5")
else:
    image_model = build_and_train_image_model()


def predict_text(text):
    vec = vectorizer.transform([text])
    pred = text_model.predict_proba(vec)
    aligned = np.zeros((1, len(labels)))

    for i, lbl in enumerate(label_order):
        if lbl in labels:
            aligned[0][labels.index(lbl)] = pred[0][i]

    return aligned


def preprocess_image(image):
    img = np.array(image.convert("L"))
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = img.reshape(1, 48, 48, 1)
    return img


def predict_image(image):
    img = preprocess_image(image)
    pred = image_model.predict(img)

    mapping = ["angry", "fear", "happy", "sad", "surprise", "neutral"]
    aligned = np.zeros((1, len(labels)))

    for i, lbl in enumerate(mapping):
        aligned[0][labels.index(lbl.capitalize())] = pred[0][i]

    return aligned


def fuse(text_pred, image_pred):
    return 0.7 * text_pred + 0.3 * image_pred


st.title("Multi-Modal Emotion Recognition")

text_input = st.text_input("Enter text")
image_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if st.button("Predict"):
    if text_input and image_file:
        image = Image.open(image_file)

        text_pred = predict_text(text_input)
        image_pred = predict_image(image)

        final_pred = fuse(text_pred, image_pred)
        emotion = labels[np.argmax(final_pred)]

        st.write("Final Emotion:", emotion)
