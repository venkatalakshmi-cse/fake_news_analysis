import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)


def load_and_preprocess(path):

    data = pd.read_csv(path)

    data["clean_text"] = data["text"].apply(clean_text)

    X = data["clean_text"]

    y = data["label"]

    return X, y