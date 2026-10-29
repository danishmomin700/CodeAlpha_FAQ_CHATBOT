
import re
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Download required NLTK resources
for resource in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(resource, quiet=True)


# Set the path of the FAQ dataset
FAQ_FILE = "faqs.csv"


def preprocess_text(text):
    """Clean and tokenize text using NLP."""

    text = str(text).lower()

    # Remove punctuation and other unwanted symbols
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))

    cleaned_tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    return " ".join(cleaned_tokens)


def load_faqs():
    """Load FAQ questions and answers from the CSV file."""

    data = pd.read_csv(FAQ_FILE)

    # Check that the required columns exist
    required_columns = {"question", "answer"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "faqs.csv must contain question and answer columns."
        )

    data = data.dropna(subset=["question", "answer"]).copy()

    data["question"] = data["question"].astype(str)
    data["answer"] = data["answer"].astype(str)

    # Preprocess all stored questions
    data["clean_question"] = data["question"].apply(preprocess_text)

    return data


# Load the dataset
faqs = load_faqs()


# Convert FAQ questions into TF-IDF numerical features
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(faqs["clean_question"])


def get_answer(user_question):
    """Find the most similar FAQ and return its answer."""

    cleaned_question = preprocess_text(user_question)

    if not cleaned_question.strip():
        return (
            "Please enter a question using words or a short sentence.",
            0.0
        )

    # Convert the user's question using the same vectorizer
    user_vector = vectorizer.transform([cleaned_question])

    # Compare the user question with every FAQ question
    similarities = cosine_similarity(user_vector, faq_vectors)[0]

    # Find the index of the highest similarity score
    best_index = similarities.argmax()
    best_score = float(similarities[best_index])

    # Avoid returning unrelated answers for weak matches
    threshold = 0.20

    if best_score < threshold:
        return (
            "Sorry, I couldn't find a suitable answer. "
            "Please try asking in a different way or contact support.",
            best_score
        )

    best_answer = faqs.iloc[best_index]["answer"]

    return best_answer, best_score


# Test the chatbot when this file is run directly
if __name__ == "__main__":
    print("College FAQ Chatbot")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYou: ")

        if question.lower().strip() == "exit":
            print("Chatbot: Goodbye!")
            break

        answer, score = get_answer(question)

        print("Chatbot:", answer)
        print(f"Similarity score: {score:.2f}")
