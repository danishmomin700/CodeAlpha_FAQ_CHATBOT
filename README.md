
# College FAQ Chatbot

## Project Overview
This project is an NLP-based chatbot that answers
college-related questions by matching user input with
a predefined FAQ dataset.

## Features
- FAQ dataset using CSV
- Text preprocessing with NLTK
- TF-IDF text vectorization
- Cosine similarity matching
- Similarity threshold for weak matches
- Interactive Streamlit chatbot interface

## Technologies Used
- Python
- NLTK
- Pandas
- Scikit-learn
- Streamlit

## Project Structure
- app.py: Chatbot user interface
- chatbot.py: NLP preprocessing and FAQ matching
- faqs.csv: Questions and answers
- requirements.txt: Required Python libraries

## Installation

Install the dependencies:

    python -m pip install -r requirements.txt

## Run the Chatbot

    python -m streamlit run app.py

## How It Works
The chatbot preprocesses FAQ questions, converts them
into TF-IDF vectors, and uses cosine similarity to find
the most similar FAQ for each user question.

## Dataset
The project includes sample college FAQs.
Replace the sample content with verified information
for your chosen topic.

## Future Improvements
- Expand the FAQ dataset
- Improve matching accuracy
- Add multilingual support
- Evaluate performance using test questions
