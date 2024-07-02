import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
import numpy as np
from concurrent.futures import ProcessPoolExecutor

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_text(text):
    doc = nlp(text.lower())
    cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(cleaned_tokens)

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def process_batch(batch):
    return [preprocess_text(text) for text in batch]

def process_all_data_in_parallel(data, batch_size=100):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_batch, batch_generator(data, batch_size)))
    # Flatten list of lists into a single list
    return [item for sublist in results for item in sublist]

def main():
    with open("train_light.json", "r") as file:
        data = json.load(file)

    questions = []
    answers = []

    for item in data:
        for annotation in item["annotations"]:
            if annotation["type"] == "multipleQAs":
                for qaPair in annotation["qaPairs"]:
                    questions.append(qaPair["question"])
                    answers.append(qaPair["answer"][0])  # Assuming one answer per question, taking the first
            elif annotation["type"] == "singleAnswer":
                questions.append(item["question"])  # Base question for singleAnswer
                answers.append(annotation["answer"][0])

    print("Starting preprocessing...")
    processed_questions = process_all_data_in_parallel(questions, batch_size=100)
    print("Preprocessing completed.")

    model = make_pipeline(TfidfVectorizer(), SGDClassifier(loss="hinge"))
    print("Starting model training...")
    model.fit(processed_questions, np.array(answers))
    print("Model training completed.")

    # Example usage
    while True:
        if user_input == "exit".lower():
            break
        else:
            user_input = input("Ask Bot a Question: ")
            processed_input = preprocess_text(user_input)
            response = model.predict([processed_input])[0]
            print(response)

if __name__ == '__main__':
    main()
