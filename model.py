import pandas as pd
import numpy as np
# import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def train_and_predict_disease(data_file):
    # Load the CSV file
    data = pd.read_csv(data_file, names=['Symptoms', 'Disease'])
    data.dropna(subset=['Symptoms','Disease'], inplace=True)
    # Split the data into features (symptoms) and labels (diseases)
    X = data['Symptoms']  # Features
    y = data['Disease']  # Labels

    # Check if the dataset is large enough to be split
    if len(X) <= 1:
        raise ValueError("Dataset is too small to be split into training and testing sets.")

    # Initialize spaCy for tokenization
    nlp = spacy.load("en_core_web_sm")

    # Tokenize and preprocess text using spaCy
    def preprocess_text(text):
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop])

    # Preprocess the symptoms data
    X = X.apply(preprocess_text)

    # Split the preprocessed data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the training symptom data and transform it into numerical features
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Transform the test symptom data into numerical features using the same vectorizer
    X_test_vectorized = vectorizer.transform(X_test)

    # Initialize and train the SVM classifier on the vectorized training data
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_vectorized, y_train)

    # Initialize and train the Naive Bayes classifier
    nb_model = make_pipeline(CountVectorizer(), MultinomialNB())
    nb_model.fit(X_train, y_train)

    # Make predictions on the test data using both classifiers
    y_pred_svm = svm_classifier.predict(X_test_vectorized)
    y_pred_nb = nb_model.predict(X_test)

    # Combine predictions using voting
    def predict_ensemble(input_text):
        input_text = preprocess_text(input_text)
        input_vectorized = vectorizer.transform([input_text])
        svm_prediction = svm_classifier.predict(input_vectorized)[0]
        nb_prediction = nb_model.predict([input_text])[0]
        # Voting mechanism
        if svm_prediction == nb_prediction:
            return svm_prediction
        else:
            # If predictions are different, return the one with higher confidence
            svm_confidence = svm_classifier.decision_function(input_vectorized)
            nb_confidence = nb_model.predict_proba([input_text])
            if np.max(svm_confidence) >= np.max(nb_confidence):
                return svm_prediction
            else:
                return nb_prediction

    # Calculate the accuracy of the models
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print("SVM Classifier Accuracy:", accuracy_svm)
    print("Naive Bayes Classifier Accuracy:", accuracy_nb)

    return predict_ensemble

