import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import pickle

def trans_Rnn_predict():
    # Load the CSV file
    data = pd.read_csv("data567.csv", header=None, names=['Symptoms', 'Label'])

    # Split the data into features (symptoms) and labels (diseases)
    X = data['Symptoms']  # Features
    y = data['Label']  # Labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    data['Symptoms'] = data['Symptoms'].str.lower()
    texts = data['Symptoms'].values
    labels = data['Label'].values
    tokenizer_rnn = Tokenizer()
    tokenizer_rnn.fit_on_texts(texts)
    vocab_size_rnn = len(tokenizer_rnn.word_index) + 1
    sequences = tokenizer_rnn.texts_to_sequences(texts)
    max_sequence_length_rnn = max([len(seq) for seq in sequences])
    padded_sequences_rnn = pad_sequences(sequences, maxlen=max_sequence_length_rnn, padding='post')
    label_encoder_rnn = LabelEncoder()
    label_encoder_rnn.fit(labels)
    encoded_labels_rnn = label_encoder_rnn.transform(labels)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    class DiseaseClassifier(tf.keras.Model):
        def __init__(self, num_classes):
            super(DiseaseClassifier, self).__init__()
            self.distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
            self.dropout = tf.keras.layers.Dropout(0.1)
            self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

        def call(self, inputs):
            outputs = self.distilbert(inputs)[0]
            pooled_output = outputs[:, 0, :]  # take [CLS] token
            pooled_output = self.dropout(pooled_output)
            logits = self.dense(pooled_output)
            return logits
        
    label_encoder = LabelEncoder()
    # Fit the LabelEncoder on the training labels
    label_encoder.fit(y_train)

    model = DiseaseClassifier(num_classes=5)
    model.load_weights('transformer_model')

    def predict_disease_ensemble(input_text):
        # Load the Transformer model
        model_rnn = tf.keras.models.load_model('rnn_model.h5')

        # Load the tokenizer
        with open('rnn_tokenizer.pickle', 'rb') as handle:
            tokenizer_rnn = pickle.load(handle)

        # Load the label encoder
        with open('rnn_label_encoder.pickle', 'rb') as handle:
            label_encoder_rnn = pickle.load(handle)

        # Tokenize the input text for Transformer model
        input_tokens = tokenizer(input_text, padding=True, truncation=True, return_tensors='tf')
        input_ids = input_tokens['input_ids']
        attention_mask = input_tokens['attention_mask']

        # Tokenize the input text for RNN model
        input_sequence_rnn = tokenizer_rnn.texts_to_sequences([input_text])
        padded_sequence_rnn = pad_sequences(input_sequence_rnn, maxlen=max_sequence_length_rnn, padding='post')

        # Make predictions with the Transformer model
        predictions_transformer = model.predict([input_ids, attention_mask])

        # Make predictions with the RNN model
        predictions_rnn = model_rnn.predict(padded_sequence_rnn)

        # Combine predictions (simple averaging for simplicity)
        combined_probs = (predictions_transformer[0] + predictions_rnn[0]) / 2
        predicted_class_index = np.argmax(combined_probs)
        predicted_disease = label_encoder_rnn.inverse_transform([predicted_class_index])[0]
        print(predictions_transformer)
        print(predictions_rnn)

        return predicted_disease
    
    # Add the return statement here
    return predict_disease_ensemble
