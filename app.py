import os
import keras
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
from flask import Flask, request, jsonify, render_template,redirect,url_for,session
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from googletrans import Translator
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import langid
import io
from keras.preprocessing import image as im
from keras.preprocessing.image import load_img
import sqlite3
import pandas as pd
from fuzzywuzzy import process

# from flask_mail import Mail, Message
import smtplib
from email.mime.text import MIMEText
from model import train_and_predict_disease
from TRans_Rnn import trans_Rnn_predict



server = smtplib.SMTP("smtp.gmail.com",587)
server.starttls()
server.login("healthvision634@gmail.com","bwcfjufmmescxijw")
curr_email = ""

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'Secret_Key'
# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config["MAIL_PORT"] = 465
# app.config["MAIL_USERNAME"] = 'healthvision634@gmail.com'
# app.config["MAIL_PASSWORD"] = 'cxhrjnbnxnngtmbs'
# app.config["MAIL_USE_TLS"] = False
# app.config["MAIL_USE_SSL"] = True
# app.config['MAIL_DEBUG']=True

# mail = Mail(app)


sqcon=sqlite3.connect('user_details',check_same_thread=False)
cursor=sqcon.cursor()
df_recommendations = pd.read_csv("shaheen/shaheen/updatedRecNew1.csv")



dataNew = pd.read_csv("shaheen/shaheen/shah-d.csv", header=None, names=['Symptoms', 'Label'])

X = dataNew['Symptoms']  # Features
y = dataNew['Label']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
@app.route("/forgot_password",methods=["GET"])
def forgot_password():
    return render_template('forget_password.html')

@app.route('/logout',methods=['GET'])
def logout():
    return render_template('login.html')

vectorizer = TfidfVectorizer()

# Fit the vectorizer on the training symptom data and tr
X_train_vectorized = vectorizer.fit_transform(X_train)

# Transform the test symptom data into numerical 
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train the SVM classifi
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_vectorized, y_train)

#Define a function to ma
def predict_disease_svm(input_text):
    input_text = input_text.lower()
    # Transform the input text into numerical features
    input_vectorized = vectorizer.transform([input_text])
    # Make predictions using the SVM model
    predicted_label = svm_classifier.predict(input_vectorized)
    return predicted_label[0]

# Load trained model
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

# Define a function to make predictions using the Transformer model
def predict_disease_transformer(input_text):
    input_text = input_text.lower()
    # Tokenize the input text
    input_tokens = tokenizer(input_text, padding=True, truncation=True, return_tensors='tf')
    input_ids = input_tokens['input_ids']
    attention_mask = input_tokens['attention_mask']
    # Make predictions using the Transformer model
    predictions = model.predict([input_ids, attention_mask])
    # Decode the predicted labels
    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_disease = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_disease

# Define a function to make predictions using both models and ensemble them
def ensemble_predict(input_text):
    # Make predictions using the SVM model
    predicted_disease_svm = predict_disease_svm(input_text)
    # Make predictions using the Transformer model
    predicted_disease_transformer = predict_disease_transformer(input_text)
    # Combine predictions using a voting mechanism
    if predicted_disease_svm == predicted_disease_transformer:
        final_prediction = predicted_disease_svm
    else:
        # If predictions are different, choose one of them or perform any other decision logic
        final_prediction = predicted_disease_svm  # Adjust this line based on your decision logic
    return final_prediction

def minor_disease(input_text):
    a=input_text.lower().split()
    print(a)
    for i in range(len(a)):
        if a[i][-1]==',' or a[i][-1]=='.' or a[i][-1]=='?':
            a[i]=a[i][:-1]

    l=['fever','cold','flu','headache','cough']
    g = ['urination','movement' ,'urine', 'thirst','inflammation','popping','grinding','tender','tenderness','persistently','darkening','flexibility','mobility','stiff','stiffness','noticing','skin','hunger', 'fatigue', 'vision','joints' 'numbness', 'infections', 'wound', 'joint', 'wheezing', 'bone', 'spur', 'nausea', 'vomiting', 'diarrhea', 'abdominal', 'fever', 'headache', 'muscle', 'loss', 'dehydration', 'shortness', 'chest', 'chills', 'sputum', 'joint pain', 'swelling', 'tenderness', 'limited range of motion', 'grating sensation', 'bone spurs', 'shortness of breath', 'chest tightness', 'shortness of breath','fits','yellow','chest tightness', 'wheezing', 'bone spur', 'nausea', 'vomiting', 'diarrhea', 'abdominal', 'fever', 'headache', 'muscle', 'loss', 'dehydration', 'shortness', 'chest', 'chills', 'sputum', 'joint pain', 'swelling', 'tenderness','experiencing','gastroenteritis','aches','limited range of motion', 'grating sensation', 'bone spurs', 'shortness of breath', 'chest tightness', 'shortness of breath', 'chest tightness', 'wheezing', 'bone spur','asthma','diabetes','hypertension','panic','difficulty','worsen','discomfort','exposed','odors','odor','perfume','perfumes','allergy','triggers','cramps', 'nausea', 'vomiting', 'diarrhea', 'abdominal', 'fever', 'headache', 'muscle', 'loss', 'dehydration', 'shortness', 'chest', 'chills', 'sputum', 'joint pain', 'swelling', 'tenderness', 'limited range of motion', 'grating sensation', 'bone spurs', 'shortness of breath', 'chest tightness', 'shortness of breath', 'chest tightness', 'wheezing', 'bone spur', 'nausea', 'vomiting', 'diarrhea', 'abdominal', 'fever', 'headache', 'muscle', 'loss', 'dehydration', 'shortness', 'chest', 'chills', 'sputum', 'joint pain', 'swelling', 'tenderness', 'limited range of motion', 'grating sensation', 'bone spurs', 'shortness of breath', 'chest tightness', 'shortness of breath', 'chest tightness', 'wheezing', 'bone spur', 'nausea', 'vomiting', 'diarrhea', 'abdominal', 'fever', 'headache', 'muscle', 'loss', 'dehydration', 'shortness', 'chest', 'chills', 'sputum', 'joint pain', 'swelling', 'tenderness', 'limited range of motion', 'grating sensation', 'bone spurs', 'shortness of breath', 'chest tightness', 'shortness of breath', 'chest tightness', 'wheezing', 'bone spur', 'nausea', 'vomiting', 'muscle aches', 'confusion', 'chest pain', 'vomit','middle','night','sweats','weight','discharge','water','bloody','pain','greenish','yellowish','white','red','black','brown','reddish','persistent','severe','consistently','legs','dizzy','irritable','angry','depressed','anxious','nervous','restless','sleepy','fatigued','weak','tired','drowsy','mouth','faint','pains','muscles','unwind','choking','lump','throat','hoarseness','voice','sore','neck','swollen','lymph','nodes','swelling','lump','neck','nose','bleeding','nose','nasal','congestion','runny','nose','sneezing','throat','pain','throat','itching','throat','hoarseness','voice','cough','sputum','cough','wheezing','chest','tightness','chest','pain','chest','palpitations','shortness','breath','cough','sputum','cough','wheezing','chest','tightness','chest','pain','chest','palpitations','shortness','breath','cough','sputum','cough','wheezing','chest','tightness','chest','pain','chest','palpitations','shortness','breath','cough','sputum','cough','wheezing','chest','tightness','chest','pain','chest','palpitations','shortness','breath','cough','sputum','cough','wheezing','chest','tightness','chest','pain','chest','palpitations','shortness','breath','cough','sputum','cough','wheezing','palpitations','bloating','mood','swings','hiccups','nosebleeds','flashes','eyes','difficulty','fatigued','feet','hand','hoarness','anxiety','depression','irritability','restlessness','sleepiness','fatigue','weakness','tiredness','drowsiness','mouth','fainting',]
    c=0
    f=0
    for i in a:
        if i in l:
            c+=1
        if i in g:
            f+=1
    return [c,f,len(a)]    
# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

label_encoder = LabelEncoder()
# Fit the LabelEncoder on the training labels
label_encoder.fit(y_train)
translator = Translator()

# Sample data
# target_labels = ['HyperTension', 'Diabetes', 'Asthma', 'Gastroenteritis', 'osteoarthritis']

# Fit label encoder on target labels
# label_encoder.fit(target_labels)
label_encoder.fit(y_train)

num_classes = 5  # Update with the number of classes
model = DiseaseClassifier(num_classes)
# model.load_weights('F:\karthik J\karthik\karthik\model')
model.load_weights('shaheen\model_new') # Load your trained model weights

# Function to preprocess input text
def preprocess_text(text):
    text = text.lower()
    return text

# Function to translate text to English
def translate_to_english(text):
    translation = translator.translate(text, src='auto', dest='en')
    return translation.text
    
# Function to translate text to the original language
def translate_to_original_language(text, original_lang):
    translation = translator.translate(text, src='en', dest=original_lang)
    return translation.text


# Home route
@app.route('/')
def home():
    return render_template('login.html')

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text_original = data['symptoms']
    print("Input Text (Original):", input_text_original)

    # Check if the input text is in English
    if input_text_original.isascii():
        # Input text is in English
        input_text_english = input_text_original
    else:
        # Input text is in non-English, translate to English
        input_text_english = translate_to_english(input_text_original)
        print("Input Text (English):", input_text_english)

    # Tokenize the input text
    input_tokens = tokenizer(input_text_english, padding=True, truncation=True, return_tensors='tf')
    input_ids = input_tokens['input_ids']
    attention_mask = input_tokens['attention_mask']

    # Make predictions
    # predictions = model.predict([input_ids, attention_mask])
    final_prediction = ensemble_predict(input_text_english)
    minor_prediction = minor_disease(input_text_english)
    x,y,z=minor_prediction[0],minor_prediction[1],minor_prediction[2]
    if y>0:
        if (x==1 or x==2) and z<=5:
            final_prediction='Minor Disease'
    elif y==0:
        final_prediction='UnabletoPredict'
    
    
        
        # Decode the predicted labels
    # predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
    # predicted_disease = label_encoder.inverse_transform([predicted_class_index])[0]
    # If the input was in non-English, translate the prediction back to the original language
    if not input_text_original.isascii():
        original_lang = translator.detect(input_text_original).lang
        predicted_disease_1 = translate_to_original_language(final_prediction, original_lang)
        print("Predicted Disease (Original Language):", predicted_disease_1,final_prediction)
        return [predicted_disease_1, final_prediction]
    print(final_prediction)
    return [final_prediction]



@app.route('/recommendation', methods=['POST'])
def recommend():
    data = request.get_json()
    print(data)
    ss = data['symptom'].lower()
    predicted_disease = data['predicted_disease'].lower()
    print(ss)
    language = detect_language(ss)
    return get_recommendation(predicted_disease.capitalize(), ss.split(),language)


@app.route('/predict_image', methods=['POST'])
def return_class():
    print(1)
    if 'image' in request.files:
        # Read the uploaded image file
        input_image = request.files['image']
        print(image)
        extension=input_image.filename.split('.')[-1]
        filename=input_image.filename.split('.')[0]
        print(extension,filename)
        input_image.save('shaheen/static/'+input_image.filename)
        # print(return_class_name(output))
        # print(image)s
        # Open the image using PIL (Python Imaging Library)
        # imgg = Image.open(io.BytesIO(input_image.read()))
        
        def return_class(img_path):
            print(img_path)
            image_model=load_model('shaheen\shaheen\main_model_4.h5')
            img = im.load_img(img_path, target_size=(224, 224))
            img_array = im.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            predictions = image_model.predict(img_array)
            return predictions
        # Preprocess the image (e.g., resize, normalize, etc.)
        # Example:
        def return_class_name(predictions):
            class_name=np.argmax(predictions)
            print(class_name) 
            if class_name==0:
                return 'Bacterial Pneumonia'
            elif class_name==1:
                return 'corona virus'
            elif class_name==2:
                return "normal lungs"
            else:
                return "Chance for Tuberculosis" 
            
        output=return_class('shaheen/static/'+input_image.filename)
        final_predicted_class = return_class_name(output)
        
        # Example response: return predictions as JSON
        # return jsonify({'predicted_Class':final_predicted_class})
        return [final_predicted_class]
    else:
        # return 'No image provided', 400
        return ['No image provided']
# Let's assume your dataset has columns: 'Disease', 'Symptoms', and 'Recommendation'


# Create a dictionary to map diseases to recommendations
disease_recommendations = {}

# Iterate over the dataset to populate the dictionary
for index, row in df_recommendations.iterrows():
    disease = row['Disease']
    symptoms = row['Symptom']
    recommendation = row['Recommendation']

    if disease not in disease_recommendations:
        disease_recommendations[disease] = {'Symptoms': [], 'Recommendations': []}

    disease_recommendations[disease]['Symptoms'].append(symptoms)
    disease_recommendations[disease]['Recommendations'].append(recommendation)

def translate_to_hindi(text):
    translator = Translator()
    translated_text = ''
    chunk_size = 500  # Adjust the chunk size as needed
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        translation = translator.translate(chunk, dest='hi').text
        translated_text += translation + ' '
    translated_text = translated_text.strip()
    return translated_text

try:
    recommendation = translate_to_hindi(recommendation)
except Exception as e:
    print("Translation error:", e)
    recommendation = "Translation Error"

def get_default_message(language):
    if language == 'hi':
        return "कोई सुझाव उपलब्ध नहीं हैं"
    else:
        return "No recommendations available"

# Define a function to get recommendations for a given disease and provided symptoms
def get_recommendation(disease, provided_symptoms, language='en'):
    print(disease, provided_symptoms)
    # print(disease_recommendations[disease])
    recommendations_info = disease_recommendations.get(disease)
    # print(recommendations_info)
    if recommendations_info:
        symptoms = recommendations_info['Symptoms']
        recommendations = recommendations_info['Recommendations']

        # Join individual symptoms into a single string
        provided_symptoms_string = ' '.join(provided_symptoms)

        # Find closest matching symptom
        matched_symptom = process.extractOne(provided_symptoms_string, symptoms)[0]

        # Find index of matched symptom
        symptom_index = symptoms.index(matched_symptom)

        # Get recommendation corresponding to matched symptom
        recommendation = recommendations[symptom_index]

        # Translate the recommendation to the detected language if it's not English
        if language != 'en':
            print(1)
            recommendation = translate_to_hindi(recommendation)

        return [recommendation]  # Return as list
    else:
        return None

def detect_language(text):
    return langid.classify(text)[0]

@app.route('/questions',methods=['POST'])
def questions():
    global curr_email
    data = request.get_json()
    print("email = ",curr_email)
    cursor.execute("UPDATE user_info SET PhysicalActivity=?, fruitsAndVegetables=?, sleep=?, stress=?, tobacco=?, alcohol=?, work=?, screen=?, disease=? WHERE email='%s'" % curr_email,
    (data['PhysicalActivity'], data['fruitsAndVegetables'], data['sleep'], data['stress'], data['tobacco'], data['alcohol'], data['work'], data['screen'], data['disease']))
    sqcon.commit()
    print("success")
    return [1]
    # return render_template('login.html')

def otpSend(x,y):
    msg = MIMEText('OTP is {}'.format(x))
    msg['Subject'] = 'OTP for HeathVision'
    msg['To'] = y
    server.sendmail("healthvision634@gmail.com",y,msg.as_string())

@app.route('/otpAuthentication', methods=['POST'])
def otpAuthentication():
    if request.method == 'POST':
        # if request.form['clicked']=='Register Now':
        data = request.get_json()
        email = data.get('email')
        #session['email']=email
        print(email)
        otp = data.get('otp')
        otpSend(otp,email)
        return [1]
    #     print(data)
    return [1]

@app.route('/questionsOpen',methods=['GET'])
def questionsOpen():
    return render_template('questions.html')

@app.route('/loginOpen',methods=['GET'])
def loginOpen():
    return render_template('login.html')

@app.route('/register',methods=['POST'])
def register():
    global curr_email
    data = request.get_json()
    print(data)
    curr_email = data['emailId']
    print(curr_email)
    #insert username, password, email, phone number,gender,dob into database
    cursor.execute('''INSERT INTO user_info(userName, dob, phoneNumber, gender, age, email, password)
                      VALUES (?, ?, ?, ?, ?, ?, ?)''',
                   (data['fullName'], data['dob'], data['phoneNumber'], data['gender'],
                    data['age'], data['emailId'], data['password']))

    sqcon.commit()
    return 'Data inserted successfully'

@app.route('/index',methods=['POST'])
def index():
    if request.method == 'POST':
        if request.form['clicked']=='Register':
            return render_template('index.html')
        email=request.form['email']
        password=request.form['password']
        print(email,password)
        cursor.execute('SELECT * FROM user_info WHERE email = ? AND password = ?', (email, password))
        account = cursor.fetchone()
        if account:
            return render_template('chatbot.html')
        return render_template('login.html')
if __name__ == '__main__':
    app.run(debug=True)