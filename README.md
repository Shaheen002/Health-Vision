# HealthVision: An Advanced Healthcare Chatbot

## Overview
HealthVision is an innovative healthcare chatbot designed to provide accessible medical guidance while minimizing the risks associated with self-diagnosis and incorrect treatment. Our platform integrates advanced deep learning techniques to deliver accurate disease classification, personalized medical advice, and a range of supplementary healthcare features.

## Features

### 1. Symptom-Based Disease Classification
- **Approach**: Initially explored deep learning techniques such as DistilBERT, RNN with LSTM, SVM, and Naive Bayes.
- **Solution**: Leveraging these traditional machine learning and deep learning techniques, we developed a robust system for analyzing symptoms and providing accurate disease classifications.

### 2. Image-Based Diagnosis
- **Technology**: Built using a convolutional neural network (CNN) model in TensorFlow and Keras.
- **Architecture**:
  - Convolutional layers and max pooling layers extract features from medical scan images.
  - Fully connected neural network performs disease classification based on the extracted features.
- **Preprocessing**: Implemented a classifier to validate input images, reducing erroneous or irrelevant inferences.

### 3. Medical PDF Summarizer
- Provides concise and comprehensive summaries of medical reports.
- Helps users without medical backgrounds understand complex medical terminology.

### 4. Drug Suggestion Mechanism
- Suggests appropriate drugs based on symptoms and diagnosed diseases.
- **Key Features**:
  - Specifies dosage recommendations for different age groups.
  - Uses RoBERTaTweet for sentiment analysis on user feedback regarding suggested drugs.
  - Continuously updates the database based on user sentiment ratings and intensity of reviews.
- **Feedback Mechanism**:
  - Asks for feedback after 3 days of drug recommendation.
  - Analyzes feedback to refine drug suggestions and improve accuracy.

### 5. Data Security
- Employs Fernet encryption to ensure secure handling of sensitive user data.

## Challenges and Solutions

### 1. Limited Medical Datasets
- Addressed by utilizing data augmentation techniques and leveraging pre-trained models for better performance with fewer resources.

### 2. Computational Constraints
- Adopted lightweight models and techniques to optimize performance on limited hardware resources.

### 3. Compatibility Issues
- Encountered package compatibility challenges in Python environments.
- Resolved by transitioning to Windows Subsystem for Linux (WSL) to ensure TensorFlow compatibility and seamless model deployment.

## Future Enhancements
- Further training on larger and more diverse datasets to improve classification accuracy.
- Expanding the chatbot's capabilities to include real-time consultations with medical professionals.
- Enhancing multilingual support to cater to a global audience.

## Getting Started

### Prerequisites
- Python 3.9 or later
- TensorFlow 2.x
- Keras
- RoBERTaTweet for sentiment analysis
- Required Python libraries: `transformers`, `torch`, `numpy`, `pandas`, `cryptography`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/HealthVision.git
   cd HealthVision
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download pre-trained models and place them in the `models/` directory.

### Usage
- Run the chatbot:
  ```bash
  python chatbot.py
  ```
- For image-based diagnosis:
  ```bash
  python image_diagnosis.py --image_path <path_to_image>
  ```
- For summarizing medical PDFs:
  ```bash
  python summarize_pdf.py --pdf_path <path_to_pdf>
  ```

## Contributing
We welcome contributions to improve HealthVision. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The developers and contributors of the open-source deep learning community.
- OpenAI and the machine learning research community for inspiration and support.

## Contact
For any questions or feedback, please contact us at [your-email@example.com](mailto:your-email@example.com).

