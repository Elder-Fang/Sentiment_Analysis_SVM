# Sentiment Analysis Project Using SVM, Light Stemming, and Chi-Square Feature Selection

## Overview
This project focuses on sentiment analysis using machine learning techniques. The goal is to classify tweets into one of four sentiment categories: **Negative**, **Neutral**, **Positive**, or **Irrelevant**. The project uses the following key techniques:
1. **Light Stemming**: To normalize text by reducing words to their root forms.
2. **Chi-Square Feature Selection**: To select the most relevant features for the model.
3. **Support Vector Machine (SVM)**: A powerful classifier for text data.

The project is implemented in Python and leverages libraries such as `scikit-learn`, `nltk`, and `pandas`.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Preprocessing](#preprocessing)
5. [Feature Extraction](#feature-extraction)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [Prediction](#prediction)
9. [Saved Files](#saved-files)
10. [How to Use](#how-to-use)
11. [Future Improvements](#future-improvements)

---

## Project Structure
sentiment-analysis/
│
├── data/
│ ├── twitter_training.csv # Training dataset
│ └── twitter_validation.csv # Validation dataset
│
├── models/
│ ├── svm_model.pkl # Trained SVM model
│ ├── vectorizer.pkl # TF-IDF vectorizer
│ └── chi2_selector.pkl # Chi-Square feature selector
│
├── scripts/
│ ├── train_model.py # Script to train and save the model
│ └── predict_sentiment.py # Script to predict sentiment for new text
│
└── README.md # Project documentation



---

## Installation
To run this project, you need the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `re`
- `pickle`

You can install the required libraries using `pip`:
```bash
pip install pandas numpy scikit-learn nltk

Download NLTK resources (if not already downloaded):

import nltk
nltk.download('stopwords')

Dataset
The dataset consists of two CSV files:

Training Data: twitter_training.csv

Validation Data: twitter_validation.csv

Each file contains the following columns:

id: Unique identifier for each tweet.

category: Category of the tweet (not used in this project).

sentiment: Sentiment label (Negative, Neutral, Positive, Irrelevant).

text: The tweet text.

The sentiment labels are mapped to numerical values as follows:

Negative: 0

Neutral: 2

Positive: 4

Irrelevant: 1

Preprocessing
The text data is preprocessed using the following steps:

Remove Numbers: All digits are removed from the text.

Tokenization: The text is split into individual words.

Lowercasing: All words are converted to lowercase.

Stopword Removal: Common stopwords (e.g., "the", "and") are removed.

Stemming: Words are reduced to their root forms using the Porter Stemmer.
```
Example:

```bash
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = tokenizer.tokenize(text.lower())  # Tokenize and lowercase
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]  # Stemming
    return ' '.join(stemmed_tokens)
```

## Feature Extraction

- TF-IDF Vectorization:

- Converts the preprocessed text into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).

- Parameters: max_features=5000, ngram_range=(1, 2).

- Chi-Square Feature Selection:

- Selects the top k=3000 features based on the Chi-Square statistical test.

- Reduces dimensionality and improves model performance.

## Model Training
- Algorithm: Support Vector Machine (SVM) with a linear kernel.

- Training: The model is trained on the preprocessed and feature-selected data.

- Saving the Model: The trained model, vectorizer, and chi-square selector are saved as .pkl files for future use.

### Example:
```bash
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_chi2, train_df['sentiment'])

# Save the model and associated files
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)
```
## Evaluation
The model is evaluated on the validation dataset using:

- **Classification Report: Precision, recall, F1-score, and support for each class.**

- **Accuracy: Overall accuracy of the model.**

### Example:
```bash
y_pred = svm_model.predict(X_valid_chi2)
print("Classification Report:")
print(classification_report(valid_df['sentiment'], y_pred))
print("Accuracy:", accuracy_score(valid_df['sentiment'], y_pred))
```
## Prediction
You can use the saved model to predict sentiment for new text. Example:
```bash
def predict_sentiment(text):
    # Load the saved model, vectorizer, and chi-square selector
    with open('svm_model.pkl', 'rb') as model_file:
        svm_model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    with open('chi2_selector.pkl', 'rb') as chi2_file:
        chi2_selector = pickle.load(chi2_file)

    # Preprocess and predict
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])
    features = chi2_selector.transform(features)
    prediction = svm_model.predict(features)
    sentiment_mapping_reverse = {0: "Negative", 1: "Irrelevant", 2: "Neutral", 4: "Positive"}
    return sentiment_mapping_reverse[prediction[0]]

# Example usage
sample_tweet = "I love this product! Highly recommended."
print("Sample Tweet Prediction:", predict_sentiment(sample_tweet))
```

## Saved Files
The following files are saved after training:

svm_model.pkl: Trained SVM model.

vectorizer.pkl: TF-IDF vectorizer.

chi2_selector.pkl: Chi-Square feature selector.

## How to Use
### Train the Model:
Run the train_model.py script to preprocess the data, train the model, and save the files.

## Predict Sentiment:
Use the predict_sentiment.py script to load the saved model and predict sentiment for new text.

##Future Improvements

### Hyperparameter Tuning:
Experiment with different SVM kernels (rbf, poly) and regularization parameters (C).

### Advanced Text Normalization:
Use lemmatization (e.g., WordNetLemmatizer) instead of stemming for more accurate word normalization.

### Cross-Validation:
Use cross-validation to evaluate the model's performance more robustly.

### Class Imbalance Handling:
Address class imbalance using techniques like SMOTE or class weighting in SVM.

## Deployment:

Deploy the model as a web application or API using frameworks like Flask or FastAPI.