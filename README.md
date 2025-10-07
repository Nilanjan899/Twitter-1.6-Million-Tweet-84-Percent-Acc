# Tweet Sentiment Analysis 🐦💬

**Author:** Nilanjan Saha

**Dataset:** [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140)

**Goal:** Develop a high-performance sentiment analysis system that achieves 83%+ accuracy in understanding social media emotions through advanced deep learning techniques.
A comprehensive sentiment analysis project implementing and comparing multiple machine learning approaches for classifying tweet sentiments as positive or negative.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🎯 Overview

This project explores three different approaches to sentiment analysis on tweet data:

1. **Traditional Machine Learning**: Logistic Regression and Naive Bayes with TF-IDF vectorization
2. **Deep Learning with Custom Embeddings**: LSTM network using GloVe embeddings
3. **Deep Learning with Learned Embeddings**: Bidirectional LSTM with trainable embeddings

The project demonstrates the evolution from classical NLP techniques to modern deep learning architectures, providing insights into their relative performance and trade-offs.

## 📊 Dataset

- **Training Data**: 1,523,975 tweets
- **Test Data**: 359 tweets
- **Classes**: Binary (0 = Negative, 1 = Positive)
- **Preprocessing**: Text cleaning, stemming/lemmatization, stopword removal

### Data Sources
- Tweet sentiment dataset (CSV format)
- GloVe pre-trained word embeddings (50d and 100d versions)

## 🤖 Models Implemented

### 1. Logistic Regression with TF-IDF
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Preprocessing**: Porter Stemming, stopword removal
- **Test Accuracy**: **80.50%**

### 2. Multinomial Naive Bayes with TF-IDF
- **Vectorization**: TF-IDF
- **Preprocessing**: Porter Stemming, stopword removal
- **Test Accuracy**: **79.67%**

### 3. LSTM with GloVe Embeddings
- **Architecture**: 3-layer LSTM (16 units each)
- **Embeddings**: GloVe 100d (pre-trained)
- **Regularization**: L2 regularization, Dropout (0.3)
- **Training**: 15% stratified sample, Early stopping
- **Test Accuracy**: **81.34%**

### 4. Bidirectional LSTM with Learned Embeddings
- **Architecture**: Bidirectional LSTM (128 units) + Dense layers
- **Embeddings**: GloVe 50d initialized, trainable
- **Vocabulary**: 50,000 most frequent words
- **Regularization**: L2 regularization, Dropout (0.5)
- **Training**: Full dataset (1.5M samples)
- **Test Accuracy**: **83.00%** ⭐

## 📁 Project Structure

```
sentiment-analysis/
│
├── data/
│   ├── train_data.csv          # Training dataset
│   ├── test_data.csv           # Test dataset
│   └── glove.6B.*.txt          # GloVe embeddings
│
├── models/
│   ├── logistic_regression/    # TF-IDF + LogReg
│   ├── naive_bayes/            # TF-IDF + NB
│   ├── lstm_glove/             # LSTM with GloVe
│   └── bilstm/                 # BiLSTM final model
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── outputs/
│   ├── best_model.weights.h5   # LSTM model weights
│   ├── bilstm_glove_fixed.h5   # BiLSTM model
│   └── tokenizer.json          # Trained tokenizer
│
└── README.md
```

## 🛠️ Requirements

```python
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
tensorflow>=2.8.0
keras>=2.8.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 💻 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

4. **Download GloVe embeddings**
- Download from [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/)
- Place `glove.6B.50d.txt` and `glove.6B.100d.txt` in the data directory

## 🚀 Usage

### Training Traditional ML Models

```python
# Load and preprocess data
df_train = pd.read_csv('train_data.csv')
df_train["Stemmed"] = df_train["sentence"].apply(stemming)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

# Train Logistic Regression
model_log = LogisticRegression()
model_log.fit(X_train, y_train)
```

### Training LSTM Model

```python
# Prepare sequences
tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=55)

# Build and train model
model = Sequential([
    Embedding(vocab_size, embed_dim, weights=[embedding_matrix]),
    Bidirectional(LSTM(128)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=6)
```

### Making Predictions

```python
# Load saved model
model = tf.keras.models.load_model('bilstm_glove_fixed.h5')

# Predict sentiment
new_text = ["This movie was absolutely fantastic!"]
sequences = tokenizer.texts_to_sequences(new_text)
padded = pad_sequences(sequences, maxlen=55)
prediction = model.predict(padded)
sentiment = "Positive" if prediction > 0.5 else "Negative"
```

## 📈 Results

### Model Comparison

| Model | Train Acc | Val Acc | Test Acc | Parameters |
|-------|-----------|---------|----------|------------|
| Logistic Regression | 77.98% | 76.86% | **80.50%** | ~77K |
| Naive Bayes | 78.23% | 75.05% | **79.67%** | ~77K |
| LSTM (GloVe 100d) | 80.30% | 78.91% | **81.34%** | ~2M |
| BiLSTM (GloVe 50d) | 85.00% | 83.00% | **83.00%** | ~3M |

### Detailed Metrics (BiLSTM - Best Model)

**Test Set Performance:**
- **Accuracy**: 83.00%
- **Precision**: 0.79
- **Recall**: 0.90
- **F1-Score**: 0.84

**Confusion Matrix:**
```
                Predicted
              Neg    Pos
Actual Neg    134     43
       Pos     18    164
```

## 📊 Visualizations

The project includes comprehensive visualizations:

1. **Training/Validation Curves**: Loss and accuracy over epochs
2. **Confusion Matrices**: For all models across train/val/test sets
3. **Metric Comparisons**: Bar charts comparing precision, recall, and F1-scores
4. **Model Performance Dashboard**: Comprehensive comparison infographics

## 🔍 Key Findings

1. **Deep Learning Superiority**: BiLSTM outperformed traditional ML by ~3% on test data
2. **Embedding Quality Matters**: Pre-trained GloVe embeddings provided strong initialization
3. **Regularization Importance**: L2 regularization and dropout prevented overfitting
4. **Bidirectionality Helps**: BiLSTM captured context better than unidirectional LSTM
5. **Data Scale Impact**: Full dataset training (1.5M samples) significantly improved performance

## 🔮 Future Improvements

- [ ] Implement Transformer-based models (BERT, RoBERTa)
- [ ] Add attention mechanisms to LSTM
- [ ] Experiment with different embedding dimensions
- [ ] Implement ensemble methods
- [ ] Add real-time prediction API
- [ ] Extend to multi-class sentiment analysis
- [ ] Deploy as web application
- [ ] Add explainability features (LIME, SHAP)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project was developed as part of a machine learning pipeline exploration. The models and techniques demonstrated here serve as a foundation for understanding sentiment analysis approaches ranging from traditional ML to modern deep learning.

⭐ If you found this project helpful, please consider giving it a star!
