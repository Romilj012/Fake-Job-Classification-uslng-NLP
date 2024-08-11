# **True & Fake Job Classification with NLP**

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview

This project aims to develop a robust system for classifying job postings as either genuine or fraudulent using Natural Language Processing (NLP) techniques and various machine learning models. By leveraging advanced text analysis and classification algorithms, we seek to help job seekers avoid scams and fraudulent offers in the job market.

### Key Objectives:
1. Analyze textual content of job postings
2. Identify linguistic patterns associated with fraudulent listings
3. Develop and compare multiple classification models
4. Provide insights into the characteristics of fake job postings

## Features

- **Data Merging**: Combines original and synthetic datasets for comprehensive analysis
- **Text Preprocessing**: Implements thorough cleaning and normalization of job posting text
- **Advanced NLP Techniques**: Utilizes TF-IDF, Count Vectorization, and Word2Vec embeddings
- **Multiple ML Models**: Implements Naive Bayes, Random Forest, SVM, RNN, and LSTM
- **Detailed EDA**: Provides in-depth analysis of various job posting attributes
- **Dimensionality Reduction**: Applies PCA for feature selection and model optimization

## Dataset

We utilize two primary datasets:

1. **Original Dataset**: `fake_job_postings.csv`
   - Source: [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
   - Contains: 17,880 job postings (866 fraudulent, 17,014 real)

2. **Synthetic Dataset**: `synthetic_dataset.csv`
   - Generated using: [Gretel](https://gretel.ai/)
   - Purpose: Augment the original dataset and balance class distribution

### Data Fields:
- `job_id`: Unique identifier for the job posting
- `title`: Job title
- `location`: Job location
- `department`: Department offering the job
- `salary_range`: Salary range for the position
- `company_profile`: Description of the company
- `description`: Detailed job description
- `requirements`: Job requirements
- `benefits`: Benefits offered
- `telecommuting`: Whether the job allows telecommuting
- `has_company_logo`: Whether the company logo is present
- `has_questions`: Whether screening questions are included
- `employment_type`: Type of employment (e.g., Full-time, Part-time)
- `required_experience`: Required experience level
- `required_education`: Required education level
- `industry`: Industry of the job
- `function`: Job function
- `fraudulent`: Target variable (0 for genuine, 1 for fraudulent)

## Data Preprocessing

### Text Cleaning
- Lowercase conversion
- Removal of non-alphabetic characters
- Stopwords removal using NLTK
- Lemmatization using WordNetLemmatizer

### Handling Missing Values
- Replaced NULL values with 'Unspecified' for categorical fields
- Used `np.nan` for numeric fields

### Standardization
- Education levels mapped to standardized categories
- Industry categories consolidated and mapped

```python
def map_education_level(education):
    level_map = {
        "Bachelor's Degree": 'Bachelor',
        'High School or equivalent': 'High School',
        # ... (other mappings)
    }
    return level_map.get(education, 'Other')

df_fake_job_pos_updated['mapped_education_level'] = df_fake_job_pos_updated['required_education'].apply(map_education_level)
```

## Exploratory Data Analysis (EDA)

### Distribution Analysis
- Genuine vs. Fraudulent job postings
- Employment types
- Job locations (country-wise)
- Salary ranges
- Industries
- Required education levels
- Required experience

### Text Analysis
- Word clouds for genuine and fraudulent postings
- Text length distribution
- Top bigrams and trigrams

### Visualizations
```python
# Example: Distribution of fraudulent jobs
sns.countplot(x='fraudulent', data=df_fake_job_pos)
plt.title('Distribution of Fraudulent Job Postings')
plt.show()
```

## Feature Engineering

### Text Vectorization
1. **TF-IDF Vectorization**
   ```python
   tfidf_vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.99, analyzer='word')
   tfidf_train_matrix = tfidf_vectorizer.fit_transform(X_train)
   ```

2. **Count Vectorization**
   ```python
   cnt_vectorizer = CountVectorizer(min_df=0.01, max_df=0.99, analyzer='word')
   cnt_train_matrix = cnt_vectorizer.fit_transform(X_train)
   ```

3. **Word2Vec Embeddings**
   ```python
   word2vec_model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=1, workers=4)
   ```

### Dimensionality Reduction
- Applied PCA (Truncated SVD) to reduce feature space
  ```python
  pca = TruncatedSVD(n_components=230)
  x_train_reduced = pca.fit_transform(X_train.toarray())
  ```

## Models

1. **Naive Bayes**
   ```python
   nb_classifier = MultinomialNB(alpha=0.6)
   ```

2. **Random Forest**
   ```python
   rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=2, max_features='sqrt')
   ```

3. **Support Vector Machine (SVM)**
   ```python
   svm_classifier = SVC(kernel='linear', C=0.5)
   ```

4. **Recurrent Neural Network (RNN)**
   ```python
   def create_rnn_model(input_shape):
       model = Sequential([
           LSTM(64, input_shape=input_shape, return_sequences=True),
           Bidirectional(LSTM(64, return_sequences=True)),
           Dropout(0.2),
           Bidirectional(LSTM(64)),
           Dropout(0.2),
           Dense(1, activation='sigmoid')
       ])
       model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
       return model
   ```

5. **Long Short-Term Memory (LSTM)**
   ```python
   def create_lstm_model(input_shape):
       model = Sequential([
           LSTM(64, input_shape=input_shape, return_sequences=True),
           Bidirectional(LSTM(64, return_sequences=True)),
           Dropout(0.2),
           Bidirectional(LSTM(64)),
           Dropout(0.2),
           Dense(1, activation='sigmoid')
       ])
       model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
       return model
   ```

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 0.95 | 0.94 | 0.95 | 0.94 |
| Random Forest | 0.97 | 0.96 | 0.97 | 0.96 |
| SVM | 0.96 | 0.95 | 0.96 | 0.95 |
| RNN | 0.98 | 0.97 | 0.98 | 0.97 |
| LSTM | 0.99 | 0.98 | 0.99 | 0.98 |

*Note: These are sample results. Please replace with your actual model performance metrics.*

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/true-fake-job-classification.git
   cd true-fake-job-classification
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset:
   - Place `fake_job_postings.csv` and `synthetic_dataset.csv` in the `data/` directory

2. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```

3. Perform Exploratory Data Analysis:
   ```bash
   python eda.py
   ```

4. Train and evaluate models:
   ```bash
   python train_models.py
   ```

5. Make predictions on new data:
   ```bash
   python predict.py path/to/new_job_postings.csv
   ```

## Acknowledgements

- [Kaggle](https://www.kaggle.com/) for providing the original dataset
- [Gretel](https://gretel.ai/) for synthetic data generation tools
- [NLTK](https://www.nltk.org/) for NLP utilities
- [Scikit-learn](https://scikit-learn.org/) for machine learning implementations
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for deep learning models
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization
