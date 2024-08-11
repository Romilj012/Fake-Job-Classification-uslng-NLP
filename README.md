# **Fake Job Classification using NLP**

## **Project Overview**
In the modern job market, the surge of online job postings has made it increasingly difficult to differentiate between legitimate opportunities and fraudulent schemes. With job scams having doubled in recent years, it has become crucial to develop tools that protect job seekers from potential financial loss and identity theft. This project addresses this issue by creating an **AI-powered classifier** that utilizes **Machine Learning** and **Natural Language Processing (NLP)** to accurately identify fraudulent job listings. Our system analyzes linguistic patterns and applies advanced algorithms to enhance the security of online job searches, ensuring a safer experience for users.

## **Dataset**
- **Origin**: University of the Aegean
- **Size**: 17,880 rows
- **Synthetic Dataset**: Gretel.ai
- **Size**: 5,000 rows

## **Initial Data Analysis**

### **Education Level**
The dataset encompasses a wide range of educational qualifications, from Bachelor's Degrees to vocational certifications. We observed that similar qualifications were often labeled differently (e.g., "Bachelor's Degree" vs. "Bachelor's or Equivalent"), highlighting the importance of data cleaning and standardization for accurate analysis.

### **Industry Level**
The job postings in our dataset span various industries, with **Information Technology and Services** representing the largest sector at 11.3% of the total entries. The top 10 industries are highlighted below, with other industries grouped under "Others" for clarity.

## **Data Cleaning and Preprocessing**

### **Data Cleaning**
- **Data Augmentation**: Integrated synthetic data to enhance model training.
- **Handling Null Values**: Replaced nulls with 'Unspecified' where appropriate.
- **Mapping Education Level**: Standardized similar educational qualifications.
- **Location Standardization**: Extracted and standardized country codes.
- **Industry Mapping**: Reorganized industries based on common traits.
- **Number to Text Transformation**: Converted binary job-related columns to descriptive labels.
- **Merging Text Columns**: Consolidated text columns for more efficient analysis.

### **Text Processing**
- **Tokenization**: Split text into individual words.
- **Lowercasing**: Converted all text to lowercase.
- **Removal of Non-Alphabetic Characters**: Retained only meaningful words.
- **Stop Word Removal**: Removed common English stop words.
- **Lemmatization**: Reduced words to their root form.
- **Part-of-Speech Tagging**: Categorized words based on grammar.
- **Detokenization**: Reconstructed preprocessed tokens into coherent text.
- **Vectorization**: Applied TF-IDF, count vectorization, Word2Vec, and BERT for numerical text representation.

## **Exploratory Data Analysis (EDA)**

### **Employment Type**
The majority of job listings are for **Full-time** positions (14,931 entries), followed by **Contract** and **Part-time** roles.

### **Locations (Country)**
Approximately **64%** of the job postings are from the United States, followed by **Great Britain** at 10.5%.

### **Industries**
Before cleaning, the dataset had various industry categories. After mapping, these were consolidated into broader groups such as **Technology**, **Finance**, and **Healthcare**.

### **Educational Qualifications**
Educational qualifications were also standardized into broader categories, including **Bachelor**, **High School**, **Unspecified**, **Master**, and **Doctorate**.

### **Experience**
The dataset shows a wide range of experience requirements, with **Mid-Senior** and **Entry-level** positions being the most common.

### **Numerical Features Correlation Analysis**
The correlation matrix reveals that job postings with company logos have a significant negative correlation with fraudulence, while job_id shows a moderate positive correlation.

### **Bigram Analysis**
Bigrams from genuine and fraudulent job postings reveal distinct linguistic patterns, aiding in the identification of scam postings.

### **Industry Disparities**
Real job postings are dominated by the **Technology** sector, while fake postings often appear in the **Energy** sector.

### **Education Requirements**
Genuine postings often require a broader range of educational qualifications, whereas fake postings tend to target lower educational levels.

## **Splitting Data into Train-Test**
The dataset was split into training and testing sets using the `train_test_split` function, with a test size of **40%**.

## **Modeling and Vectorization**

### **Overview**
A variety of vectorization methods and machine learning algorithms were employed, including **TF-IDF**, **Count Vectorization**, **Word2Vec**, **Naive Bayes**, **Random Forest**, and **SVM**. Neural network models like **RNN** and **LSTM** were also explored, with these models showing superior performance in generalizing to unseen data.

## **Conclusion**
Our project successfully classified real and fake jobs using several machine learning models. **Naive Bayes**, **Random Forest**, and **SVM** achieved accuracies between 97% and 98%, but exhibited overfitting. **RNN** and **LSTM** models performed better on validation data, achieving **99%** accuracy, demonstrating their robustness in detecting employment scams.

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/fake-job-classification.git
   cd fake-job-classification
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## **How to Run the Project**

1. **Preprocess the data:**
   Run the data cleaning and preprocessing script:
   ```bash
   python preprocess_data.py
   ```

2. **Train the models:**
   Run the model training script:
   ```bash
   python train_model.py
   ```

3. **Evaluate the models:**
   Run the evaluation script to assess model performance:
   ```bash
   python evaluate_model.py
   ```

4. **Make predictions:**
   Use the trained model to classify new job postings:
   ```bash
   python predict.py --input sample_job_posting.txt
   ```
