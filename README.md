# IMDb Sentiment Analysis

## Overview
This project performs sentiment analysis on IMDb movie reviews.  
The goal is to classify reviews as **positive** or **negative** using NLP techniques and machine learning models.

## Tech Stack
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- NLTK (text preprocessing)
- TensorFlow/Keras (LSTM model)

## Project Structure
- `data/` → dataset (IMDb reviews)
- `notebooks/` → Jupyter notebook with code
- `requirements.txt` → dependencies
- `README.md` → documentation
## Dataset
The dataset is too large to upload on GitHub.  
Download from Kaggle: [IMDb Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Place the dataset inside the `data/` folder before running the notebook.

##  Key Steps
1. Data cleaning and text preprocessing (stopwords removal, stemming, tokenization)
2. Feature extraction (TF-IDF / word embeddings)
3. Model training:
   - Logistic Regression
   - Naive Bayes
   - LSTM (deep learning)
4. Model evaluation (accuracy, confusion matrix, F1-score)

##  Results
- Logistic Regression: ~85% accuracy
- Naive Bayes: ~83% accuracy
- LSTM Model: ~90% accuracy (best performer)

##  How to Run
1. Clone this repo  
   ```bash
   git clone https://github.com/<your-username>/IMDb-Sentiment-Analysis.git
   cd IMDb-Sentiment-Analysis
