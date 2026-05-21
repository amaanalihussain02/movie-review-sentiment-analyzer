Movie Review Sentiment Analyzer

Project Overview
This project is a movie review sentiment analysis web app. The app allows a user to enter a movie review and predicts whether the review sounds positive or negative. It also shows a confidence score for the prediction.

Project Topic
Sentiment Analysis Tool for Movie Reviews

Main Features
- User can type or paste a movie review.
- App predicts positive or negative sentiment.
- App shows a confidence score.
- App displays model comparison results.
- App shows evaluation metrics including accuracy, precision, recall and F1-score.
- App includes confusion matrices for model evaluation.

Technology Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Naive Bayes
- Logistic Regression
- Joblib
- Matplotlib

Dataset
The project uses the IMDb 50K Movie Reviews dataset.
The dataset contains:
- 50,000 total reviews
- 25,000 positive reviews
- 25,000 negative reviews

Model Training
Two models were trained and compared:

1. Naive Bayes
   Accuracy: 87.95%
   Precision: 87.39%
   Recall: 88.70%
   F1-score: 88.04%

2. Logistic Regression
   Accuracy: 90.86%
   Precision: 90.36%
   Recall: 91.48%
   F1-score: 90.92%

Final Selected Model
Logistic Regression was selected as the final model because it achieved the best F1-score and overall performance.

Folder Structure
Movie_Review_Sentiment_Analyzer/
- app.py
- train_model.py
- check_dataset.py
- requirements.txt
- README.txt
- data/
  - movie_reviews.csv
- models/
  - sentiment_model.pkl
  - tfidf_vectorizer.pkl
  - best_model_info.txt
- evaluation/
  - model_comparison_results.csv
  - logistic_regression_report.txt
  - logistic_regression_confusion_matrix.png
  - naive_bayes_report.txt
  - naive_bayes_confusion_matrix.png
- screenshots/

How to Run the Project

Step 1: Open the project folder in VS Code.

Step 2: Create and activate a virtual environment.

Command:
python -m venv venv

For Windows PowerShell:
.\venv\Scripts\Activate.ps1

Step 3: Install required packages.

Command:
pip install -r requirements.txt

Step 4: Check the dataset.

Command:
python check_dataset.py

Step 5: Train the model.

Command:
python train_model.py

Step 6: Run the web app.

Command:
streamlit run app.py

Step 7: Open the app in browser.

Local URL:
http://localhost:8501

How to Use the App
1. Open the app.
2. Go to the Check Review page.
3. Type or paste a movie review.
4. Click Check Review.
5. View the positive or negative prediction and confidence score.

Example Positive Review
The film had a strong story, great acting, and a very emotional ending. I really enjoyed watching it and would recommend it to others.

Example Negative Review
The movie was boring, slow, and a complete waste of time. The acting was weak and the story was dull.

Important Notes
- The app uses saved model files from the models folder.
- If the model files are missing, run train_model.py first.
- The dataset is only needed for checking and training.
- The trained app uses sentiment_model.pkl and tfidf_vectorizer.pkl for prediction.