import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# PATHS
# =========================
DATASET_PATH = "data/movie_reviews.csv"
MODEL_DIR = "models"
EVALUATION_DIR = "evaluation"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)


# =========================
# TEXT CLEANING FUNCTION
# =========================
def clean_text(text):
    """
    Cleans movie review text before training.
    """
    text = str(text)

    # Remove HTML tags such as <br />
    text = re.sub(r"<.*?>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove anything that is not a letter, number, or space
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# LOAD DATASET
# =========================
print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

# Remove missing values if any
df = df.dropna(subset=["review", "sentiment"])

# Clean review text
print("Cleaning review text...")
df["clean_review"] = df["review"].apply(clean_text)

# Convert labels to numbers
# positive = 1, negative = 0
df["label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Remove any rows where label mapping failed
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)


# =========================
# TRAIN TEST SPLIT
# =========================
X = df["clean_review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================
# TF-IDF VECTORIZATION
# =========================
print("Converting text into TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =========================
# MODEL TRAINING
# =========================
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear")
}

results = []

best_model = None
best_model_name = ""
best_f1 = 0

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })

    print(f"{model_name} Results:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    report = classification_report(
        y_test,
        y_pred,
        target_names=["negative", "positive"]
    )

    with open(f"{EVALUATION_DIR}/{model_name.replace(' ', '_').lower()}_report.txt", "w") as file:
        file.write(report)

    # Save confusion matrix for each model
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "positive"]
    )

    display.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"{EVALUATION_DIR}/{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = model_name


# =========================
# SAVE RESULTS
# =========================
results_df = pd.DataFrame(results)
results_df.to_csv(f"{EVALUATION_DIR}/model_comparison_results.csv", index=False)

print("\nModel comparison:")
print(results_df)

print("\nBest model:", best_model_name)


# =========================
# SAVE BEST MODEL AND VECTORIZER
# =========================
joblib.dump(best_model, f"{MODEL_DIR}/sentiment_model.pkl")
joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.pkl")

with open(f"{MODEL_DIR}/best_model_info.txt", "w") as file:
    file.write(f"Best model: {best_model_name}\n")
    file.write(f"Best F1 Score: {round(best_f1, 4)}\n")

print("\nTraining completed successfully.")
print("Saved model to: models/sentiment_model.pkl")
print("Saved vectorizer to: models/tfidf_vectorizer.pkl")
print("Saved evaluation results to: evaluation/")