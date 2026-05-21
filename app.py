import re
import joblib
import pandas as pd
import streamlit as st


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Analysis Tool for Movie Reviews",
    page_icon="🎬",
    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
    /* Hide Streamlit default toolbar / menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}

    .stApp {
        background: #f8fafc;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }

    .top-header {
        background: white;
        padding: 18px 26px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .brand-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 3px;
    }

    .brand-subtitle {
        font-size: 14px;
        color: #64748b;
    }

    .model-pill {
        background: #eff6ff;
        color: #1d4ed8;
        padding: 10px 16px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 700;
        border: 1px solid #bfdbfe;
    }

    .nav-card {
        background: white;
        padding: 10px 18px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 28px;
    }

    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 45%, #2563eb 100%);
        padding: 42px;
        border-radius: 24px;
        color: white;
        box-shadow: 0 16px 38px rgba(15, 23, 42, 0.18);
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 850;
        margin-bottom: 12px;
        line-height: 1.12;
    }

    .hero-subtitle {
        font-size: 17px;
        line-height: 1.7;
        color: #e5e7eb;
        max-width: 900px;
    }

    .section-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 22px;
    }

    .feature-card {
        background: white;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        min-height: 145px;
    }

    .feature-title {
        font-size: 19px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 8px;
    }

    .feature-text {
        color: #4b5563;
        font-size: 15px;
        line-height: 1.65;
    }

    .positive-box {
        background: #dcfce7;
        padding: 24px;
        border-radius: 16px;
        border-left: 8px solid #16a34a;
        color: #14532d;
        font-size: 24px;
        font-weight: 800;
        margin-top: 22px;
        margin-bottom: 15px;
    }

    .negative-box {
        background: #fee2e2;
        padding: 24px;
        border-radius: 16px;
        border-left: 8px solid #dc2626;
        color: #7f1d1d;
        font-size: 24px;
        font-weight: 800;
        margin-top: 22px;
        margin-bottom: 15px;
    }

    .summary-box {
        background: white;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-top: 20px;
        margin-bottom: 18px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding-top: 25px;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.25);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        color: white;
        border: none;
    }

    textarea {
        border-radius: 14px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 10px;
    }

    /* Horizontal navigation style */
    [data-testid="stRadio"] > div {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }

    [data-testid="stRadio"] label {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        padding: 8px 14px;
        border-radius: 999px;
        cursor: pointer;
    }

    [data-testid="stRadio"] label:hover {
        background: #e0ecff;
        border-color: #93c5fd;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# TEXT CLEANING FUNCTION
# =========================
def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# LOAD MODEL AND VECTORIZER
# =========================
@st.cache_resource
def load_model_files():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_model_files()
except FileNotFoundError:
    st.error("Model files were not found. Please run train_model.py first.")
    st.stop()


# =========================
# TOP HEADER
# =========================
st.markdown(
    """
    <div class="top-header">
        <div>
            <div class="brand-title">Sentiment Analysis Tool for Movie Reviews</div>
            <div class="brand-subtitle">A simple web app for checking positive and negative movie review sentiment</div>
        </div>
        <div class="model-pill">Final model: Logistic Regression</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# TOP NAVIGATION
# =========================
st.markdown('<div class="nav-card">', unsafe_allow_html=True)

page = st.radio(
    "Navigation",
    ["Home", "Check Review", "Model Results", "About"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# HOME PAGE
# =========================
if page == "Home":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Sentiment Analysis Tool for Movie Reviews</div>
            <div class="hero-subtitle">
                This project checks whether a movie review sounds mostly positive or negative.
                The user enters a review, and the app gives a prediction with a confidence score.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Review input</div>
                <div class="feature-text">
                    The user can type or paste a movie review directly into the app.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Sentiment prediction</div>
                <div class="feature-text">
                    The app predicts whether the review is mainly positive or negative.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Model results</div>
                <div class="feature-text">
                    The results page shows the testing evidence for the trained models.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">
            <h2>Project aim</h2>
            <p>
                The aim of this project is to build and test a sentiment analysis tool for movie reviews.
                The tool uses natural language processing and machine learning to classify review text
                as positive or negative.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-card">
            <h2>How the app works</h2>
            <p>
                The review is cleaned first, then converted into numerical features using TF-IDF.
                The trained Logistic Regression model then predicts the sentiment and returns a confidence score.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# CHECK REVIEW PAGE
# =========================
elif page == "Check Review":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Check a movie review</div>
            <div class="hero-subtitle">
                Paste a movie review below and the app will predict whether it sounds positive or negative.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    user_review = st.text_area(
        "Movie review:",
        height=210,
        placeholder="Example: The film had a strong story, great acting and a very emotional ending."
    )

    check_button = st.button("Check Review")

    if check_button:
        if user_review.strip() == "":
            st.warning("Please enter a movie review first.")
        else:
            cleaned_review = clean_text(user_review)
            review_vector = vectorizer.transform([cleaned_review])
            prediction = model.predict(review_vector)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(review_vector)[0]
                confidence = max(probabilities) * 100
            else:
                confidence = 0

            if prediction == 1:
                st.markdown(
                    """
                    <div class="positive-box">
                        This review sounds positive.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="negative-box">
                        This review sounds negative.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Confidence", f"{confidence:.2f}%")

            with col2:
                st.metric("Model", "Logistic Regression")

            with col3:
                st.metric("Words entered", len(user_review.split()))

            st.progress(int(confidence))

            with st.expander("View cleaned text used by the model"):
                st.write(cleaned_review)


# =========================
# MODEL RESULTS PAGE
# =========================
elif page == "Model Results":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Model results</div>
            <div class="hero-subtitle">
                Two models were tested. Logistic Regression performed better, so it was selected for the final app.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        results = pd.read_csv("evaluation/model_comparison_results.csv")

        st.markdown(
            """
            <div class="summary-box">
                <h3>Testing summary</h3>
                <p>
                    The dataset was split into training and testing data. Naive Bayes was used as the baseline model,
                    and Logistic Regression was used as the improved model. The final choice was based mainly on
                    the F1-score because it gives a balanced view of precision and recall.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Model comparison")
        st.dataframe(results, use_container_width=True)

        logistic_row = results[results["Model"] == "Logistic Regression"].iloc[0]

        st.subheader("Final model performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{logistic_row['Accuracy'] * 100:.2f}%")

        with col2:
            st.metric("Precision", f"{logistic_row['Precision'] * 100:.2f}%")

        with col3:
            st.metric("Recall", f"{logistic_row['Recall'] * 100:.2f}%")

        with col4:
            st.metric("F1-score", f"{logistic_row['F1 Score'] * 100:.2f}%")

        st.success("Logistic Regression was selected as the final model.")

        tab1, tab2 = st.tabs(
            ["Logistic Regression confusion matrix", "Naive Bayes confusion matrix"]
        )

        with tab1:
            st.image(
                "evaluation/logistic_regression_confusion_matrix.png",
                use_container_width=True
            )

        with tab2:
            st.image(
                "evaluation/naive_bayes_confusion_matrix.png",
                use_container_width=True
            )

        st.markdown(
            """
            <div class="summary-box">
                <h3>What the results show</h3>
                <p>
                    Logistic Regression achieved an accuracy of about 90.86% and an F1-score of about 90.92%.
                    This means it handled the test reviews well and performed better than the baseline model.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    except FileNotFoundError:
        st.warning("Evaluation files were not found. Please run train_model.py first.")


# =========================
# ABOUT PAGE
# =========================
elif page == "About":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">About the project</div>
            <div class="hero-subtitle">
                This project was built as a practical sentiment analysis tool for movie reviews.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h2>Tools used</h2>
                <ul>
                    <li>Python for development</li>
                    <li>Pandas for handling the dataset</li>
                    <li>Scikit-learn for model training</li>
                    <li>TF-IDF for text features</li>
                    <li>Streamlit for the web interface</li>
                    <li>Joblib for saving the trained model</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h2>Process followed</h2>
                <ol>
                    <li>The IMDb review dataset was loaded.</li>
                    <li>The review text was cleaned.</li>
                    <li>The data was split into training and testing sets.</li>
                    <li>Two models were trained and compared.</li>
                    <li>The better model was connected to this app.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div class="footer">
        Sentiment Analysis Tool for Movie Reviews | Python and Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)