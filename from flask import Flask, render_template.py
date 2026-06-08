from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)

# -----------------------------
# Text Preprocessing Function
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    return text.lower()

# -----------------------------
# Load & Balance Dataset
# -----------------------------
data = pd.read_csv("Youtube01.csv")
data = data[['CONTENT', 'CLASS']]
spam = data[data['CLASS'] == 1]
not_spam = data[data['CLASS'] == 0]
spam_upsampled = resample(spam, replace=True, n_samples=len(not_spam), random_state=42)
balanced_data = pd.concat([not_spam, spam_upsampled])
balanced_data['CLASS'] = balanced_data['CLASS'].map({0: 'NOT A SPAM COMMENT', 1: 'SPAM COMMENT'})

# -----------------------------
# Vectorization with TF-IDF
# -----------------------------
x = np.array([clean_text(text) for text in balanced_data['CONTENT']])
y = np.array(balanced_data['CLASS'])

cv = TfidfVectorizer(binary=True, stop_words='english', max_features=3000)
x = cv.fit_transform(x)

# -----------------------------
# Train/Test Split and Model
# -----------------------------
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

bnb = BernoulliNB()
svm = SVC(probability=True, random_state=42)
rf = RandomForestClassifier(random_state=42)
ensemble_model = VotingClassifier(estimators=[('bnb', bnb), ('svm', svm), ('rf', rf)], voting='soft')
ensemble_model.fit(xtrain, ytrain)

# -----------------------------
# Evaluate Model (Accuracy & F1 Score)
# -----------------------------
y_pred = ensemble_model.predict(xtest)
accuracy = round(accuracy_score(ytest, y_pred) * 100, 2)
f1 = round(f1_score(ytest, y_pred, pos_label='SPAM COMMENT') * 100, 2)

# -----------------------------
# Emoji & Sentiment Analyzer
# -----------------------------
analyzer = SentimentIntensityAnalyzer()
strong_spam_emojis = ['💰', '🎁', '🤑', '💸', '👉', '🔥', '🚀', '🎉', '😱', '🏆', '🔓', '📥', '🔗', '📢', '📧']

def contains_strong_spam_emoji(comment):
    return any(e in comment for e in strong_spam_emojis)

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_manual_features(comment):
    length = len(comment)
    num_emojis = len([char for char in comment if char in emoji.EMOJI_DATA])
    emojis = re.findall(r'[\U0001F300-\U0001F6FF]', comment)
    emoji_sentiment = sum(analyzer.polarity_scores(e)['compound'] for e in emojis)
    emoji_sentiment /= (num_emojis if num_emojis > 0 else 1)
    spam_emoji_flag = contains_strong_spam_emoji(comment)

    # Spam keywords
    spam_keywords = ['click', 'offer', 'free', 'win', 'money', 'subscribe', 'gift', 'visit', 'join', 'prize', 'now', 'act fast']
    text = clean_text(comment)
    keyword_flag = any(kw in text for kw in spam_keywords)

    return length, num_emojis, emoji_sentiment, spam_emoji_flag, keyword_flag

# -----------------------------
# Relevance Score
# -----------------------------
def calculate_relevance_score(comment, post=""):
    if not post:
        return 0
    comment_words = set(comment.lower().split())
    post_words = set(post.lower().split())
    common_words = comment_words.intersection(post_words)
    return len(common_words) / len(post_words) if post_words else 0

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('text_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    post = request.form.get('post', '')

    # Extract features
    processed_comment = emoji.demojize(comment)
    cleaned_text = clean_text(processed_comment)
    length, num_emojis, emoji_sentiment, spam_emoji_flag, keyword_flag = extract_manual_features(comment)
    
    transformed = cv.transform([cleaned_text]).toarray()
    prediction = ensemble_model.predict(transformed)[0]
    probabilities = ensemble_model.predict_proba(transformed)[0]
    class_index = list(ensemble_model.classes_).index(prediction)
    confidence = round(probabilities[class_index] * 100, 2)

    # Manual rule override
    if spam_emoji_flag or keyword_flag:
        prediction = 'SPAM COMMENT'
        confidence = max(confidence, 95.0)

    relevance_score = round(calculate_relevance_score(comment, post) * 100, 2)

    tooltip = (f"Comment Length: {length} chars | "
               f"Number of Emojis: {num_emojis} | "
               f"Emoji Sentiment Score: {emoji_sentiment:.2f} | "
               f"Strong Spam Emoji Detected: {'Yes' if spam_emoji_flag else 'No'} | "
               f"Keyword Spam Detected: {'Yes' if keyword_flag else 'No'} | "
               f"Relevance to Post: {relevance_score}%")

    return render_template('text_form.html',
                           comment=comment,
                           post=post,
                           result=prediction,
                           confidence=confidence,
                           processed=processed_comment,
                           tooltip=tooltip,
                           relevance_score=relevance_score,
                           emoji_sentiment=emoji_sentiment,
                           accuracy=accuracy,
                           f1=f1)

# -----------------------------
# Run the App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)