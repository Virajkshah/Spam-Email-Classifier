import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert label to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict Custom Message
def predict(message):
    message = vectorizer.transform([message])
    result = model.predict(message)
    return "Spam" if result[0] == 1 else "Not Spam"

print(predict("Congratulations! You won a free lottery ticket"))
