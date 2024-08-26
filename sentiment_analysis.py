import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import movie_reviews
import random

##Hassan Harb
##CPSC 481 AI
##04/26/2024

# Get the data set to train the model with
nltk.download('movie_reviews')

# Load the dataset
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

# Shuffle the documents to ensure random distribution of data
random.shuffle(documents)

# Separate data into texts and labels
texts = [" ".join(words) for words, category in documents]

labels = [category for words, category in documents]

# Create a DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert between text and numerical data using the CountVectorizer function.
vectorizer = CountVectorizer(stop_words='english')

X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes model
model = MultinomialNB()

model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification:\n", classification_report(y_test, y_pred))

# Test with new arbitrary reviews
new_reviews = ["This movie was awful, but I really liked the scenes with Scarlett Johansson","This was the best movie of all time! My favorite part was when Jared Leto morbed."]

new_reviews_vectorized = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_vectorized)

# Output the results
for review, sentiment in zip(new_reviews, predictions):
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")