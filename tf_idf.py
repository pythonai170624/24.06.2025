import pandas as pd

df = pd.read_csv('smsspamcollection.tsv', sep='\t')
df.head()

# Count how many 'ham' and 'spam' emails we have in the dataset
df['label'].value_counts()

# Perform regular train/test split on our data
from sklearn.model_selection import train_test_split

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
# Get the feature names
feature_names = vectorizer.get_feature_names_out()

# Convert the first row to a dense array and get non-zero elements
row = X_train_tfidf[0].toarray()[0]
non_zero_indices = row.nonzero()[0]

# Print the non-zero values with their corresponding feature names
for idx in non_zero_indices:
    print(f"{feature_names[idx]}: {row[idx]}")

# Alternative method to print all non-zero elements in one line
print(dict(zip(feature_names[row.nonzero()[0]], row[row.nonzero()[0]])))

from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)

from sklearn.pipeline import Pipeline

svc_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                       ('svc', LinearSVC())
                      ])

svc_pipeline.fit(X_train, y_train)

predictions = svc_pipeline.predict(X_test)
predictions

from sklearn import metrics

print(metrics.confusion_matrix(y_test, predictions))
print()
print(metrics.classification_report(y_test, predictions))
print()
print(metrics.accuracy_score(y_test, predictions))

svc_pipeline.predict([
   "TEXT WON to 12345 to get your prize, you have been selected as WINNER!!"
])

svc_pipeline.predict(["Hey Tom, how are you today?"])

