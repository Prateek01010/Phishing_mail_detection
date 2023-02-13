import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the email data into a pandas dataframe
emails_df = pd.read_csv('emails.csv')

# Vectorize the email text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
email_features = vectorizer.fit_transform(emails_df['text'])

# Cluster the email features using K-means clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(email_features)

# Assign each email to a cluster
cluster_labels = kmeans.predict(email_features)

# Add the cluster labels to the email dataframe
emails_df['cluster'] = cluster_labels

# Inspect the distribution of emails in each cluster
print(emails_df.groupby('cluster').count())
