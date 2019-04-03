from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

captions = []
caption_file = open("descriptions.txt", encoding="utf8")
for caption in caption_file:
   captions.append(caption.split(' ', 1)[1])

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(captions)

true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),

print("\n")
print("Prediction")

Y = vectorizer.transform(["Anvesh have two dogs"])
prediction = model.predict(Y)
print("Anvesh have two dogs")
print(prediction)