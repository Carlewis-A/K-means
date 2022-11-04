
# Artificial Intelligence
# Name: Akana Carlewis Chambang
# Department: Computer Engineering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = np.random.randint(100, size=100)
x1 = np.random.randint(100, size=100)
y = list(zip(x, x1))
print('*' * 100)
print('Y \n', y)
print('*' * 100)

plt.figure(figsize=(10, 6))
plt.scatter(x, x1, s=50)
plt.show()
plt.close()

individual_clustering_score = []
for i in range(1, 4):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=3, random_state=0)
    kmeans.fit(y)
    individual_clustering_score.append(kmeans.inertia_)

print('*' * 100)
print('Individual Clustering Score \n', individual_clustering_score)
print('*' * 100)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 4), individual_clustering_score)
plt.ylabel('Number Of Clusters')
plt.xlabel('Clustering Score')
plt.title('Elbow Method')
plt.show()
plt.close()

Kmean = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=3, random_state=0)
Kmean.fit(y)
Centriods = Kmean.cluster_centers_
print('*' * 100)
print('Centriods \n', Centriods)
print('*' * 100)

Predic = kmeans.fit_predict(y)
print('*' * 100)
print('Prediction \n', Predic)
print('*' * 100)

plt.figure(figsize=(10, 6))
plt.scatter(x, x1, s=50, c=kmeans.labels_)
plt.scatter(Centriods[0][0], Centriods[0][1], s=200, marker='*')
plt.scatter(Centriods[1][0], Centriods[1][1], s=200, marker='*')
plt.scatter(Centriods[2][0], Centriods[2][1], s=200, marker='*')
plt.title('Clusters With Thier Centriod')
plt.show()
plt.close()
