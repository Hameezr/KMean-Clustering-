import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sns as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Instructions in comments

# For reading the data
df = pd.read_csv('C:\\Files\\FAST NUCES\\7th Semester\\Artificial Intelligence\\A4\\CC_GENERAL.csv')

# To remove NaN and infinite values
df.replace('null',np.NaN, inplace=True)
df.replace(r'^\s*$', np.NaN, regex=True, inplace=True)
df.fillna(value=df.mean(), inplace=True)

# Range of my dataset
X = df.iloc[:, 1:18]

# X_std = StandardScaler().fit_transform(df)

# Creating the PCA instances
pca = PCA(n_components=5, random_state=1)
principalComponents = pca.fit_transform(X)

# Plotting the explained variances
features = range(pca.n_components_)
plt.figure(1)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)

# Saving the PCA components to a DataFrame and plotting them
plt.figure(2)
PCA_components = pd.DataFrame(principalComponents)
p1, = plt.plot(PCA_components[0], 'b.')
p2, = plt.plot(PCA_components[1], 'g.')
p3, = plt.plot(PCA_components[2], 'm.')
p4, = plt.plot(PCA_components[3], 'r.')
p5, = plt.plot(PCA_components[4], 'y.')
plt.legend([p1, p2, p3, p4, p5], ['A', 'B', 'C', 'D', 'E'], loc='upper left')

# Using K means to find clusters of the PCA components
plt.figure(3)
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(principalComponents)
    distortions.append(kmeanModel.inertia_)

# Plotting the elbow plot to determine the value of optimal K
plt.plot(K, distortions, '-o', color='black')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.xticks(K)

# From the elbow plot, K=5
model = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=1)
model = model.fit(principalComponents)
y = model.predict(principalComponents)

# Scatter plotting the data now after minimizing the columns and predicting the best column values
plt.figure(4)
plt.scatter(principalComponents[y == 0, 0], principalComponents[y == 0, 1], s = 10, c = 'blue', label = 'Cluster 1')
plt.scatter(principalComponents[y == 1, 0], principalComponents[y == 1, 1], s = 10, c = 'yellow', label = 'Cluster 2')
plt.scatter(principalComponents[y == 2, 0], principalComponents[y == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
plt.scatter(principalComponents[y == 3, 0], principalComponents[y == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')
plt.scatter(principalComponents[y == 4, 0], principalComponents[y == 4, 1], s = 10, c = 'purple', label = 'Cluster 5')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 50, c = 'blue', label = 'Centroids')
plt.title('Clusters')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

# Using TSNE on minimized columns

tsne = TSNE()
tsne1 = TSNE(n_components=2, random_state =0)
tsne_result = tsne1.fit_transform(X)
target_ids = range(17)
target_names='BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',\
             'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', \
             'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', \
             'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE '

plt.figure(5)
colors = 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'grey', 'orange', \
         'indigo', 'yellowgreen', 'chocolate', 'gold', 'brown', 'lawngreen', 'dodgerblue'
for i, c, labels in zip(target_ids, colors, target_names):
    plt.scatter(tsne_result[y == i, 0], tsne_result[y == i, 1], c=c, label=labels)
plt.legend()
plt.show()
