import math
from scipy.stats import zscore
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt

meas = load_iris().data  # load 4 στήλες

X = meas[:, [2, 3]]  # χρήση των στηλών 3 και 4

dbscan = DBSCAN(eps=0.1, min_samples=5).fit(X)
IDX = dbscan.labels_

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1])  # Αναπαράσταση των 2 πρώτων στηλών του πίνακα X.
# Απλό διάγραμμα διασποράς των δεδομένων.
plt.title('Δεδομένα ΠΡΙΝ τη συσταδοποίηση με τον αλγόριθμο DBSCAN')
plt.show()

plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=IDX)
# Παρουσίαση των δεδομένων μετά την εφαρμογή του αλγορίθμου DBSCAN.
# Έχουν χωριστεί σε 2 συστάδες(1. Σημεία με κίτρινο χρώμα, 2, Σημεία με πράσινο χρώμα).
# Επίσης, τα σημεία θορύβου έχουν σχεδιαστεί με σκούρο μπλε χρώμα.
plt.title('Δεδομένα ΜΕΤΑ τη συσταδοποίηση με τον αλγόριθμο DBSCAN')
plt.show()
print("Silhouette Coefficient 1): %0.3f" % metrics.silhouette_score(X, IDX))

# Κανονικοποίηση του πίνακα X με τη μέθοδο zscore
# Έπειτα, εφαρμογή ξανά του αλγόριθμου συσταδοποίησης DBSCAN
# Παραλλαγή του δοθέντος κώδικα στο συγκεκριμένο σημείο, καθώς δεν ήταν δυνατή η οπτικοποίηση του γραφήματος

# xV1 = zscore(X[:, 0])
# xV2 = zscore(X[:, 1])
# X2 = [xV1, xV2]

X_normalized = zscore(X)

dbscan2 = DBSCAN(eps=0.1, min_samples=5).fit(X_normalized)
IDX2 = dbscan2.labels_

plt.figure(3)
# plt.scatter(xV1, xV2)
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=IDX)
plt.title('Δεδομένα ΜΕΤΑ την κανονικοποίηση του πίνακα X με τη μέθοδο zscore(DBSCAN)')
plt.show()
print("Silhouette Coefficient 2): %0.3f" % metrics.silhouette_score(X_normalized, IDX2))


# -----------------------------------------------------------------------------------------------
# Χρήση κώδικα από το Μέρος 1ο της εργασίας
# k-means συσταδοποίηση
# Μέτρα:SSE και Silhouette Coefficient


def k_means_clustering(X):
    k = 3  # 3 συστάδες, αφού υπάρχουν 3 είδη ίριδας στο dataset

    kmeans = KMeans(n_clusters=k).fit(X)

    IDX_kmeans = kmeans.labels_
    C_kmeans = kmeans.cluster_centers_

    plt.figure(4)
    plt.plot(X[IDX_kmeans == 0][:, 0], X[IDX_kmeans == 0][:, 1], 'limegreen', marker='o', linewidth=0, label='C1')
    plt.plot(X[IDX_kmeans == 1][:, 0], X[IDX_kmeans == 1][:, 1], 'yellow', marker='o', linewidth=0, label='C2')
    plt.plot(X[IDX_kmeans == 2][:, 0], X[IDX_kmeans == 2][:, 1], 'c.', marker='o', label='C3')

    plt.scatter(C_kmeans[:, 0], C_kmeans[:, 1], marker='x', color='black', s=150, linewidth=3, label="Centroids", zorder=10)
    plt.legend()
    plt.show()

    return IDX_kmeans, C_kmeans


def sse_calculation(X, numberOfRows, IDX_kmeans, C_means, k=3):
    sse = 0.0
    for i in range(k):
        for j in range(numberOfRows):
            if IDX_kmeans[j] == i:
                sse = sse + math.dist(X[j], C_means[i]) ** 2
    return sse


numberOfRows, numberOfColumns = X.shape

IDX3, C3 = k_means_clustering(X)
sse = sse_calculation(X, numberOfRows, IDX3, C3)

# Εκτύπωση αποτελεσμάτων για SSE και Silhouette Coefficient
print("Silhouette Coefficient(k-means): %0.3f" % metrics.silhouette_score(X, IDX3))
print("\n\nSSE(k-means) = %.3f" % sse)

# -------------------------------------------------------------------------------------------
# Αλλαγή στις παραμέτρους του αλγορίθμου DBSCAN
# Υπολογίζουμε ξανά το μέσο Συντελεστή Περιγράμματος τη συσταδοποίησης(πριν και μετά την κανονικοποίηση)
# Ακολουθούμε ακριβώς τα ίδια βήματα με πριν.

dbscan3 = DBSCAN(eps=0.5, min_samples=15).fit(X)
IDX3 = dbscan3.labels_

plt.figure(3)
plt.scatter(X[:, 0], X[:, 1], c=IDX3)
plt.title('Δεδομένα ΜΕΤΑ τη συσταδοποίηση με τον αλγόριθμο DBSCAN(eps=0.5 | min_samples=15)')
plt.show()
print("Silhouette Coefficient 3): %0.3f" % metrics.silhouette_score(X, IDX3))

dbscan4 = DBSCAN(eps=0.5, min_samples=15).fit(X_normalized)
IDX4 = dbscan4.labels_

plt.figure(4)
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=IDX4)
plt.title('Δεδομένα ΜΕΤΑ την κανονικοποίηση του πίνακα X με τη μέθοδο zscore(DBSCAN(eps=0.5 | min_samples=15))')
plt.show()
print("Silhouette Coefficient 4): %0.3f" % metrics.silhouette_score(X_normalized, IDX4))
