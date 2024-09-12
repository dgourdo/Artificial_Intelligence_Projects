import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics

mat_file = scipy.io.loadmat('mydata.mat')  # Φόρτωση ενός αρχείο Matlab
X = np.array(mat_file['X'])  # Δημιουργία πίνακα x

dbscan = DBSCAN(eps=0.5, min_samples=15).fit(X)
# DBSCAN συσταδοποίηση με τη βοήθεια της συνάρτησης fit(όπως και για την k-means)
# Πυκνότητα: Αριθμός σημείων(min_samples) σε ακτίνα eps.
# 1) Κεντρικά σημεία/Σημεία πυρήνα:Έχουν πυκνότητα μεγαλύτερη από min_samples.
# 2) Οριακά σημεία:Έχουν πυκνότητα μικρότερη από min_samples,
#    αλλά απέχουν από ένα κεντρικό σημείο απόσταση μικρότερη από eps.
# 3) Θόρυβος:Ανήκουν σε περιοχές χαμηλής πυκνότητας,
#    είναι δηλαδή τα σημεία που δεν ανήκουν σε κάποια από τις 2 παραπάνω κατηγορίες.

IDX = dbscan.labels_  # Ετικέτες των συστάδων που εξάγονται από τον αλγόριθμο DBSCAN(σημεία πυρήνα,οριακά σημεία).

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1])  # Αναπαράσταση των 2 πρώτων στηλών του πίνακα X.
# Απλό διάγραμμα διασποράς των δεδομένων.
plt.title('Δεδομένα ΠΡΙΝ τη συσταδοποίηση με τον αλγόριθμο DBSCAN')
plt.show()

plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=IDX)  # παράμετρος c=IDX, ώστε να απεικονιστούν τα labels.
# Παρουσίαση των δεδομένων μετά την εφαρμογή του αλγορίθμου DBSCAN.
# Έχουν χωριστεί σε 2 συστάδες(1. Σημεία με κίτρινο χρώμα, 2, Σημεία με πράσινο χρώμα).
# Επίσης, τα σημεία θορύβου έχουν σχεδιαστεί με σκούρο μπλε χρώμα.
plt.title('Δεδομένα ΜΕΤΑ τη συσταδοποίηση με τον αλγόριθμο DBSCAN')
plt.show()


# -----------------------------------------------------------------------------------------------
# Χρήση κώδικα από το Μέρος 1ο της εργασίας
# k-means συσταδοποίηση
# Μέτρα:SSE και Silhouette Coefficient
# Σύγκριση (της ποιότητας-αποτελεσματικότητας) των αλγορίθμων DBSCAN και k-means

def k_means_clustering(X):
    k = 2  # 2 συστάδες(αφού σε χωρίζονται από τον DBSCAN)

    kmeans = KMeans(n_clusters=k).fit(X)

    IDX_kmeans = kmeans.labels_
    C_kmeans = kmeans.cluster_centers_

    plt.plot(X[IDX_kmeans == 0][:, 0], X[IDX_kmeans == 0][:, 1], 'limegreen', marker='o', linewidth=0, label='C1')
    plt.plot(X[IDX_kmeans == 1][:, 0], X[IDX_kmeans == 1][:, 1], 'yellow', marker='o', linewidth=0, label='C2')

    plt.scatter(C_kmeans[:, 0], C_kmeans[:, 1], marker='x', color='black', s=150, linewidth=3, label="Centroids", zorder=10)
    plt.legend()
    plt.title('Δεδομένα ΜΕΤΑ τη συσταδοποίηση με τον αλγόριθμο k-means')
    plt.show()

    return IDX_kmeans, C_kmeans  # χρειάζονται στη συνάρτηση sse_calculation, οπότε πρέπει να επιστρέφονται


def sse_calculation(X, numberOfRows, IDX_kmeans, C_means, k=2):
    sse = 0.0
    for i in range(k):
        for j in range(numberOfRows):
            if IDX_kmeans[j] == i:
                sse = sse + math.dist(X[j], C_means[i]) ** 2
    return sse


numberOfRows, numberOfColumns = X.shape

IDX2, C2 = k_means_clustering(X)
sse = sse_calculation(X, numberOfRows, IDX2, C2)

# Εκτύπωση αποτελεσμάτων για SSE και Silhouette Coefficient
print("Silhouette Coefficient(DBSCAN): %0.3f" % metrics.silhouette_score(X, IDX))
print("Silhouette Coefficient(k-means): %0.3f" % metrics.silhouette_score(X, IDX2))
print("\n\nSSE(k-means) = %.3f" % sse)

# -----------------------------------------------------------------------------------------------
# Αλλαγή των παραμέτρων MinPts(min_samples) και Eps(eps), ώστε να δούμε αλλαγές στο γράφημα,
# αλλά και στην ποιότητα της συσταδοποίησης.
# Σημείωση:Οι τιμές που δόθηκαν στις παραμέτρους eps και min_samples επιλέχθηκαν αυθαίρετα,
# για να φανεί απλώς η αντίδραση του dataset στις διάφορες τιμές τους.

# 1)
dbscan2 = DBSCAN(eps=0.5, min_samples=5).fit(X)
IDX2 = dbscan2.labels_  # Ετικέτες των συστάδων που εξάγονται από τον αλγόριθμο DBSCAN(σημεία πυρήνα,οριακά σημεία).

plt.figure(3)
plt.scatter(X[:, 0], X[:, 1], c=IDX2)
plt.title('eps=0.5 | min_samples=5')
plt.show()

dbscan3 = DBSCAN(eps=0.5, min_samples=25).fit(X)
IDX3 = dbscan3.labels_  # Ετικέτες των συστάδων που εξάγονται από τον αλγόριθμο DBSCAN(σημεία πυρήνα,οριακά σημεία).

plt.figure(4)
plt.scatter(X[:, 0], X[:, 1], c=IDX3)
plt.title('eps=0.5 | min_samples=25')
plt.show()

# 2)
dbscan4 = DBSCAN(eps=0.3, min_samples=15).fit(X)
IDX4 = dbscan4.labels_  # Ετικέτες των συστάδων που εξάγονται από τον αλγόριθμο DBSCAN(σημεία πυρήνα,οριακά σημεία).

plt.figure(5)
plt.scatter(X[:, 0], X[:, 1], c=IDX4)
plt.title('eps=0.3 | min_samples=15')
plt.show()

dbscan5 = DBSCAN(eps=0.7, min_samples=15).fit(X)
IDX5 = dbscan5.labels_  # Ετικέτες των συστάδων που εξάγονται από τον αλγόριθμο DBSCAN(σημεία πυρήνα,οριακά σημεία).

plt.figure(6)
plt.scatter(X[:, 0], X[:, 1], c=IDX5)
plt.title('eps=0.7 | min_samples=15')
plt.show()

# 3)
dbscan6 = DBSCAN(eps=0.3, min_samples=5).fit(X)
IDX6 = dbscan6.labels_  # Ετικέτες των συστάδων που εξάγονται από τον αλγόριθμο DBSCAN(σημεία πυρήνα,οριακά σημεία).

plt.figure(7)
plt.scatter(X[:, 0], X[:, 1], c=IDX6)
plt.title('eps=0.3 | min_samples=5')
plt.show()

dbscan7 = DBSCAN(eps=0.7, min_samples=25).fit(X)
IDX7 = dbscan7.labels_  # Ετικέτες των συστάδων που εξάγονται από τον αλγόριθμο DBSCAN(σημεία πυρήνα,οριακά σημεία).

plt.figure(8)
plt.scatter(X[:, 0], X[:, 1], c=IDX7)
plt.title('eps=0.7 | min_samples=25')
plt.show()

# 4)
dbscan8 = DBSCAN(eps=0.7, min_samples=5).fit(X)
IDX8 = dbscan8.labels_  # Ετικέτες των συστάδων που εξάγονται από τον αλγόριθμο DBSCAN(σημεία πυρήνα,οριακά σημεία).

plt.figure(9)
plt.scatter(X[:, 0], X[:, 1], c=IDX8)
plt.title('eps=0.7 | min_samples=5')
plt.show()

dbscan9 = DBSCAN(eps=0.3, min_samples=25).fit(X)
IDX9 = dbscan9.labels_  # Ετικέτες των συστάδων που εξάγονται από τον αλγόριθμο DBSCAN(σημεία πυρήνα,οριακά σημεία).

plt.figure(10)
plt.scatter(X[:, 0], X[:, 1], c=IDX9)
plt.title('eps=0.3 | min_samples=25')
plt.show()


#%%
