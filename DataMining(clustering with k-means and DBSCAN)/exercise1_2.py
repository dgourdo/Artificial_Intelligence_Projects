import math
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

mat_file = scipy.io.loadmat('xV.mat')  # Φόρτωση ενός αρχείο Matlab
xV = np.array(mat_file['xV'])  # Δημιουργία πίνακα x


# Δημιουργία των συναρτήσεων k_means_clustering και sse_calculation, καθώς τα βήματα 2,3,4 και 5
# χρησιμοποιούν τον ίδιο κώδικα.


def k_means_clustering(X):
    # Οι επόμενες εντολές έχουν αναλυθεί για την άσκηση 1.1(αρχείο exercise1_1.py)
    k = 3  # 3 συστάδες

    kmeans = KMeans(n_clusters=k).fit(X)
    # Η μέθοδος πραγματοποιεί τη συσταδοποίηση, με την Ευκλείδεια Απόσταση(η μόνη απόσταση που επιτρέπεται στην Python
    # για k-means).

    IDX = kmeans.labels_  # Η μέθοδος labels_ επιστρέφει μία λίστα με τις ετικέτες των δειγμάτων στις συστάδες,
    # δηλαδή την ανάθεση κάθε δείγματος στην εκάστοτε συστάδα.
    C = kmeans.cluster_centers_  # Η μέθοδος cluster_centers_ περιέχει τις συντεταγμένες των κέντρων των συστάδων που
    # δημιουργήθηκαν κατά τη λειτουργία του αλγόριθμου συσταδοποίησης k-means.

    plt.plot(X[IDX == 0][:, 0], X[IDX == 0][:, 1], 'limegreen', marker='o', linewidth=0, label='C1')
    plt.plot(X[IDX == 1][:, 0], X[IDX == 1][:, 1], 'yellow', marker='o', linewidth=0, label='C2')
    plt.plot(X[IDX == 2][:, 0], X[IDX == 2][:, 1], 'c.', marker='o', label='C3')

    plt.scatter(C[:, 0], C[:, 1], marker='x', color='black', s=150, linewidth=3, label="Centroids", zorder=10)
    plt.legend()
    plt.show()

    return IDX, C  # χρειάζονται στη συνάρτηση sse_calculation, οπότε πρέπει να επιστρέφονται


def sse_calculation(X, numberOfRows, IDX, C, k=3):  # k: default παράμετρος, αφού παραμένει 3 για όλα τα βήματα
    # Στη συνάρτηση χρησιμοποιούμε τον ίδιο κώδικα με το αρχείο exercise1_1 για τον υπολογισμό του SSE
    sse = 0.0
    for i in range(k):
        for j in range(numberOfRows):
            if IDX[j] == i:
                sse = sse + math.dist(X[j], C[i]) ** 2
    return sse


# Βήμα 2
X2 = xV[:, [0, 1]]  # Χρήση των πρώτων 2 στηλών - χαρακτηριστικών του πίνακα xV
numberOfRows2, numberOfColumns2 = X2.shape

IDX2, C2 = k_means_clustering(X2)
sse2 = sse_calculation(X2, numberOfRows2, IDX2, C2)
print("\n\nSSE2 = %.3f" % sse2)

# Βήμα 3
X3 = xV[:, [296, 305]]  # Χρήση των στηλών - χαρακτηριστικών 297 και 306 του πίνακα xV
numberOfRows3, numberOfColumns3 = X3.shape

IDX3, C3 = k_means_clustering(X3)
sse3 = sse_calculation(X3, numberOfRows3, IDX3, C3)
print("\n\nSSE3 = %.3f" % sse3)

# Βήμα 4
X4 = xV[:, [-1, -2]]  # Χρήση των 2 τελευταίων στηλών - χαρακτηριστικών του πίνακα xV
numberOfRows4, numberOfColumns4 = X4.shape

IDX4, C4 = k_means_clustering(X4)
sse4 = sse_calculation(X4, numberOfRows4, IDX4, C4)
print("\n\nSSE4 = %.3f" % sse4)

# Βήμα 5
X5 = xV[:, [205, 175]]  # Χρήση των 2 τελευταίων στηλών - χαρακτηριστικών του πίνακα xV
numberOfRows5, numberOfColumns5 = X5.shape

IDX5, C5 = k_means_clustering(X5)
sse5 = sse_calculation(X5, numberOfRows5, IDX5, C5)
print("\n\nSSE5 = %.3f" % sse5)


# Βήμα 6
# Σύγκριση των αποτελεσμάτων των παραπάνω βημάτων


def plot_sse_per_step(sse_values, steps):
    # Γραφική Παράσταση SSE - βήματα
    plt.figure(1)
    plt.plot(steps, sse_values, marker='o')
    plt.title('SSE για διαφορετικά βήματα της άσκησης')
    plt.xlabel('Βήματα(2,3,4 και 5)')
    plt.ylabel('SSE')
    plt.show()


sse_values = [sse2, sse3, sse4, sse5]
steps = [2, 3, 4, 5]
# Σύγκριση της τιμής του SSE σε κάθε βήμα, δηλαδή για κάθε διαφορετικό ζευγάρι στηλών.
# Κάθε ζευγάρι στηλών του πίνακα xV έχει διαφορετική επίδραση στη συσταδοποίηση, γεγονός που παρατηρείται
# και με <<γυμνό μάτι>> κατά το σχεδιασμό των γραφικών παραστάσεων της συνάρτησης k_means_clustering,
# δηλαδή των γραφικών παραστάσεων των δειγμάτων στο χώρο των χαρακτηριστικών.
plot_sse_per_step(sse_values, steps)

# Για τη σύγκριση των αποτελεσμάτων των παραπάνω βημάτων θα χρησιμοποιήσουμε και το Συντελεστή Περιγράμματος
silhouette_coefficient_values = []
# Υπολογισμός Silhouette Coefficient
silhouette_coefficient2 = metrics.silhouette_score(X2, IDX2)
silhouette_coefficient3 = metrics.silhouette_score(X3, IDX3)
silhouette_coefficient4 = metrics.silhouette_score(X4, IDX4)
silhouette_coefficient5 = metrics.silhouette_score(X5, IDX5)

# Εισαγωγή του υπολογιζόμενου silhouette_coefficient στη λίστα silhouette_coefficient_values
silhouette_coefficient_values.append(silhouette_coefficient2)
silhouette_coefficient_values.append(silhouette_coefficient3)
silhouette_coefficient_values.append(silhouette_coefficient4)
silhouette_coefficient_values.append(silhouette_coefficient5)


# Έπειτα δημιουργούμε παρόμοια συνάρτηση με την plot_sse_per_step, για τις τιμές του Συντελεστή Περιγράμματος
# ανά Βήμα της άσκησης

def plot_silhouette_coefficient_per_step(silhouette_coefficient_values, steps):
    # Γραφική Παράσταση SSE - βήματα
    plt.figure(2)
    plt.plot(steps, silhouette_coefficient_values, marker='o')
    plt.title('Silhouette Coefficient για διαφορετικά βήματα της άσκησης')
    plt.xlabel('Βήματα(2,3,4 και 5)')
    plt.ylabel('Silhouette Coefficient')
    plt.show()


plot_silhouette_coefficient_per_step(silhouette_coefficient_values, steps)
