from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import math

meas = load_iris().data  # load 4 στήλες

X = meas[:, [2, 3]]  # χρήση των στηλών 3 και 4
k = 3  # 3 συστάδες, αφού υπάρχουν 3 είδη ίριδας στο dataset

kmeans = KMeans(n_clusters=k).fit(X)  # όπου n_clusters:αριθμός συστάδων
# Η μέθοδος πραγματοποιεί τη συσταδοποίηση, με την Ευκλείδεια Απόσταση(η μόνη απόσταση που επιτρέπεται στην Python
# για k-means).

IDX = kmeans.labels_  # Η μέθοδος labels_ επιστρέφει μία λίστα με τις ετικέτες των δειγμάτων στις συστάδες,
# δηλαδή την ανάθεση κάθε δείγματος στην εκάστοτε συστάδα.
C = kmeans.cluster_centers_  # Η μέθοδος cluster_centers_ περιέχει τις συντεταγμένες των κέντρων των συστάδων που
# δημιουργήθηκαν κατά τη λειτουργία του αλγόριθμου συσταδοποίησης k-means.

plt.figure(1)
# Διάγραμμα 1
plt.plot(IDX[:], 'o')  # Παρουσίαση των δειγμάτων με μπλε κυκλάκια.
# Τα δείγματα της 1ης συστάδας βρίσκονται σε ευθεία οριζόντια γραμμή, όπου στον άξονα y Cluster Label=0.
plt.ylabel('Cluster Label')  # 0,1,2
plt.show()

# Διάγραμμα 2
# Οι 3 εντολές που ακολουθούν χρησιμοποιούνται για παρασταθούν τα δείγματα στο χώρο(50 για κάθε είδος
# ίριδας) των χαρακτηριστικών, ώς κυκλάκια(marker='o'). Αξίζει να αναφερθεί ότι τα δείγματα κάθε συστάδας
# παρουσιάζονται με διαφορετικό χρώμα(π.χ. limegreen για τη 1η συστάδα) Επιπροσθέτως, η παράμετρος linewidth=0
# (πάχος γραμμής) σημαίνει πρακτικά ότι δεν υπάρχει γραμμή που να συνδέει τα δείγματα μεταξύ τους.
# Τέλος, η εντολή:X[IDX == 0][:, 0],X[IDX == 0][:, 1]
# σηματοδοτεί τη χρήση αποκλειστικά των στηλών 0 και 1 της 1ης συστάδας(IDX == 0), δηλαδή των
# δειγμάτων που έχουν το 0 για ετικέτα. Αυτά τα δείγματα έχουν στη γραφική παράσταση την ετικέτα C1(label='C1')
plt.plot(X[IDX == 0][:, 0], X[IDX == 0][:, 1], 'limegreen', marker='o', linewidth=0, label='C1')
plt.plot(X[IDX == 1][:, 0], X[IDX == 1][:, 1], 'yellow', marker='o', linewidth=0, label='C2')
plt.plot(X[IDX == 2][:, 0], X[IDX == 2][:, 1], 'c.', marker='o', label='C3')

plt.scatter(C[:, 0], C[:, 1], marker='x', color='black', s=150, linewidth=3, label="Centroids", zorder=10)
# Διάγραμμα διασποράς:Εμφάνιση των κέντρων των συστάδων ως μαύρα Χ στο χώρο των χαρακτηριστικών.
# Το s αναφέρεται στο μέγεθος των μαύρων Χ, που έχουν και την ετικέτα Centroids.
# Η παράμετρος zorder=10 χρησιμοποιείται με στόχο την εμφάνιση των κέντρων πάνω από τα δείγματα της κάθε συστάδας,
# δηλαδή για να είναι ευδιάκριτα.
plt.legend()
plt.show()

# Βήμα 4
# Ανάλυση του SSE και του συντελεστή περιγράμματος στο αρχείο .pdf

# Υπολογισμός SSE, Silhouette Coefficient για τη συγκεκριμένη συσταδοποίηση(k=3)
# Χρήση εντολών από το αρχείο που υπάρχει στο eclass: K-means_example1.py
numberOfRows, numberOfColumns = X.shape
sse = 0.0

for i in range(k):
    for j in range(numberOfRows):
        if IDX[j] == i:
            sse = sse + math.dist(X[j], C[i])**2

print("\n\nSSE = %.3f" % sse)

if k < numberOfRows:
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, IDX))

k_values = [3, 4, 5, 6, 7, 10]  # Τιμές του k για τις οποίες θα υπολογίσουμε τα SSE και Silhouette Coefficient

# Αρχικοποίηση λιστών που περιέχουν τις τιμές των SSE και Silhouette Coefficient για τις διάφορες τιμές του k
sse_values = []
silhouette_coefficient_values = []

for k in k_values:  # loop για τις διάφορες τιμές του k(αριθμός συστάδων)
    kmeans = KMeans(n_clusters=k).fit(X)
    IDX = kmeans.labels_
    C = kmeans.cluster_centers_

    # Υπολογισμός SSE
    sse = 0.0
    for i in range(k):
        for j in range(numberOfRows):
            if IDX[j] == i:
                sse = sse + math.dist(X[j], C[i]) ** 2
    sse_values.append(sse)  # Εισαγωγή του υπολογιζόμενου sse στη λίστα sse_values

    # Υπολογισμός Silhouette Coefficient
    silhouette_coefficient = metrics.silhouette_score(X, IDX)
    # Εισαγωγή του υπολογιζόμενου silhouette_coefficient στη λίστα silhouette_coefficient_values
    silhouette_coefficient_values.append(silhouette_coefficient)

# Γραφική Παράσταση 1 (SSE - k)
plt.figure(2)
plt.plot(k_values, sse_values, marker='o')
plt.title('SSE για διαφορετικά k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE')
plt.show()

# Γραφική Παράσταση 1 (Silhouette Coefficient - k)
plt.figure(3)
plt.plot(k_values, silhouette_coefficient_values, marker='o')
plt.title('Silhouette Coefficient για διαφορετικά k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Coefficient')
plt.show()

#%%
