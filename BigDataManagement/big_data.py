import math
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import time

# Διαβάζουμε το dataset με τη βοήθεια ενός DataFrame
data = pd.read_csv('Billionaires Statistics Dataset.csv')

print("Στατιστικά")
print(data.describe())
print("\n")

print("Πληροφορίες για το dataset")
print(data.info())
print("\n\n")

# Καθαρισμός και Προεπεξεργασία δεδομένων
print("Έλεγχος για ελλιπείς τιμές")
print(data.isnull().sum())
print("\n")

print("Έλεγχος για διπλότυπα(σύνολο διπλότυπων)")
print(data.duplicated().sum())
print("\n")

print("Χειρισμός κατηγορικών(και boolean) δεδομένων: Στήλες category, selfMade, gender\n")
# Χρησιμοποιούμε τη μέθοδο unique() για να βρούμε τις μοναδικές τιμές στη στήλη "category"
unique_categories = data['category'].unique()
# print(unique_categories)
print("\n")

# Δημιουργούμε ένα αντικείμενο LabelEncoder
label_encoder = LabelEncoder()

# Εφαρμόζουμε το LabelEncoder στη στήλη category
data['category_encoded'] = label_encoder.fit_transform(data['category'])

# Εμφανίζουμε τις αντιστοιχίσεις κατηγοριών με αριθμούς
print(dict(zip(data['category'], data['category_encoded'])))
print("\n")

# selfMade [False  True] -> (0,1)
data['selfMade_encoded'] = label_encoder.fit_transform(data['selfMade'])

# Εμφανίζουμε τις αντιστοιχίσεις
print(dict(zip(data['selfMade'], data['selfMade_encoded'])))
print("\n")

# gender ['M' 'F'] -> (1,0)
data['gender_encoded'] = label_encoder.fit_transform(data['gender'])

# Εμφανίζουμε τις αντιστοιχίσεις
print(dict(zip(data['gender'], data['gender_encoded'])))
print("\n")

# Δημιουργία νέου csv αρχείου που περιέχει τις στήλες category_encoded,selfMade_encoded, gender_encoded
data.drop(['category', 'selfMade', 'gender'], axis=1, inplace=True)

data.to_csv('Billionaires Statistics Dataset_version2.csv' , index=False)

# Αφαιρούμε από το updated dataset εγγραφές που περιέχουν ελλιπείς τιμές σε κάποια από τις εξής στήλες:
# age,country,cpi_country,gdp_country,gross_tertiary_education_enrollment,total_tax_rate_country
data_upd = pd.read_csv('Billionaires Statistics Dataset_version2.csv')

missing_values = ['age', 'country', 'cpi_country', 'gdp_country', 'gross_tertiary_education_enrollment',
                  'total_tax_rate_country']

data_upd.dropna(subset=missing_values, inplace=True)
data_upd.to_csv('Billionaires Statistics Dataset_version2.csv', index=False)
# Συνολικές εγγραφές: 2641 -> 2408

# backup του αρχείου
shutil.copyfile('Billionaires Statistics Dataset_version2.csv', 'backup_Billionaires Statistics Dataset_version2.csv')

# Διορθώσεις στη στήλη rank που προκύπτει από το finalWorth
# (από τη διαγραφή ορισμένων εγγραφών δημιουργήθηκαν θέματα στο rank)
data_upd = pd.read_csv('Billionaires Statistics Dataset_version2.csv')
data_upd['rank'] = data_upd['finalWorth'].rank(ascending=False).astype(int)
data_upd.to_csv('Billionaires Statistics Dataset_version2.csv', index=False)

# Χειρισμός ακραίων τιμών(Πηγές: https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/ ,
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html ,
# https://www.geeksforgeeks.org/python-seaborn-tutorial/)

# Έλεγχος των τιμών των στηλών:finalWorth,age,gdp_country,gross_tertiary_education_enrollment,total_tax_rate_country
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(data_upd['finalWorth'], ax=axs[0, 0])
axs[0, 0].set_title('Αρχικό Box Plot της στήλης finalWorth')

sns.boxplot(data_upd['age'], ax=axs[0, 1])
axs[0, 1].set_title('Αρχικό Box Plot της στήλης age')

sns.boxplot(data_upd['gdp_country'], ax=axs[1, 0])
axs[1, 0].set_title('Αρχικό Box Plot της στήλης gdp_country')

sns.boxplot(data_upd['total_tax_rate_country'], ax=axs[1, 1])
axs[1, 1].set_title('Αρχικό Box Plot της στήλης total_tax_rate_country')

plt.suptitle('Εύρεση ακραίων τιμών')
plt.tight_layout()
plt.show()

# Ανάλυση Συσχέτισης και Μείωση μεγέθους του συνόλου δεδομένων

# Πηγή: https://www.geeksforgeeks.org/exploring-correlation-in-python/

# Για στήλες που περιέχουν αριθμητικά δεδομένα
corr_columns = ['age', 'birthYear', 'category_encoded', 'finalWorth', 'gender_encoded']
corr = data_upd[corr_columns].corr(method='pearson')  # χρήση της συσχέτισης Pearson

plt.figure(figsize=(10, 8), dpi=50)
sns.heatmap(corr, annot=True, fmt=".2f", linewidth=.5, annot_kws={"size": 18})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Για στήλες που περιέχουν κατηγορικά δεδομένα:
# Αναμφίβολα οι στήλες personName-lastName, country-countryOfCitizenship, organization-source
# εμφανίζουν ανά 2 πολύ ισχυρή συσχέτιση, οπότε μπορούμε να κρατήσουμε στο σύνολο δεδομένων μας
# μόνο μία από κάθε ζευγάρι(δε χρησιμοποιείται η Pearson για κατηγορικές μεταβλητές στην Python).


# Μείωση του μεγέθους του συνόλου δεδομένων
# Αριθμός στηλών: 35 -> 13
# Προέκυψε έπειτα από την ανάλυση συσχέτισης, αλλά και λόγω του ότι κάποιες στήλες
# δεν ταίριαζαν στη συγκεκριμένη εργασία και στην ανάλυση που θέλουμε να κάνουμε.

final_columns = ['rank', 'finalWorth', 'category_encoded', 'personName', 'age', 'country', 'source', 'selfMade_encoded',
                 'gender_encoded','cpi_country', 'gdp_country', 'gross_tertiary_education_enrollment', 'total_tax_rate_country']

final_data = data_upd[final_columns]
final_data.to_csv('Billionaires Statistics Dataset_version2.csv', index=False)

'''
Τελικές Στήλες:
rank: The ranking of the billionaire in terms of wealth.
finalWorth: The final net worth of the billionaire in U.S. dollars.
category_encoded: The category or industry in which the billionaire's business operates.
personName: The full name of the billionaire. [object]
age: The age of the billionaire.
country: The country in which the billionaire resides. [object]
source: The source of the billionaire's wealth. [object]
selfMade_encoded: Indicates whether the billionaire is self-made (True/False).
gender_encoded: The gender of the billionaire.
cpi_country: Consumer Price Index (CPI) for the billionaire's country.
gdp_country: Gross Domestic Product (GDP) for the billionaire's country.
gross_tertiary_education_enrollment: Enrollment in tertiary education in the billionaire's country.
total_tax_rate_country: Total tax rate in the billionaire's country.
'''

# Κανονικοποίηση
columns = ['finalWorth', 'age', 'cpi_country', 'gdp_country', 'gross_tertiary_education_enrollment',
           'total_tax_rate_country']

# Δημιουργούσε πρόβλημα η στήλη gdp_country(περιέχει τα σύμβολα ',' και '$")
data_upd['gdp_country'] = data_upd['gdp_country'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Πηγή: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# Χρήση της μεθόδου z-score, καθώς παρέχει καλύτερα αποτελέσματα σε μεταβλητές που υπάρχουν ακραία δεδομένα
scaler = StandardScaler()
normalized_data = data_upd.copy()  # αντίγραφο του dataframe

normalized_data[columns] = scaler.fit_transform(data_upd[columns])

normalized_data.to_csv('Billionaires Statistics Dataset_normalized.csv', index=False)


# Δημιουργία συναρτήσεων για την εφαρμογή των μεθόδων ιεραρχική συσταδοποίηση, k-means και DBSCAN,
# ώστε να μπορούμε να συγκρίνουμε τους αλγορίθμους για τουλάχιστον 1 ζευγάρι στηλών του dataset,
# αλλά και να αλλάξουμε κάποιες άλλες παραμέτρους.

# -------------------------------------------------------------------------------------------------------
# ---------------------------- ΙΕΡΑΡΧΙΚΗ ΟΜΑΔΟΠΟΙΗΣΗ-----------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def hierarchical_clustering(filename, chosen_columns, num_of_clusters=3, distance_metric='cityblock'):
    # Για την ομαδοποίηση θα χρησιμοποιήσουμε το κανονικοποιημένο .csv αρχείο
    data_n = pd.read_csv(filename)
    clustering_data = data_n[chosen_columns].values

    # Για το clustering πρέπει να διαλέξουμε 2 στήλες από τη λίστα, ανάλογα τι συμπεράσματα θέλουμε να παρουσιάσουμε,
    # για τώρα έχουμε επιλέξει τον πλούτο και την ηλικία.

    # Υπολογισμός απόστασης των επιλεγμένων δεδομένων, υπάρχουν πολλοί τρόποι μερικοί απο αυτούς είναι
    # η Ευκλείδεια και Μανχάταν απόσταση
    # Επιλέγουμε ανάμεσα στις 2 μεθόδους υπολογισμού απόστασης και την περνάμε ως παράμετρο
    # distance_mat = pdist(clustering_data, metric='euclidean')
    distance_mat = pdist(clustering_data, metric=distance_metric)
    linkage_mat = linkage(distance_mat, method='ward')

    # Το Δεντρόγραμμα της Ιεραρχικής Συσταδοποίησης
    plt.figure(figsize=(10, 8))
    dendrogram(linkage_mat, labels=data_n['personName'].values, leaf_rotation=90, leaf_font_size=10)
    plt.title('Δεντρόγραμμα της Ιεραρχικής Συσταδοποίησης ')
    plt.xlabel('Ονόματα')
    plt.ylabel(f'Απόσταση {distance_metric.capitalize()}')
    plt.show()

    # Μέτρηση του χρόνου εκτέλεσης της μεθόδου ιεραρχική συσταδοποίηση
    start_time1 = time.time()

    # Φτιάχνουμε τα clusters.
    # Με το δεντρόγραμμα υπολογίσαμε τις αποστάσεις μεταξύ των σημείων.
    # Επιλέγοντας συγκεκριμένο αριθμό συστάδων(με τη χρήση της παραμέτρου criterion='maxclust')
    # πραγματοποιείται κόψιμο του δεντρογράμματος σε συγκεκριμένο σημείο.
    clusters = fcluster(linkage_mat, num_of_clusters, criterion='maxclust')

    time1 = time.time() - start_time1

    # Προσθέτουμε τις συστάδες στο dataframe που χρησιμοποιούμε και το κάνουμε save σε ένα άλλο .csv αρχείο
    data_n['Cluster'] = clusters
    data_n.to_csv('Billionaires Statistics Dataset_clusters.csv', index=False)

    # Αποτελεί ένα μέτρο που χρησιμοποιείται για να παρέχει πληροφορίες
    # για την τοποθέτηση ενός δείγματος στο χώρο, σε σχέση με τις συστάδες.
    # Εύρος τιμών:-1 έως 1
    # Μεγάλες τιμές του συντελεστή, δηλαδή τιμές κοντά στο 1 σημαίνουν καλή συσταδοποίηση.
    silhouette_coefficient_hc = metrics.silhouette_score(clustering_data, clusters)

    # Φορτώνουμε τη μεταβλητή data με το νέο .csv αρχείο που περιεχέι τα clusters
    data_c = pd.read_csv('Billionaires Statistics Dataset_clusters.csv')

    # Εμφανίζουμε το scatterplot με τις συστάδες
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data_c, x=chosen_columns[0], y=chosen_columns[1], hue='Cluster', palette='magma', s=100)
    plt.title('Ιεραρχική Συσταδοποίηση')
    plt.xlabel(f'{chosen_columns[0]}(Σε κανονικοποιημένη μορφή)')
    plt.ylabel(f'{chosen_columns[1]}(Σε κανονικοποιημένη μορφή)')
    plt.legend(title='Cluster')
    plt.show()

    # Οι 2 παρακάτω τιμές χρειάζονται στο βήμα σύγκρισης των αλγορίθμων συσταδοποίησης
    return time1, silhouette_coefficient_hc


# -------------------------------------------------------------------------------------------------------
# ------------------------------------------- k-means ---------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def kmeans(filename, chosen_columns, k=3, k_values=[3, 4, 5, 6, 7, 10]):
    data_n = pd.read_csv(filename)
    chosen_data = data_n[chosen_columns].values

    # Μέτρηση του χρόνου εκτέλεσης της μεθόδου k-means
    start_time2 = time.time()

    kmeans = KMeans(n_clusters=k).fit(chosen_data)  # όπου n_clusters:αριθμός συστάδων
    # Η μέθοδος πραγματοποιεί τη συσταδοποίηση με την Ευκλείδεια Απόσταση(η μόνη απόσταση που επιτρέπεται στην Python
    # για k-means), σε αντίθεση με την ιεραρχική συσταδοποίηση που μας δίνεται η δυνατότητα επιλογής.

    time2 = time.time() - start_time2

    IDX = kmeans.labels_  # Η μέθοδος labels_ επιστρέφει μία λίστα με τις ετικέτες των δειγμάτων στις συστάδες,
    # δηλαδή την ανάθεση κάθε δείγματος στην εκάστοτε συστάδα.
    C = kmeans.cluster_centers_  # Η μέθοδος cluster_centers_ περιέχει τις συντεταγμένες των κέντρων των συστάδων που
    # δημιουργήθηκαν κατά τη λειτουργία του αλγόριθμου συσταδοποίησης k-means.

    # Οι 3 εντολές που ακολουθούν χρησιμοποιούνται για παρασταθούν τα δείγματα στο χώρο
    # των χαρακτηριστικών, ώς κυκλάκια(marker='o'). Αξίζει να αναφερθεί ότι τα δείγματα κάθε συστάδας
    # παρουσιάζονται με διαφορετικό χρώμα(π.χ. limegreen για τη 1η συστάδα).
    # Τέλος, η εντολή:chosen_data[IDX == 0][:, 0],chosen_data[IDX == 0][:, 1]
    # σηματοδοτεί τη χρήση αποκλειστικά των στηλών 0 και 1 της 1ης συστάδας(IDX == 0), δηλαδή των
    # δειγμάτων που έχουν το 0 για ετικέτα. Αυτά τα δείγματα έχουν στη γραφική παράσταση την ετικέτα C1(label='C1')
    plt.plot(chosen_data[IDX == 0][:, 0], chosen_data[IDX == 0][:, 1], 'limegreen', marker='o', linewidth=0, label='C1')
    plt.plot(chosen_data[IDX == 1][:, 0], chosen_data[IDX == 1][:, 1], 'yellow', marker='o', linewidth=0, label='C2')
    plt.plot(chosen_data[IDX == 2][:, 0], chosen_data[IDX == 2][:, 1], 'c.', marker='o', label='C3')

    plt.scatter(C[:, 0], C[:, 1], marker='x', color='black', s=150, linewidth=3, label="Centroids", zorder=10)
    # Διάγραμμα διασποράς:Εμφάνιση των κέντρων των συστάδων ως μαύρα Χ στο χώρο των χαρακτηριστικών.
    # Το s αναφέρεται στο μέγεθος των μαύρων Χ, που έχουν και την ετικέτα Centroids.
    # Η παράμετρος zorder=10 χρησιμοποιείται με στόχο την εμφάνιση των κέντρων πάνω από τα δείγματα της κάθε συστάδας,
    # δηλαδή για να είναι ευδιάκριτα.
    plt.xlabel(f'{chosen_columns[0]}(Σε κανονικοποιημένη μορφή)')
    plt.ylabel(f'{chosen_columns[1]}(Σε κανονικοποιημένη μορφή)')
    plt.title('Συσταδοποίηση k-means')
    plt.legend()
    plt.show()

    # SSE: Συνολικό Τετραγωνικό Σφάλμα
    # Υπολογίζει το συνολικό άθροισμα των τετραγώνων των αποστάσεων
    # μεταξύ κάθε δείγματος και του κέντρου της συστάδας στην οποία ανήκει το δείγμα.
    # Όσο μικρότερο είναι το SSE αντιστοιχεί σε καλύτερη συσταδοποίηση.

    # Σε πραγματικές εφαρμογές δε χρησιμοποιείται μόνο ένα μέτρο υπολογισμού της ποιότητας συσταδοποίησης,
    # αλλά συνδυασμός τους και ανάλυση των αποτελεσμάτων του.
    # Υπολογισμός SSE, Silhouette Coefficient για τη συγκεκριμένη συσταδοποίηση(k=3)
    numberOfRows, numberOfColumns = chosen_data.shape
    sse = 0.0

    for i in range(k):
        for j in range(numberOfRows):
            if IDX[j] == i:
                sse = sse + math.dist(chosen_data[j], C[i]) ** 2

    print("\n\nk-means:SSE για k=3 = %.3f" % sse)

    silhouette_coefficient_kmeans = metrics.silhouette_score(chosen_data, IDX)

    # Αρχικοποίηση λιστών που περιέχουν τις τιμές των SSE και Silhouette Coefficient για τις διάφορες τιμές του k
    sse_values = []
    silhouette_coefficient_values = []

    for k in k_values:  # loop για τις διάφορες τιμές του k(αριθμός συστάδων)
        kmeans = KMeans(n_clusters=k).fit(chosen_data)
        IDX = kmeans.labels_
        C = kmeans.cluster_centers_

        # Υπολογισμός SSE
        sse = 0.0
        for i in range(k):
            for j in range(numberOfRows):
                if IDX[j] == i:
                    sse = sse + math.dist(chosen_data[j], C[i]) ** 2
        sse_values.append(sse)  # Εισαγωγή του υπολογιζόμενου sse στη λίστα sse_values

        # Υπολογισμός Silhouette Coefficient
        silhouette_coefficient = metrics.silhouette_score(chosen_data, IDX)
        # Εισαγωγή του υπολογιζόμενου silhouette_coefficient στη λίστα silhouette_coefficient_values
        silhouette_coefficient_values.append(silhouette_coefficient)

    # Γραφική Παράσταση SSE - k
    plt.figure(2)
    plt.plot(k_values, sse_values, marker='o')
    plt.title('k-means: SSE για διαφορετικά k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE')
    plt.show()

    # Γραφική Παράσταση Silhouette Coefficient - k
    plt.figure(3)
    plt.plot(k_values, silhouette_coefficient_values, marker='o')
    plt.title('k-means: Silhouette Coefficient για διαφορετικά k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.show()

    return time2, silhouette_coefficient_kmeans


# -------------------------------------------------------------------------------------------------------
# ------------------------------------------- DBSCAN ----------------------------------------------------
# -------------------------------------------------------------------------------------------------------

# Χρησιμοποιούμε την ιδία μεταβλητή δεδομένων που χρησιμοποιήσαμε και στις άλλες 2 μεθόδους(chosen_data). Για να
# βελτιστοποιήσουμε την απόδοση του dbscan αλγορίθμου θα μπορούσαμε να έχουμε χρησιμοποιήσει το zscore για την
# κανονικοποίηση των αποστάσεων, αλλά απο τη στιγμή που έχουμε ήδη κανονικοποιήσει τα δεδομένα, δε χρειάζεται.
# Με το time.time() μετράμε τον χρόνο εκτέλεσης του dbscan, στο start_time3 ξεκινάει η μέτρηση του χρόνου και στο time3
# υπολογίζεται ο χρόνος εκτέλεσης του αλγορίθμου.

def dbscan_clustering(filename, chosen_columns, eps=0.3, min_samples=15):
    data_n = pd.read_csv(filename)
    chosen_data = data_n[chosen_columns].values

    start_time3 = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(chosen_data)

    # DBSCAN συσταδοποίηση με τη βοήθεια της συνάρτησης fit(όπως και για την k-means)
    # O DBSCAN είναι κατάλληλος για ομάδες που έχουν υψηλή
    # πυκνότητα σημείων, οι οποίες είναι διαχωρισμένες από άλλα σημεία χαμηλότερης
    # πυκνότητας. Με τον όρο πυκνότητα εννοούμε τον αριθμό σημείων(min_samples ή
    # MinPts) σε ακτίνα eps(ή Eps).

    # Διαχωρισμός σημείων σε:
    # 1) Κεντρικά σημεία/Σημεία πυρήνα:Έχουν πυκνότητα μεγαλύτερη από min_samples.
    # 2) Οριακά σημεία:Έχουν πυκνότητα μικρότερη από min_samples,
    #    αλλά απέχουν από ένα κεντρικό σημείο απόσταση μικρότερη από eps.
    # 3) Θόρυβος:Ανήκουν σε περιοχές χαμηλής πυκνότητας,
    #    είναι δηλαδή τα σημεία που δεν ανήκουν σε κάποια από τις 2 παραπάνω κατηγορίες.

    # Σημείωση: Η φιλοσοφία του αλγορίθμου DBSCAN δεν επιτρέπει την εκ των προτέρων επιλογή
    # του αριθμού των συστάδων. Βέβαια, ο αριθμός των συστάδων επηρεάζεται από τις τιμές των
    # παραμέτρων eps και min_samples.

    time3 = time.time() - start_time3

    IDX_dbscan = dbscan.labels_

    # Εμφανίζουμε την dbscan μέθοδο
    plt.figure(figsize=(10, 8))

    # Κανονικά σημεία
    plt.scatter(chosen_data[IDX_dbscan != -1, 0], chosen_data[IDX_dbscan != -1, 1], c=IDX_dbscan[IDX_dbscan != -1],
                cmap='viridis', s=50, label='Clusters')

    # Σημεία θορύβου(σχεδιασμός με μικρότερο μέγεθος:αλλαγή του s από 50 σε 10)
    plt.scatter(chosen_data[IDX_dbscan == -1, 0], chosen_data[IDX_dbscan == -1, 1], c='red', s=10, label='Noise')

    plt.title('Συσταδοποίηση DBSCAN(με κόκκινο χρώμα σχεδιάζονται τα σημεία θορύβου)')
    plt.xlabel(f'{chosen_columns[0]} (Σε κανονικοποιημένη μορφή)')
    plt.ylabel(f'{chosen_columns[1]} (Σε κανονικοποιημένη μορφή)')
    plt.show()

    # Υπολογισμός του silhouette coefficient ώστε να μπορούμε να το συγκρίνουμε
    silhouette_coefficient_dbscan = metrics.silhouette_score(chosen_data, IDX_dbscan)

    return time3, silhouette_coefficient_dbscan


# Συνάρτηση main
# ----------------------------------Σύγκριση αλγορίθμων--------------------------------------------
def main():
    filename = 'Billionaires Statistics Dataset_normalized.csv'

    chosen_columns = ['finalWorth', 'age']

    print("ΠΑΡΑΔΕΙΓΜΑ 1")
    # Κλήση της 1ης συνάρτησης
    time1, silhouette_coefficient_hc = hierarchical_clustering(filename, chosen_columns)

    # Κλήση της 2ης συνάρτησης
    time2, silhouette_coefficient_kmeans = kmeans(filename, chosen_columns)

    # Κλήση της 3ης συνάρτησης
    time3, silhouette_coefficient_dbscan = dbscan_clustering(filename, chosen_columns)

    print("\n\n-Silhouette Coefficient-")
    print("Ιεραρχική Συσταδοποίηση:Silhouette Coefficient για k=3: %0.3f" % silhouette_coefficient_hc)
    print("k-means:Silhouette Coefficient για k=3: %0.3f" % silhouette_coefficient_kmeans)
    print("DBSCAN:Silhouette Coefficient: %0.3f" % silhouette_coefficient_dbscan)

    print("\n\n-Χρόνος εκτέλεσης αλγορίθμου-")
    print(f"Χρόνος εκτέλεσης αλγορίθμου ιεραρχικής συσταδοποίησης για αριθμό συστάδων=3: {time1:.4f} seconds")
    print(f"Χρόνος εκτέλεσης αλγορίθμου k-means για k=3: {time2:.4f} seconds")
    print(f"Χρόνος εκτέλεσης αλγορίθμου DBSCAN: {time3:.4f} seconds")

    print("------------------------------------------------------------------")
    # 2ο παράδειγμα: επιλογή άλλων στηλών του συνόλου δεδομένων

    chosen_columns2 = ['finalWorth', 'total_tax_rate_country']
    print("ΠΑΡΑΔΕΙΓΜΑ 2")

    # Κλήση της 1ης συνάρτησης
    time1_2, silhouette_coefficient_hc1_2 = hierarchical_clustering(filename, chosen_columns2)

    # Κλήση της 2ης συνάρτησης
    time2_2, silhouette_coefficient_kmeans2_2 = kmeans(filename, chosen_columns2)

    # Κλήση της 3ης συνάρτησης
    time3_2, silhouette_coefficient_dbscan3_2 = dbscan_clustering(filename, chosen_columns2)

    print("\n\n-Silhouette Coefficient-")
    print("Ιεραρχική Συσταδοποίηση:Silhouette Coefficient για k=3: %0.3f" % silhouette_coefficient_hc1_2)
    print("k-means:Silhouette Coefficient για k=3: %0.3f" % silhouette_coefficient_kmeans2_2)
    print("DBSCAN:Silhouette Coefficient: %0.3f" % silhouette_coefficient_dbscan3_2)

    print("\n\n-Χρόνος εκτέλεσης αλγορίθμου-")
    print(f"Χρόνος εκτέλεσης αλγορίθμου ιεραρχικής συσταδοποίησης για αριθμό συστάδων=3: {time1_2:.4f} seconds")
    print(f"Χρόνος εκτέλεσης αλγορίθμου k-means για k=3: {time2_2:.4f} seconds")
    print(f"Χρόνος εκτέλεσης αλγορίθμου DBSCAN: {time3_2:.4f} seconds")


if __name__ == "__main__":
    main()
