import math
import requests
from bs4 import BeautifulSoup
import re
import nltk
import json
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from tkinter import *

nltk.download('stopwords')


def preprocess_text(text):
    # Αφαίρεση ειδικών χαρακτήρων
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenization
    tokens = text.split()

    # Κανονικοποίηση κειμένου
    tokens = [word.lower() for word in tokens]

    # Stemming -> εύρεση του κυρίως στελέχους της λέξης
    porter = nltk.PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]

    # Αφαίρεση σημείων στίξης και προθημάτων (stop words)
    # print(string.punctuation)  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    cleaned_tokens = []
    for token in tokens:
        if token not in string.punctuation:  # not !!!
            cleaned_tokens.append(token)

    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')
    # print(stopwords)

    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords:  # not !!!
            cleaned_tokens.append(token)

    tokens.extend(cleaned_tokens)

    # Ενσωμάτωση των λέξεων σε ένα string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # 3ο όρισμα:το αρχείο γίνεται ευανάγνωστο


def crawler(url_param):
    response = requests.get(url_param)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('div', class_='meta')
    info = []

    for result in results:
        title = result.find('div', class_='list-title').get_text(strip=True)
        authors = [author.get_text(strip=True) for author in result.find('div', class_='list-authors')]
        abstract = result.find('p', class_='mathjax').text.strip() if result.find('p',
                                                                                  class_='mathjax') else "Not Found"
        date = result.find_all('div', class_='list-dateline').text.strip() if result.find('div',
                                                                                          class_='list-dateline') else "Not Found"

        results_info = {
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'date': date
        }
        info.append(results_info)

    return info


def create_inverted_index(data):
    inverted_index = {}

    for i, entry in enumerate(data, 1):
        abstract = entry.get('abstract',
                             '')  # Συλλογή των εγγράφων για ευρετηρίαση(τιμή που αντιστοιχεί στο key abstract)
        terms = set(preprocess_text(abstract).split())  # Μετατροπή κάθε εγγράφου σε λίστα στοιχείων
        # enumerate:επιστρέφει ένα tuple με 2 στοιχεία
        # όπου i:αριθμός επανάληψης(αρχίζει από 1) και entry:το στοιχείο της λίστας

        for term in terms:
            # Έλεγχος για αριθμούς και γράμματα
            if term.isalpha() and len(term) > 1:
                if term not in inverted_index:
                    inverted_index[term] = {i}  # !!!
                else:
                    inverted_index[term].add(i)  # !!!

    # Ταξινόμηση των όρων αλφαβητικά
    sorted_inverted_index = dict(sorted(inverted_index.items()))

    return sorted_inverted_index


def engine(inverted_index, documents, user_query, processed_query, choice, date_filter, author_filter, date_value, combined_results=None):
    search_results = []

    # Ο χρήστης επιλέγει τον αλγόριθμο ανάκτησης μέσω του gui
    if combined_results is None:
        if choice == 1:
            search_results = boolean_retrieval(inverted_index, processed_query)
        elif choice == 2:
            search_results = vector_space_model(processed_query, inverted_index, documents)
        elif choice == 3:
            search_results = okapi_bm25(processed_query, inverted_index, documents)
    else:
        search_results = combined_results

    # Ανάκτηση εγγραφών με χρήση του ανεστραμμένου ευρετηρίου
    for term in user_query:
        if term in inverted_index:
            search_results.extend(inverted_index[term])

    # Αφαίρεση διπλότυπων εγγραφών
    search_results = list(set(search_results))

    # Κατάταξη αποτελεσμάτων
    ranked_results = rank_documents(processed_query, search_results, documents)

    # Εφαρμογή φίλτρων
    filtered_results = filter_results(ranked_results)

    # Εκτύπωση των τελικών αποτελεσμάτων
    print_results(filtered_results)


# ------------------------------------------------------------
# Βοηθητικές συναρτήσεις
def toggle_date_entry(date_var, date_entry):
    # Ενεργοποίηση ή απενεργοποίηση του date_entry ανάλογα με την κατάσταση του date_var
    # Δηλαδή αν δεν τσεκαριστεί το checkButton 'Ημερομηνία δημοσίευσης', δε γίνεται να γράψει ο χρήστης στο 1ο entry.
    date_entry['state'] = NORMAL if date_var.get() == 1 else DISABLED


def toggle_author_entry(author_var, author_entry):
    # Ενεργοποίηση ή απενεργοποίηση του author_entry ανάλογα με την κατάσταση του author_var
    # Δηλαδή αν δεν τσεκαριστεί το checkButton 'Συγγραφέας', δε γίνεται να γράψει ο χρήστης στο 2ο entry.
    author_entry['state'] = NORMAL if author_var.get() == 1 else DISABLED


def preprocess_query(query):
    # Tokenization
    query_tokens = query.split()

    # Κανονικοποίηση ερωτήματος
    query_tokens = [word.lower() for word in query_tokens]

    return query_tokens


def intersection(p1, p2):
    if p1 is not None and p2 is not None:
        intersection = list(set(p1) & set(p2))
        return intersection
    else:
        return []


def union(p1, p2):
    if p1 is not None and p2 is not None:
        return list(set().union(p1, p2))
    else:
        return []


def NOT(p1, p2):
    if p1 is not None and p2 is not None:
        return list(set(p1) - set(p2))
    else:
        return []


# ------------------------------------------------------------

def user_interface(inverted_index, documents):
    root = Tk()
    root.title("Διεπαφή χρήστη για την αναζήτηση ακαδημαϊκών εργασιών")
    root.geometry("600x400")

    label = Label(root, text="Ερώτημα χρήστη")
    label.pack()

    query_entry = Entry(root, width=50, bg="lightgray")
    query_entry.pack()

    Label(root, text="").pack()

    label2 = Label(root, text="Επιλογή αλγόριθμου ανάκτησης")
    label2.pack()

    user_choice = IntVar()
    Radiobutton(root, text='Boolean retrieval', variable=user_choice, value=1).pack(anchor=W)
    Radiobutton(root, text='Vector Space Model(VSM)', variable=user_choice, value=2).pack(anchor=W)
    Radiobutton(root, text='Probabilistic retrieval model(Okapi BM25)', variable=user_choice, value=3).pack(anchor=W)

    Label(root, text="").pack()

    label3 = Label(root, text="Φίλτρα αναζήτησης")
    label3.pack()

    date_var = IntVar()
    # χρήση της βοηθητικής συνάρτησης toggle_date_entry
    date_checkbutton = Checkbutton(root, text='Ημερομηνία δημοσίευσης', variable=date_var,
                                   command=lambda: toggle_date_entry(date_var, date_entry))
    date_checkbutton.pack(anchor=W)

    date_entry = Entry(root, width=50, bg="lightgray", state=DISABLED)  # αρχικά DISABLED το γράψιμο στο 1ο entry
    date_entry.pack()

    author_var = IntVar()
    # χρήση της βοηθητικής συνάρτησης toggle_author_entry
    author_checkbutton = Checkbutton(root, text='Συγγραφέας', variable=author_var,
                                     command=lambda: toggle_author_entry(author_var, author_entry))
    author_checkbutton.pack(anchor=W)

    author_entry = Entry(root, width=50, bg="lightgray", state=DISABLED)  # αρχικά DISABLED το γράψιμο στο 2ο entry
    author_entry.pack()

    # Εσωτερική συνάρτηση
    def search_and_display_results(query_entry, user_choice, date_var, author_var, date_entry, author_entry,
                                   inverted_index, documents):
        user_query = query_entry.get()

        # Βήμα 4 - Επεξεργασία Ερωτήματος
        query_parts = re.split(r'\b(AND|OR|NOT)\b', user_query)
        processed_queries = [preprocess_text(part.strip()) for part in query_parts]

        date_filter = date_var.get() == 1

        author_filter = None
        if author_var.get() == 1:
            author_filter = author_entry.get()

        date_value = None
        if date_filter:
            date_value = date_entry.get()

        if len(processed_queries) > 1:
            # Ερωτήματα με περισσότερες από μία λέξεις
            boolean_results = []

            for proc_query in processed_queries:
                search_results = boolean_retrieval(inverted_index, proc_query)
                boolean_results.append(search_results)

            # Εφαρμογή των πράξεων AND/OR/NOT
            combined_results = boolean_results[0]

            for i in range(1, len(boolean_results), 2):
                if i + 1 < len(boolean_results):  # χρήση των βοηθητικών συναρτήσεων
                    if query_parts[i] == "AND":
                        combined_results = intersection(combined_results, boolean_results[i + 1])
                    elif query_parts[i] == "OR":
                        combined_results = union(combined_results, boolean_results[i + 1])
                    elif query_parts[i] == "NOT":
                        combined_results = NOT(combined_results, boolean_results[i + 1])
            # Κλήση της engine
            engine(inverted_index, documents, user_query, processed_queries[0], user_choice.get(), date_filter,
                   author_filter, date_value, combined_results)
        else:
            # Κλήση της engine(Περίπτωση ερωτήματος μιας λέξης)
            engine(inverted_index, documents, user_query, processed_queries[0], user_choice.get(), date_filter,
                   author_filter, date_value)

    # Κλήση της εσωτερικής συνάρτησης με το πάτημα του κουμπιού 'Αναζήτηση'
    search = Button(root, text="Αναζήτηση", command=lambda: search_and_display_results, bg="black", fg="white")
    search.pack()

    result = Label(root, text="")
    result.pack()

    root.mainloop()


def rank_documents(query, results, documents):
    # Προεπεξεργασία του ερωτήματος
    if query:
        processed_query = preprocess_text(query)
    else:
        processed_query = ""

    # συγχώνευση εγγράφων σε ένα corpus(συλλογή)
    corpus = [doc['abstract'] for doc in documents if doc['abstract']]

    # Προεπεξεργασία
    preprocessed_corpus = [preprocess_text(doc) for doc in corpus]

    # δημιουργία αντικειμένου
    tfidf = TfidfVectorizer()

    # tf-df values
    result = tfidf.fit_transform(preprocessed_corpus)

    # Υπολογισμός TF-IDF για το ερώτημα
    query_tfidf = tfidf.transform([processed_query])

    # get indexing
    print('\nWord indexes:')
    print(tfidf.vocabulary_)

    print('\ntf-idf values:')
    print(result)

    # Υπολογισμός cosine similarity μεταξύ του ερωτήματος και των εγγράφων
    similarities = result.dot(query_tfidf.T).toarray().flatten()  # χρήση του result
    # 2ος τρόπος: cosine = np.dot(A,B)/(norm(A)*norm(B)))

    # Ταξινόμηση των εγγράφων με βάση την ομοιότητα
    ranked_results = [doc for _, doc in sorted(zip(similarities, results), reverse=True)]

    return ranked_results


def boolean_retrieval(inverted_index, query_tokens):
    results = inverted_index.get(query_tokens[0], set())

    i = 1
    while i < len(query_tokens):
        if query_tokens[i] == "AND":
            results = intersection(results, inverted_index.get(query_tokens[i + 1], set()))
        elif query_tokens[i] == "OR":
            results = union(results, inverted_index.get(query_tokens[i + 1], set()))
        elif query_tokens[i] == "NOT":
            results = NOT(results, inverted_index.get(query_tokens[i + 1], set()))
        i += 2

    return list(results)


def vector_space_model(processed_query, inverted_index, documents):
    corpus = [doc['abstract'] for doc in documents if doc['abstract']]

    # Προεπεξεργασία
    preprocessed_corpus = [preprocess_text(doc) for doc in corpus]

    # Υλοποίηση του TF-IDF, βασικού αλγορίθμου κατάταξης
    vectorizer = TfidfVectorizer()

    if processed_query is not None:
        X = vectorizer.fit_transform(preprocessed_corpus)

        # Υπολογισμός TF-IDF για το ερώτημα
        query_tfidf = vectorizer.transform([processed_query])

        # Υπολογισμός cosine similarity μεταξύ του ερωτήματος και των εγγράφων
        similarities = X.dot(query_tfidf.T).toarray().flatten()

        # Ταξινόμηση των εγγράφων με βάση την ομοιότητα
        ranked_results = [doc for _, doc in sorted(zip(similarities, documents), reverse=True)]

        return ranked_results
    else:
        # Επιστροφή κενού αποτελέσματος αν η processed_query είναι None
        return []


def okapi_bm25(processed_query, inverted_index, documents):
    k1 = 1.5
    b = 0.75

    # Υπολογισμός αθροίσματος μήκους κειμένου
    avgdl = sum(len(doc['abstract'].split()) for doc in documents) / len(documents)

    # Υπολογισμός BM25 για κάθε έγγραφο
    scores = []
    for doc in documents:
        doc_len = len(doc['abstract'].split())
        score = 0

        for term in processed_query.split():
            if term in inverted_index:
                df = len(inverted_index[term])
                idf = math.log((len(documents) - df + 0.5) / (df + 0.5) + 1.0)
                tf = doc['abstract'].split().count(term)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
                score += idf * numerator / denominator

        scores.append(score)

    # Ταξινόμηση των εγγράφων με βάση το σκορ
    ranked_results = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]

    return ranked_results


def filter_results(results, date_filter, author_filter):
    # Υλοποίηση φίλτρων
    filtered_results = results.copy()

    # Φίλτρο 1: Ημερομηνία δημοσίευσης
    if date_filter:
        filtered_results = [result for result in filtered_results if result['date'] != "Not Found"]

    # Φίλτρο 2: Συγγραφέας
    if author_filter is not None:
        filtered_results = [result for result in filtered_results if
                            author_filter.lower() in [author.lower() for author in result['authors']]]

    return filtered_results


def print_results(results):
    # Εκτύπωση των αποτελεσμάτων σε φιλική προς το χρήστη μορφή

    for i, result in enumerate(results, 1):
        print(f"{i}. Title: {result['title']}")
        print(f"   Authors: {', '.join(result['authors'])}")
        print(f"   Date: {result['date']}")
        print(f"   Abstract: {result['abstract']}")
        print("---------------------------------------------------\n")



if __name__ == '__main__':
    url = "https://arxiv.org/list/cs/new"
    results = crawler(url)

    # Αποθήκευση αρχείου JSON πριν από το preprocess
    save_to_json(results, 'arXiv_results_raw.json')

    # Αριθμός εγγράφων:150
    doc_num = results[:150]

    # Προεπεξεργασία του κειμενικού περιεχομένου (της abstract δηλαδή)
    for i, res in enumerate(doc_num, 1):
        preprocessed_abstract = preprocess_text(res['abstract'])
        res['abstract'] = preprocessed_abstract

    # Αποθήκευση αρχείου JSON μετά το preprocess
    save_to_json(doc_num, 'arXiv_results_preprocessed.json')

    # Δημιουργία του ανεστραμμένου ευρετηρίου
    inverted_index = create_inverted_index(doc_num)

    # Μετατροπή σε λίστες
    for term, documents in inverted_index.items():
        inverted_index[term] = list(documents)  # Παραγωγή λίστας κανονικοποιημένων συστατικών και Ευρετηρίαση των
        # εγγράφων που περιέχουν τους όρους

    # Αποθήκευση του ανεστραμμένου ευρετηρίου σε αρχείο JSON
    save_to_json(inverted_index, 'inverted_index.json')

    # Βήμα 4 (η συνάρτηση user_interface καλεί τις υπόλοιπες συναρτήσεις)
    user_interface(inverted_index, doc_num)


#%%
