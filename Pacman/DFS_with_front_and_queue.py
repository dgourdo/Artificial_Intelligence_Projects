import copy
import sys
import pprint  # Support to pretty-print lists, tuples, & dictionaries recursively.
import random

sys.setrecursionlimit(10 ** 6)

in_state = [[[' ' for elem in range(2)] for col in range(4)] for row in range(4)]

"""
 Κενή 2d λίστα 4x4,
 όπου κάθε θέση της αποτελείται από λίστα 2 θέσεων
 (2 κελιά -> Στο αριστερό υπάρχει ο pacman, και στο δεξί τα φρούτα)
"""


def can_eat(state):
    for i in range(len(state)):  # διατρέχουμε τη λίστα
        for j in range(len(state)):
            if state[i][j][0] == 'p' and state[i][j][1] == 'f':  # ['p' , 'f']
                print("Το pacman μπορεί να φάει το φρούτο!")
                return 1  # επιτυχία


def can_move_right(state):
    for row in range(len(state)):
        for col in range(len(state)):
            if state[row][len(state) - 1][0] == 'p':  # Απαγορεύεται η κίνηση προς τα δεξιά όταν:
                print("Το pacman δε μπορεί κινηθεί δεξιά!")  # α) Ο pacman βρίσκεται στην τελευταία(δεξιότερη) στήλη.
                return None  # β) Υπάρχει τοίχος('w' , wall) ακριβώς μία στήλη
            elif state[row][col][0] == 'p' and state[row][col + 1][0] == 'w':  # δεξιότερα από το pacman.
                print("Τοίχος.Το pacman δε μπορεί κινηθεί δεξιά!")
                return None
    return 1


"""
  Απαγορεύεται η κίνηση προς τα αριστερά όταν:
  α) Ο pacman βρίσκεται στην πρώτη(αριστερότερη, [0]) στήλη.
  β) Υπάρχει τοίχος('w' , wall) ακριβώς μία στήλη
  αριστερότερα από το pacman.
"""


def can_move_left(state):
    for row in range(len(state)):
        for col in range(len(state)):
            if state[row][0][0] == 'p':
                print("Το pacman δε μπορεί κινηθεί αριστερά!")
                return None
            elif state[row][col][0] == 'p' and state[row][col - 1][0] == 'w':
                print("Τοίχος.Το pacman δε μπορεί κινηθεί αριστερά!")
                return None
    return 1


"""
 # Απαγορεύεται η κίνηση προς τα πάνω όταν:
 # α) Ο pacman βρίσκεται στην πρώτη([0]) γραμμή.
 # β) Υπάρχει τοίχος('w' , wall) ακριβώς μία γραμμή
 # πιο πάνω από το pacman.
"""


def can_move_up(state):
    for row in range(len(state)):
        for col in range(len(state)):
            if state[0][col][0] == 'p':
                print("Το pacman δε μπορεί κινηθεί προς τα πάνω!")
                return None
            elif state[row][col][0] == 'p' and state[row - 1][col][0] == 'w':
                print("Τοίχος.Το pacman δε μπορεί κινηθεί προς τα πάνω!")
                return None
    return 1


"""
  Απαγορεύεται η κίνηση προς τα κάτω όταν:
  α) Ο pacman βρίσκεται στην τελευταία γραμμή.
  β) Υπάρχει τοίχος('w' , wall) ακριβώς μία γραμμή
  πιο κάτω από το pacman.
"""


def can_move_down(state):
    for row in range(len(state)):
        for col in range(len(state)):
            if state[len(state) - 1][col][0] == 'p':
                print("Το pacman δε μπορεί κινηθεί προς τα κάτω!")
                return None
            elif state[row][col][0] == 'p' and state[row + 1][col][0] == 'w':
                print("Τοίχος.Το pacman δε μπορεί κινηθεί προς τα κάτω!")
                return None
    return 1


def eat(state):
    if can_eat(state):
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j][0] == 'p' and state[i][j][1] == 'f':
                    state[i][j][1] = ''  # ['p' , 'f'] -> ['p' , '']
                    return state  # επιτυχία
    else:
        return None  # αποτυχία


def move_right(state):
    if can_move_right(state):
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j][0] == 'p':
                    state[i][j][0] = ''  # το κελί αδειάζει με τη φυγή του pacman
                    state[i][j + 1][0] = 'p'  # ο pacman μεταφέρεται στο αμέσως δεξιότερο κελί
                    return state
    else:
        return None


def move_left(state):
    if can_move_left(state):
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j][0] == 'p':
                    state[i][j][0] = ''  # το κελί αδειάζει με τη φυγή του pacman
                    state[i][j - 1][0] = 'p'  # ο pacman μεταφέρεται στο αμέσως αριστερότερο κελί
                    return state
    else:
        return None


def move_up(state):
    if can_move_up(state):
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j][0] == 'p':
                    state[i][j][0] = ''  # το κελί αδειάζει με τη φυγή του pacman
                    state[i - 1][j][0] = 'p'  # ο pacman μεταφέρεται στο αμέσως υψηλότερο κελί
                    return state
    else:
        return None


def move_down(state):
    if can_move_down(state):
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j][0] == 'p':
                    state[i][j][0] = ''  # το κελί αδειάζει με τη φυγή του pacman
                    state[i + 1][j][0] = 'p'  # ο pacman μεταφέρεται στο αμέσως χαμηλότερο κελί
                    return state
    else:
        return None


def is_goal_state(state):  # goal_state: όλα τα φρούτα έχουν <<φαγωθεί>>
    for i in range(len(state)):  # 2ος Τρόπος: if fruit_count == 0
        for j in range(len(state)):
            if state[i][j][1] == 'f':
                print("|Not a goal state|")
                return 0
    print("|A goal state|")
    return 1


def find_children(state):
    children = []
    right_state = copy.deepcopy(state)  # Στην περίπτωση του deepcopy, ένα αντίγραφο του αντικειμένου
    child_right = move_right(right_state)  # αντιγράφεται σε άλλο αντικείμενο.
    left_state = copy.deepcopy(state)  # Αυτό σημαίνει ότι τυχόν αλλαγές που έγιναν σε ένα αντίγραφο του
    child_left = move_left(left_state)  # αντικειμένου δεν αντικατοπτρίζονται στο αρχικό αντικείμενο.
    up_state = copy.deepcopy(state)
    child_up = move_up(up_state)
    down_state = copy.deepcopy(state)
    child_down = move_down(down_state)
    eat_state = copy.deepcopy(state)
    child_eat = eat(eat_state)

    if child_right is not None:  # 2ος τρόπος: if not child_right == None:
        children.append(child_right)

    """
        Εάν η κίνηση προς τα δεξιά είναι δυνατή και πραγματοποιηθεί, τότε
        μέσω της συνάρτησης move_right επιστρέφεται η νέα μορφή του state.
        Η προαναφερθείσα νέα μορφή του state τοποθετείται στο τέλος της λίστας
        children, η οποία αποτελεί κλειδί για τον αλγόριθμο.
        Η ίδια διαδικασία θα πραγματοποιηθεί(εξέταση των υπόλοιπων 4 αντιγράφων του state)
        για τις move_left,move_up,move_down,eat.
    """

    if child_left is not None:
        children.append(child_left)

    if child_up is not None:
        children.append(child_up)

    if child_down is not None:
        children.append(child_down)

    if child_eat is not None:
        children.append(child_eat)

    return children


def make_front(state):  # Αρχικοποίηση μετώπου
    return [state]


"""
     Στη μεταβλητή node αποθηκεύουμε το 1ο στοιχείο του μετώπου.
     Έπειτα, με τη βοήθεια της find_children και τη διάσχιση της λίστας children
     επεκτείνουμε το μέτωπο και τοποθετούμε στην αρχή(DFS) του νέου μετώπου
     τα παραγόμενα παιδιά.
"""


def expand_front(front, method):  # Επέκταση μετώπου
    if method == 'DFS':
        if front:
            print("Front:")
            pprint.pprint(front)
            print(" ")
            node = front.pop(0)
            for child in find_children(node):
                front.insert(0, child)
    return front


def make_queue(state):  # Αρχικοποίηση ουράς
    return [[state]]


def extend_queue(queue, method):  # Επέκταση ουράς
    if method == 'DFS':
        print("Queue:")
        pprint.pprint(queue)  # Support to pretty-print lists, tuples, & dictionaries recursively.
        print(" ")
        node = queue.pop(0)  # αφαιρούμε το 1ο στοιχείο της queue
        queue_copy = copy.deepcopy(queue)
        children = find_children(node[-1])  # εκχωρούμε τον τελευταίο(κάτω-κάτω) κόμβο ως όρισμα στη find_children
        for child in children:
            path = copy.deepcopy(node)
            path.append(child)  # προσαρτούμε κάθε παιδί που δημιουργείται στο τέλος της διαδρομής
            queue_copy.insert(0, path)
            # Ουρά:για κάθε κόμβο του front η διαδρομή από τη ρίζα
    return queue_copy


"""
 find_solution:
    4 περιπτώσεις:
    1) Αποτυχία(Δεν υπάρχει μέτωπο αναζήτησης)
    2) Αφαιρούμε τη γονική κατάσταση από το μέτωπο και την ουρά.
       Δημιουργία και εξέταση των νέων αντιγράφων για μέτωπο και ουρά αναζήτησης αντίστοιχα.
       Αναδρομική κλήση της find_solution, με τα νέα(αναφερόμενα παραπάνω) ορίσματα.
    3) Επιτυχία(Μηδενικός αριθμός εναπομείναντων φρούτων)
       Τυπώνεται το μονοπάτι από τη ρίζα έως και την κατάσταση-στόχο, και
       το κλειστό σύνολο.
    4) Καμία από τις 3 παραπάνω καταστάσεις.
       Τοποθέτηση της κατάστασης-γονέα του μετώπου αναζήτησης στο κλειστό σύνολο.
       Δημιουργία αντιγράφων για μέτωπο,ουρά αναζήτησης και κλειστό σύνολο
       και επεκτείνουμε τα πρώτα δύο με τις συναρτήσεις expand_front και extend_queue.
       Ακολουθεί αναδρομική κλήση της find_solution.
"""


def find_solution(front, queue, closed, method):
    if not front:
        print('_NO_SOLUTION_FOUND_')

    elif front[0] in closed:
        new_front = copy.deepcopy(front)
        new_front.pop(0)
        new_queue = copy.deepcopy(queue)
        new_queue.pop(0)
        find_solution(new_front, new_queue, closed, method)

    elif is_goal_state(front[0]):
        print('_GOAL_FOUND_')
        pprint.pprint(queue[0])
        print(" ")
        print("Closed:")
        pprint.pprint(closed)

    else:
        closed.append(front[0])
        front_copy = copy.deepcopy(front)
        front_children = expand_front(front_copy, method)
        queue_copy = copy.deepcopy(queue)
        queue_children = extend_queue(queue_copy, method)
        closed_copy = copy.deepcopy(closed)
        find_solution(front_children, queue_children, closed_copy, method)


# Δημιουργούμε τη συνάρτηση print_initial_state με a, b ώστε να μπορεί να μεταβληθεί
# τόσο ο αριθμός των γραμμών(a), όσο και των στηλών της λίστας(b) αντίστοιχα.


def print_initial_state(a, b, c=2):  # function with a default parameter(c)
    fruit_count = random.randrange(c - 1, a + c)  # Using randrange() to generate numbers from 1-5(παρακάτω in_row = 4)
    i = 0

    wr1 = random.randrange(a)  # Using randrange() to generate numbers from 0-3(παρακάτω in_row = 4)
    wr2 = random.randrange(b)
    in_state[wr1][wr2][c - 2] = 'w'
    in_state[wr1][wr2][c - 1] = 'w'  # wall -> pacman cannot approach this position of the array
    # check: do not put 'p' or 'f' in a wall('p') position

    pr1 = random.randrange(a)  # pacman-random-position1
    pr2 = random.randrange(b)  # pacman-random-position2

    fr1 = random.randrange(a)  # χρησιμοποιούμε μεταβλητές για να εξάγουμε ελέγχους με if,
    fr2 = random.randrange(b)  # ώστε να ισχύσουν οι περιορισμοί της αρχικής κατάστασης του προβλήματος

    if in_state[pr1][pr2][c - 2] != 'w':  # έλεγχος για ύπαρξη τοίχου
        in_state[pr1][pr2][c - 2] = 'p'  # τοποθέτηση pacman,pacman is in a random position

        while i < fruit_count:
            if in_state[fr1][fr2][c - 1] != 'w':  # έλεγχος για ύπαρξη τοίχου
                in_state[fr1][fr2][c - 1] = 'f'  # τοποθέτηση fruit
                fr1 = random.randrange(a)
                fr2 = random.randrange(b)
                i = i + 1
    return in_state


in_row = 4
in_col1 = 4

pprint.pprint(print_initial_state(in_row, in_col1))
print("")


def main():
    initial_state = in_state
    method = 'DFS'

    print('------------')
    print(f'|    {method}   |') # f-string
    print('------------')
    print('____BEGIN__SEARCHING____')
    find_solution(make_front(initial_state), make_queue(initial_state), [], method)


if __name__ == "__main__":
    main()
