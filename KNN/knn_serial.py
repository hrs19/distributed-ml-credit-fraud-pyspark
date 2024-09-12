import csv
import random
import math
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(weights='distance', metric='minkowski', p=2)

param_grid = {
    'n_neighbors': [3, 5, 7],
    'lambda': [250,300,400,900],
    'mu': [250,300,400, 900]
}

grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5)

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:
                dataset.append([float(value) for value in row])
    return dataset

def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return train_set, test_set

def euclidean_distance(instance1, instance2):
    distance = 0
    for i in range(len(instance1)-1):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)

def get_neighbors(train_set, test_instance, k):
    distances = []
    for train_instance in train_set:
        distance = euclidean_distance(train_instance, test_instance)
        distances.append((train_instance, distance))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

def get_majority_class(neighbors):
    class_count = {}
    for neighbor in neighbors:
        class_value = neighbor[-1]
        if class_value in class_count:
            class_count[class_value] += 1
        else:
            class_count[class_value] = 1
    sorted_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_count[0][0]

def knn(train_set, test_set, k):
    predictions = []
    for test_instance in test_set:
        neighbors = get_neighbors(train_set, test_instance, k)
        majority_class = get_majority_class(neighbors)
        predictions.append(majority_class)
    return predictions

def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return (correct / float(len(actual))) * 100.0

dataset = load_dataset('creditcard.csv')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1, random_state=42)

k = 5
predicted = knn(train_set, test_set, k)

actual = [row[-1] for row in test_set]

acc = accuracy(actual, predicted)
print('Accuracy:', acc)

precision = precision_score(actual, predicted)

recall = recall_score(actual, predicted)

f1 = f1_score(actual, predicted)

grid_search.fit(X_train, y_train)

print("Best hyperparameters: ", grid_search.best_params_)
print("Validation score: ", grid_search.best_score_)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)



