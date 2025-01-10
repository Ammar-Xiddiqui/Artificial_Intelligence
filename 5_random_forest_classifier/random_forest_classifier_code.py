import pandas as pd
import math
from collections import Counter
from sklearn.utils import resample

# Sample dataset
data_rf = pd.DataFrame({
    'Weather': ['Sunny', 'Overcast', 'Rainy', 'Sunny', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Mild', 'Mild', 'Mild', 'Hot'],
    'Play?': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No']
})

# Calculate entropy
def calculate_entropy(data, target_col):
    counts = Counter(data[target_col])
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

# Calculate information gain
def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    values = data[attribute].unique()
    weighted_entropy = sum((len(data[data[attribute] == value]) / len(data)) * 
                           calculate_entropy(data[data[attribute] == value], target_col) for value in values)
    return total_entropy - weighted_entropy

# Build decision tree
def build_tree(data, attributes, target_col, depth=0, max_depth=3):
    if depth == max_depth or len(attributes) == 0:
        return Counter(data[target_col]).most_common(1)[0][0]
    best_attr = max(attributes, key=lambda attr: calculate_information_gain(data, attr, target_col))
    tree = {best_attr: {}}
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        tree[best_attr][value] = build_tree(subset, [a for a in attributes if a != best_attr], target_col, depth + 1)
    return tree

# Predict using a decision tree
def predict(tree, data_point):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    value = data_point[attribute]
    subtree = tree[attribute].get(value, None)
    if subtree is None:
        return None
    return predict(subtree, data_point)

# Build Random Forest
def build_random_forest(data, attributes, target_col, n_trees=2):
    trees = []
    for _ in range(n_trees):
        subset = resample(data)
        tree = build_tree(subset, attributes, target_col)
        trees.append(tree)
    return trees

# Predict using Random Forest
def predict_forest(forest, data_point):
    predictions = [predict(tree, data_point) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]

# Example: Building and testing Random Forest
attributes_rf = ['Weather', 'Temperature']
forest = build_random_forest(data_rf, attributes_rf, 'Play?', n_trees=2)
new_data_point_rf = {'Weather': 'Sunny', 'Temperature': 'Mild'}
prediction_rf = predict_forest(forest, new_data_point_rf)
print("Random Forest Prediction:", prediction_rf)
