import math
from collections import Counter

# Calculate entropy of a dataset
def calculate_entropy(data, target_col):
    target_values = data[target_col]
    total_count = len(target_values)
    class_counts = Counter(target_values)
    entropy = 0

    for count in class_counts.values():
        probability = count / total_count
        entropy -= probability * math.log2(probability)
    
    return entropy

# Calculate information gain
def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    total_count = len(data)

    # Group data by the attribute's values
    subsets = data.groupby(attribute)
    weighted_entropy = 0

    for _, subset in subsets:
        subset_count = len(subset)
        weighted_entropy += (subset_count / total_count) * calculate_entropy(subset, target_col)
    
    return total_entropy - weighted_entropy

# Build the decision tree
def build_tree(data, attributes, target_col):
    # Base case: All instances have the same class
    if len(set(data[target_col])) == 1:
        return data[target_col].iloc[0]

    # Base case: No attributes left to split
    if not attributes:
        return data[target_col].mode()[0]

    # Select the best attribute
    gains = {attr: calculate_information_gain(data, attr, target_col) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    # Create tree node
    tree = {best_attr: {}}

    # Recur for each value of the best attribute
    for value, subset in data.groupby(best_attr):
        subtree = build_tree(subset, [attr for attr in attributes if attr != best_attr], target_col)
        tree[best_attr][value] = subtree
    
    return tree

# Predict the class for a given data point
def predict(tree, data_point):
    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))
    value = data_point[attribute]
    if value in tree[attribute]:
        return predict(tree[attribute][value], data_point)
    else:
        return None  # Value not seen during training
