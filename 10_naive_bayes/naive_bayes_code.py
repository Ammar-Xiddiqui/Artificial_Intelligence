import string
from collections import defaultdict
import math

# Step 1: Preprocess text
def preprocess_text(text):
    """
    Convert text to lowercase, remove punctuation, and split into words.
    """
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.split()  # Split into words

# Step 2: Build vocabulary
def build_vocabulary(dataset):
    """
    Create a list of unique words from the dataset.
    """
    vocabulary = set()
    for text, _ in dataset:
        words = preprocess_text(text)
        vocabulary.update(words)
    return list(vocabulary)

# Step 3: Calculate probabilities
def calculate_probabilities(dataset, vocabulary):
    """
    Compute prior probabilities and likelihoods for words.
    """
    word_counts = {0: defaultdict(int), 1: defaultdict(int)}
    class_counts = {0: 0, 1: 0}
    
    for text, label in dataset:
        class_counts[label] += 1
        words = preprocess_text(text)
        for word in words:
            word_counts[label][word] += 1
    
    # Calculate prior probabilities
    total_samples = len(dataset)
    prior_probs = {label: class_counts[label] / total_samples for label in [0, 1]}
    
    # Calculate likelihoods
    word_likelihoods = {0: {}, 1: {}}
    for label in [0, 1]:
        total_words = sum(word_counts[label].values())
        for word in vocabulary:
            # Additive smoothing (Laplace smoothing)
            word_likelihoods[label][word] = (word_counts[label][word] + 1) / (total_words + len(vocabulary))
    
    return prior_probs, word_likelihoods

# Step 4: Naive Bayes classifier
def naive_bayes_classifier(text, prior_probs, word_likelihoods, vocabulary):
    """
    Classify new text using the Naive Bayes algorithm.
    """
    words = preprocess_text(text)
    scores = {0: math.log(prior_probs[0]), 1: math.log(prior_probs[1])}
    
    for label in [0, 1]:
        for word in words:
            if word in vocabulary:
                scores[label] += math.log(word_likelihoods[label].get(word, 1 / (len(vocabulary) + 1)))
            else:
                # Handle unseen words with a fallback likelihood
                scores[label] += math.log(1 / (len(vocabulary) + 1))
    
    return 1 if scores[1] > scores[0] else 0


# Step 5: Evaluate the classifier
def evaluate_classifier(test_data, prior_probs, word_likelihoods, vocabulary):
    """
    Compute the accuracy of the classifier.
    """
    correct_predictions = 0
    for text, label in test_data:
        prediction = naive_bayes_classifier(text, prior_probs, word_likelihoods, vocabulary)
        if prediction == label:
            correct_predictions += 1
    return correct_predictions / len(test_data)

# Main Program
if __name__ == "__main__":
    # Dataset
  dataset = [
    ("I love this movie", 1),
    ("This film is fantastic", 1),
    ("What an amazing experience", 1),
    ("I dislike this movie", 0),
    ("Not a great film", 0),
    ("This is terrible", 0),
    ("I hate this", 0),
    ("A wonderful experience", 1),
    ("Such a great film", 1),
    ("This is awful", 0)
]

    
    # Split dataset into training and test sets
training_data = dataset[:8]
test_data = dataset[8:]
    
    # Build vocabulary
vocabulary = build_vocabulary(training_data)
    
    # Calculate probabilities
prior_probs, word_likelihoods = calculate_probabilities(training_data, vocabulary)
    
    # Evaluate the classifier
accuracy = evaluate_classifier(test_data, prior_probs, word_likelihoods, vocabulary)
print(f"Accuracy: {accuracy * 100:.2f}%")
