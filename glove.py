import numpy as np
import math
from collections import Counter, defaultdict

# --------------------------------------------------------------
# 1. Toy corpus and tokenization
# --------------------------------------------------------------
corpus = [
    "I like NLP and NLP likes me",
    "GloVe is about capturing co-occurrences in embeddings",
    "I like deep learning"
]

# Very simple tokenization by splitting on whitespace
# In practice, you'd handle punctuation, casing, etc. more carefully
tokens = []
for line in corpus:
    tokens.extend(line.lower().split())

print("Tokens:", tokens)

# --------------------------------------------------------------
# 2. Build vocabulary
# --------------------------------------------------------------
word_counts = Counter(tokens)
vocab = sorted(word_counts.keys())  # list of words
vocab_size = len(vocab)

word_to_id = {word: idx for idx, word in enumerate(vocab)}
id_to_word = {idx: word for word, idx in word_to_id.items()}

print("Vocabulary:", vocab)

# --------------------------------------------------------------
# 3. Construct the co-occurrence matrix
# --------------------------------------------------------------
window_size = 2  # context window on each side
cooccurrences = defaultdict(float)

for i, word in enumerate(tokens):
    w_i = word_to_id[word]
    
    # Collect left/right context within window_size
    start = max(i - window_size, 0)
    end = min(i + window_size + 1, len(tokens))
    
    for j in range(start, end):
        if j == i:
            continue
        w_j = word_to_id[tokens[j]]
        # Increase co-occurrence count
        cooccurrences[(w_i, w_j)] += 1.0

print("Number of non-zero co-occurrences:", len(cooccurrences))

# --------------------------------------------------------------
# 4. Define GloVe parameters and cost function
# --------------------------------------------------------------
embedding_dim = 10
# Word embeddings + bias for each word
# We'll learn "center" embeddings and "context" embeddings
W = np.random.randn(vocab_size, embedding_dim) * 0.01  # center embeddings
W_tilde = np.random.randn(vocab_size, embedding_dim) * 0.01  # context embeddings
b = np.zeros(vocab_size)  # center bias
b_tilde = np.zeros(vocab_size)  # context bias

# Weighting function f(x). Typically:
# f(x) = (x / x_max)^alpha if x < x_max, else 1
x_max = 100
alpha = 0.75

def weighting_func(x):
    return (x / x_max) ** alpha if x < x_max else 1

# Hyperparameters
learning_rate = 0.05
n_epochs = 100

# --------------------------------------------------------------
# 5. Training loop
# --------------------------------------------------------------
for epoch in range(n_epochs):
    total_cost = 0.0
    
    for (i, j), Xij in cooccurrences.items():
        if Xij < 1e-10:  # safeguard
            continue
        
        # Compute weight
        w_ij = weighting_func(Xij)
        
        # Dot product of embeddings + biases
        w_i = W[i]
        w_jt = W_tilde[j]
        b_i = b[i]
        b_jt = b_tilde[j]
        
        # The expression (w_i^T w_j + b_i + b_j - log(Xij))
        inner = np.dot(w_i, w_jt) + b_i + b_jt - math.log(Xij)
        
        # Cost contribution
        cost = w_ij * (inner ** 2)
        total_cost += 0.5 * cost  # 1/2 for convenience in derivative

        # Compute gradients
        grad = w_ij * inner
        
        # Grad w.r.t. w_i and w_jt
        grad_w_i = grad * w_jt
        grad_w_jt = grad * w_i
        
        # Update embeddings
        W[i]     -= learning_rate * grad_w_i
        W_tilde[j] -= learning_rate * grad_w_jt
        
        # Update biases
        b[i]     -= learning_rate * grad
        b_tilde[j] -= learning_rate * grad
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Cost: {total_cost:.4f}")

# --------------------------------------------------------------
# 6. Examine learned embeddings
# --------------------------------------------------------------
embeddings = W  # we can just look at the "center" embeddings
print("\nWord Embeddings:")
for idx, word in id_to_word.items():
    print(f"{word:15s} => {embeddings[idx]}")
