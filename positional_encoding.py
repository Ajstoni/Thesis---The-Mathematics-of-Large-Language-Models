import numpy as np
import math
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

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

# For plotting, let's keep embedding_dim=2 so we can see in 2D
embedding_dim = 2

# Word embeddings + bias for each word
W = np.random.randn(vocab_size, embedding_dim) * 0.01  # center embeddings
W_tilde = np.random.randn(vocab_size, embedding_dim) * 0.01  # context embeddings
b = np.zeros(vocab_size)  # center bias
b_tilde = np.zeros(vocab_size)  # context bias

x_max = 100
alpha = 0.75

def weighting_func(x):
    return (x / x_max) ** alpha if x < x_max else 1

# Hyperparameters
learning_rate = 0.05
n_epochs = 50

# --------------------------------------------------------------
# 5. Training loop
# --------------------------------------------------------------
for epoch in range(n_epochs):
    total_cost = 0.0
    
    for (i, j), Xij in cooccurrences.items():
        if Xij < 1e-10:  # safeguard
            continue
        
        w_ij = weighting_func(Xij)
        
        w_i = W[i]
        w_jt = W_tilde[j]
        b_i = b[i]
        b_jt = b_tilde[j]
        
        inner = np.dot(w_i, w_jt) + b_i + b_jt - math.log(Xij)
        
        cost = w_ij * (inner ** 2)
        total_cost += 0.5 * cost
        
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
# 6. Final Embeddings
# --------------------------------------------------------------
embeddings = W  # We'll visualize the "center" embeddings
print("\nWord Embeddings (2D):")
for idx, word in id_to_word.items():
    print(f"{word:15s} => {embeddings[idx]}")

# --------------------------------------------------------------
# 7. Sinusoidal Position Encoding
# --------------------------------------------------------------
def positional_encoding(pos, d_model):
    """
    Returns a d_model-dimensional vector for the position `pos`.
    We alternate sin() for even indices, cos() for odd indices.
    """
    pe = np.zeros(d_model)
    for i in range(d_model):
        # i is dimension index
        if i % 2 == 0:
            pe[i] = np.sin(pos / (10000 ** (i / d_model)))
        else:
            pe[i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
    return pe

# --------------------------------------------------------------
# 8. Choose a test sentence and apply positional encoding
# --------------------------------------------------------------

test_sentence =  "I like NLP and NLP likes me. GloVe is about capturing co-occurrences in embeddings. I like deep learning"
test_tokens = test_sentence.lower().split()

# Convert to IDs, retrieve embeddings, and apply positional encoding
original_points = []
pos_encoded_points = []

for idx, token in enumerate(test_tokens):
    if token not in word_to_id:  # If it's out of vocab
        # create a dummy small random embedding
        w_original = np.random.randn(embedding_dim) * 0.01
    else:
        w_original = embeddings[word_to_id[token]]
    
    original_points.append(w_original)
    
    # compute position-based encoding
    pos_vector = positional_encoding(idx+1, embedding_dim)
    
    # final = original + position
    w_final = w_original + pos_vector
    
    pos_encoded_points.append(w_final)

original_points = np.array(original_points)
pos_encoded_points = np.array(pos_encoded_points)

# --------------------------------------------------------------
# 9. Plot the effect of positional encoding
# --------------------------------------------------------------
plt.figure(figsize=(6,6))

# Plot original embeddings
plt.scatter(original_points[:, 0], original_points[:, 1], 
            label='Original Embeddings', color='blue', marker='o')

# Plot pos-encoded embeddings
plt.scatter(pos_encoded_points[:, 0], pos_encoded_points[:, 1], 
            label='With Positional Encoding', color='red', marker='x')

# Draw arrows to visualize the shift
for i in range(len(test_tokens)):
    plt.arrow(original_points[i, 0], original_points[i, 1],
              pos_encoded_points[i, 0] - original_points[i, 0],
              pos_encoded_points[i, 1] - original_points[i, 1],
              length_includes_head=True, head_width=0.02, color='gray', alpha=0.7)
    
    plt.text(original_points[i, 0], original_points[i, 1], test_tokens[i],
             fontsize=9, color='blue', ha='right')
    plt.text(pos_encoded_points[i, 0], pos_encoded_points[i, 1], test_tokens[i],
             fontsize=9, color='red', ha='left')

plt.title("Toy GloVe Embeddings vs. Positional Encodings (2D)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.legend()
plt.show()
