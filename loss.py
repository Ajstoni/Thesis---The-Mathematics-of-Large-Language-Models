import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

def single_head_attention(query, key, value):
    """
    Computes single-head attention.
    
    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_k)
        key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_k)
        value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_v)
    
    Returns:
        torch.Tensor: Attention output of shape (batch_size, seq_len, d_v)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output

# Simulation parameters
batch_size = 1
seq_len = 3
d_k = 8  # Dimension of query and key
d_v = 8  # Dimension of value
num_iterations = 1000
learning_rate = 0.01

# Randomly initialize Q, K, V with requires_grad=True for gradient computation
query = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
key = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
value = torch.randn(batch_size, seq_len, d_v, requires_grad=True)

# Define a synthetic target output (ground truth) for the attention mechanism
# In practice, this would be derived from a training dataset
target = torch.ones(batch_size, seq_len, d_v)  # Example: all ones for simplicity

# Loss function (Mean Squared Error)
loss_fn = torch.nn.MSELoss()

# Lists to store loss values for plotting
losses = []

# Gradient descent for loop
for iteration in range(num_iterations):
    # Compute the attention output
    output = single_head_attention(query, key, value)
    
    # Compute the loss
    loss = loss_fn(output, target)
    losses.append(loss.item())
    
    # Zero out gradients
    if query.grad is not None:
        query.grad.zero_()
    if key.grad is not None:
        key.grad.zero_()
    if value.grad is not None:
        value.grad.zero_()
    
    # Backpropagation
    loss.backward()
    
    # Update Q, K, V using gradient descent
    with torch.no_grad():
        query -= learning_rate * query.grad
        key -= learning_rate * key.grad
        value -= learning_rate * value.grad
        
        # Ensure requires_grad is preserved after in-place operations
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

# Plot the loss over iterations
plt.figure(figsize=(8, 6))
plt.plot(range(num_iterations), losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Loss During Training of Single-Head Attention')
plt.grid(True)
plt.legend()
plt.show()