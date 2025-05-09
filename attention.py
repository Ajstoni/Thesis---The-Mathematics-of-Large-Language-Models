import torch
import torch.nn.functional as F

def single_head_attention(query, key, value, mask=None):
    """
    Computes single-head attention.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_k)
        key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_k)
        value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_v)
        mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len)fortinf2rere

    Returns:
        torch.Tensor: Attention output of shape (batch_size, seq_len, d_v)
    """
    # Get the dimension of the key vectors
    d_k = query.size(-1)
    
    # Compute the attention scores: (Q * K^T) / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply mask if provided (e.g., for padding or future tokens)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute the weighted sum of values
    output = torch.matmul(attention_weights, value)
    
    return output

# Example usage
if __name__ == "__main__":
    # Define dimensions
    batch_size = 2
    seq_len = 3
    d_k = 4  # Dimension of query and key
    d_v = 4  # Dimension of value

    # Create random input tensors
    query = torch.randn(batch_size, seq_len, d_k)
    key = torch.randn(batch_size, seq_len, d_k)
    value = torch.randn(batch_size, seq_len, d_v)

    # Compute attention output
    output = single_head_attention(query, key, value)
    
    # Print the shape of the output
    print("Output shape:", output.shape)  # Expected: (2, 3, 4)