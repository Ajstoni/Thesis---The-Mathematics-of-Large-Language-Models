import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --------------------------------------------------------------
# 1. Positional Encoding Function
# --------------------------------------------------------------
def positional_encoding(pos, d_model):
    """
    Returns a d_model-dimensional vector for position `pos`.
    Alternates sin() for even indices, cos() for odd indices.
    """
    pe = np.zeros(d_model)
    for i in range(d_model):
        if i % 2 == 0:
            pe[i] = np.sin(pos / (10000 ** (i / d_model)))
        else:
            pe[i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
    return pe

# --------------------------------------------------------------
# 2. Parameters for Visualization
# --------------------------------------------------------------
d_model = 512  # Increased for more realism
max_pos = 1000  # Number of positions to visualize

# Compute positional encodings for positions 1 to max_pos
positions = np.arange(1, max_pos + 1)
pe_matrix = np.zeros((max_pos, d_model))
for pos in positions:
    pe_matrix[pos-1, :] = positional_encoding(pos, d_model)

# --------------------------------------------------------------
# 3. Create Enhanced Visualization
# --------------------------------------------------------------
fig = plt.figure(figsize=(12, 10))

# Subplot 1: Heatmap of Positional Encodings
ax1 = fig.add_subplot(211)
im = ax1.imshow(pe_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(im, ax=ax1, label='Encoding Value')
ax1.set_xticks(np.arange(d_model))
ax1.set_xticklabels([f'{i}' for i in range(d_model)])
ax1.set_yticks(np.arange(max_pos))
ax1.set_yticklabels([f'{i+1}' for i in range(max_pos)])
ax1.set_xlabel('Dimension Index (i)')
ax1.set_ylabel('Position (pos)')
ax1.set_title('Positional Encoding Values Across Positions and Dimensions')

# Subplot 2: Line Plot for Selected Dimensions
ax2 = fig.add_subplot(212)
selected_dims = [0, 2]  # Reduced for clarity
for dim in selected_dims:
    if dim % 2 == 0:
        label = f'sin(pos/10000^{{{dim}/{d_model}}})'
    else:
        label = f'cos(pos/10000^{{{dim-1}/{d_model}}})'
    ax2.plot(positions, pe_matrix[:, dim], marker='o', label=label)
ax2.set_xlabel('Position (pos)')
ax2.set_ylabel('Encoding Value')
ax2.set_title('Positional Encoding for Selected Dimensions')
ax2.grid(True)
ax2.legend()

# Add Numerical Table for First Few Positions
table_text = "Positional Encoding Values (First 3 Positions):\n"
for pos in range(3):  # Positions 1, 2, 3
    values = pe_matrix[pos, :4]  # First 4 dimensions
    table_text += f"Pos {pos+1}: {values.round(3)}\n"
plt.gcf().text(0.02, 0.02, table_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('positional_encoding_descriptive.png')
plt.show()