# %%
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
# %%

model = HookedTransformer.from_pretrained("gelu-4l")
model("Hello world!")

# %%
model.cfg

# %%
dir(model)


# %%
"""
'OV',
 'QK',
 'T_destination',
 'W_E',
 'W_E_pos',
 'W_K',
 'W_O',
 'W_Q',
 'W_U',
 'W_V',
 'W_in',
 'W_out',
 'W_pos',

  'b_K',
 'b_O',
 'b_Q',
 'b_U',
 'b_V',
 'b_in',
 'b_out',
"""
# %%

d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_model = model.cfg.d_model
d_vocab = model.cfg.d_vocab
n_ctx = model.cfg.n_ctx
n_heads = model.cfg.n_heads
n_layers = model.cfg.n_layers

# make a nice printout of the above values
print(f"d_head: {d_head}\nd_mlp: {d_mlp}\nd_model: {d_model}\nd_vocab: {d_vocab}\nn_ctx: {n_ctx}\nn_heads: {n_heads}\nn_layers: {n_layers}")


# %%
OPEN_TOKEN = model.to_single_token("(")
CLOSE_TOKEN = model.to_single_token(")")
OPEN_SPACE_TOKEN = model.to_single_token(" (")
CLOSE_SPACE_TOKEN = model.to_single_token(" )")
print(f"OPEN_TOKEN: {OPEN_TOKEN}\nCLOSE_TOKEN: {CLOSE_TOKEN}, \nOPEN_SPACE_TOKEN: {OPEN_SPACE_TOKEN}\nCLOSE_SPACE_TOKEN: {CLOSE_SPACE_TOKEN}")
# %%

# %%
tokenizer = model.tokenizer
# Assume 'tokenizer' is the tokenizer used by your model
all_tokens = list(tokenizer.get_vocab().keys())

# Filter the list to include only tokens that contain ")"
tokens_with_bracket = [token for token in all_tokens if ")" in token]
print(f"Number of tokens containing ')': {len(tokens_with_bracket)}")

tokens_with_close_bracket = [token for token in all_tokens if "(" in token]
print(f"Number of tokens containing '(': {len(tokens_with_close_bracket)}")

# do any tokens contain both ( and )?
tokens_with_both_brackets = [token for token in all_tokens if "(" in token and ")" in token]
print(f"Number of tokens containing both '(': {len(tokens_with_both_brackets)}")

# print the tokens that contain both
print(tokens_with_both_brackets)

# get tokens that have ( that dont have )
tokens_with_open_bracket = [token for token in all_tokens if "(" in token and ")" not in token]

# get tokens that have ) that dont have (

tokens_with_close_bracket = [token for token in all_tokens if ")" in token and "(" not in token]


OPEN_TOKENS = [model.to_single_token(token) for token in tokens_with_open_bracket]
CLOSE_TOKENS = [model.to_single_token(token) for token in tokens_with_close_bracket]

# %%
# get embedding matrix
W_pos = model.W_pos # positional embeddings (n_ctx, d_model) this maps to the position in the sequence
W_E = model.W_E # token embeddings (d_vocab, d_model) this maps to the token id
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# subtract the mean embedding from W_E
W_E_mean = W_E.mean(dim=0)
W_E_centered = W_E - W_E_mean

# get the centered embedding for the open and close tokens
OPEN_EMBEDDING = W_E_centered[OPEN_TOKEN]
CLOSE_EMBEDDING = W_E_centered[CLOSE_TOKEN]
OPEN_SPACE_EMBEDDING = W_E_centered[OPEN_SPACE_TOKEN]
CLOSE_SPACE_EMBEDDING = W_E_centered[CLOSE_SPACE_TOKEN]

# lets do cosine sim between each of these 4 vectors
def cosine_sim(a, b):
    return (a @ b) / (a.norm() * b.norm())

vecs = {'(': OPEN_EMBEDDING, ' (': OPEN_SPACE_EMBEDDING, ')': CLOSE_EMBEDDING, ' )': CLOSE_SPACE_EMBEDDING}
labels = list(vecs.keys())
matrix = []

for k1, v1 in vecs.items():
    row = []
    for k2, v2 in vecs.items():
        row.append(cosine_sim(v1, v2))
    matrix.append(row)

# Convert list of lists to numpy array, but convert elements away from torch
matrix = np.array([[item.detach().cpu().numpy() for item in row] for row in matrix])

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Display an image on the axes
cax = ax.imshow(matrix, cmap='viridis') 

# Set labels, add " marks around the labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
labels = [f'"{label}"' for label in labels]
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Loop over data dimensions and create text annotations
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, round(matrix[i, j], 2),
                       ha="center", va="center", color="w")

# Create colorbar
fig.colorbar(cax)
plt.title('The cosine similarity between the embeddings of the open and close parentheses')

# Show plot
plt.show()

# %%



# %%
# Import the load_dataset function from the Hugging Face's datasets library
from datasets import load_dataset

# Load the "NeelNanda/code-10k" dataset and take the training split
code_data = load_dataset("NeelNanda/code-10k", split="train")

# Print the type of the code_data object for debugging
print(type(code_data))

# Extract the "text" field from the second entry (index 1) in the dataset
text = code_data[0]["text"]

# Convert the text (presumably code) into tokens for processing using the model
tokens = model.to_tokens(text)[0]

# Print the shape of the tokens tensor for debugging
print(tokens.shape)

# Run a forward pass of the model with the given tokens and cache intermediate computations
logits, cache = model.run_with_cache(tokens)

# Initialize a list to keep track of the cumulative count of brackets in the code
cumsum_brackets = [0]

# Iterate over all tokens returned by the model's to_str_tokens function (skipping the first token)
for tok in model.to_str_tokens(tokens)[1:]:
    # Append the cumulative count of "(" and ")" to the cumsum_brackets list
    cumsum_brackets.append(cumsum_brackets[-1] + tok.count("(") - tok.count(")"))

# Convert the list cumsum_brackets to a NumPy array for easier processing
cumsum_brackets = np.array(cumsum_brackets)

# Create a list of strings where each string is a token and its index, useful for debugging or analysis
token_list = [f"{s}/{i}" for i, s in enumerate(model.to_str_tokens(tokens))]

log_probs = logits.softmax(dim=-1)
logit_close = log_probs[0, :, CLOSE_TOKEN]
logit_close_space = log_probs[0, :, CLOSE_SPACE_TOKEN]
logit_close_total = logit_close + logit_close_space
logit_close_total = logit_close_total.detach().cpu().numpy()

plt.scatter(cumsum_brackets, logit_close_total)


# %%
import torch
# Get all single tokens
single_tokens = [token for token in all_tokens if len(tokenizer.tokenize(token)) == 1]

# Get single tokens containing either "(" or ")"
single_tokens_with_brackets = [token for token in single_tokens if "(" in token or ")" in token]

# Calculate the difference between the counts of "(" and ")" in each single token
single_token_diffs = {token: token.count("(") - token.count(")") for token in single_tokens_with_brackets}

# Sort the single tokens based on the difference
sorted_single_tokens_with_brackets = sorted(single_token_diffs.items(), key=lambda item: item[1])

# Get sorted list of single tokens for analysis
sorted_single_tokens = [item[0] for item in sorted_single_tokens_with_brackets]

# Get embeddings
single_token_embeddings = {token: W_E_centered[model.to_single_token(token)] for token in sorted_single_tokens}
# Assuming single_token_embeddings is a dictionary where the keys are tokens and 
# the values are the corresponding embeddings

# First, convert the embeddings dictionary to an ordered list of embeddings and an ordered list of tokens
ordered_embeddings = [v for k, v in single_token_embeddings.items()]
ordered_tokens = [k for k, v in single_token_embeddings.items()]

# Convert list of embeddings to a tensor
embedding_matrix = torch.stack(ordered_embeddings)

# Normalize the embeddings
norms = embedding_matrix.norm(p=2, dim=1, keepdim=True)
normalized_embeddings = embedding_matrix.div(norms)

# Compute the cosine similarity matrix using matrix multiplication
cos_sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

# Convert the tensor to a numpy array
cos_sim_array = cos_sim_matrix.detach().cpu().numpy()

# %%

# use plotly express
import plotly.express as px
import pandas as pd

# Create a dataframe from the cosine similarity matrix
df = pd.DataFrame(cos_sim_array, columns=ordered_tokens, index=ordered_tokens)

# Create a heatmap using plotly express
fig = px.imshow(df, color_continuous_scale='viridis')

# add x and y labels
fig.update_xaxes(title_text='Tokens ordered by the diff between # of "(" and ")"')
fig.update_yaxes(title_text='Tokens')

# add title
fig.update_layout(title_text='Cosine similarity between the embeddings of tokens')
# Show plot
fig.show()

# %%
# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Display an image on the axes
cax = ax.imshow(cos_sim_array, cmap='viridis')

# Set labels, add " marks around the labels
ax.set_xticks(np.arange(len(ordered_tokens)))
ax.set_yticks(np.arange(len(ordered_tokens)))
ordered_tokens = [f'"{token}"' for token in ordered_tokens]
ax.set_xticklabels(ordered_tokens)
ax.set_yticklabels(ordered_tokens)

# Rotate x-labels for readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations
for i in range(len(ordered_tokens)):
    for j in range(len(ordered_tokens)):
        text = ax.text(j, i, round(cos_sim_array[i, j], 2),
                       ha="center", va="center", color="w")

# Create colorbar
fig.colorbar(cax)
plt.title('The cosine similarity between the embeddings of the single tokens')

# Show plot
plt.show()

# %%
# Import the necessary libraries
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tqdm

# Load the "NeelNanda/code-10k" dataset and take the training split
code_data = load_dataset("NeelNanda/code-10k", split="train")

# The lists to hold all bracket counts and corresponding probabilities
all_brackets = []
all_probs = []

# Select the first 1000 examples and create a new Dataset
subset_data = code_data.select(range(100))

# Process the first 1000 examples
for example in tqdm.tqdm(subset_data):
    text = example["text"]
    tokens = model.to_tokens(text)[0]
    
    # Get model output
    output, cache = model.run_with_cache(tokens)

    # Cumulative bracket count
    cumsum_brackets = [0]
    for tok in model.to_str_tokens(tokens)[1:]:
        cumsum_brackets.append(cumsum_brackets[-1] + tok.count("(") - tok.count(")"))
    cumsum_brackets = np.array(cumsum_brackets)
    
    # Calculate probabilities
    probs = output.softmax(dim=-1)
    prob_close = probs[0, :, CLOSE_TOKEN]
    prob_close_space = probs[0, :, CLOSE_SPACE_TOKEN]
    prob_close_total = prob_close + prob_close_space
    prob_close_total = prob_close_total.detach().cpu().numpy()

    # Add to our lists
    all_brackets.extend(cumsum_brackets)
    all_probs.extend(prob_close_total)

# Convert to DataFrame for use with seaborn
data = pd.DataFrame({
    'Brackets': all_brackets,
    'Probability of )': all_probs
})
#  %%
# Create a scatter plot with seaborn, showing bootstrap estimates of the mean
sns.lmplot(x='Brackets', y='Probabilities', data=data, x_estimator=np.mean, fit_reg=False)




# %%
