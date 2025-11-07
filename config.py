import torch

# Device config
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# General network config
batch_size = 128
epochs = 50
learning_rate = 0.1

# Optimizer config
min_lr = 1e-6
optimizer = "sgd"
momentum = 0.9
weight_decay = 1e-3
scheduler = "cosine"

# Learning rate warmup
warmup_epochs = 0
warmup_start_lr = 1e-6

# Checkpoint saving
save_freq = 10
