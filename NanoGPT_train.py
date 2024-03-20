import torch
import torch.nn.functional as F
import time

from NanoGPT import NanoGPT
from NanoGPT_Tokenizer import NanoGPT_Tokenizer
from NanoGPT_Dataset import NanoGPT_Dataset
from ModelCheckpoint import ModelCheckpoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

n_gpus = torch.cuda.device_count()
print(n_gpus)

# Load and create datasets
input_data_file = "./input.txt"
train_percent = 0.9
tokenizer = NanoGPT_Tokenizer(input_data_file)
dataset = NanoGPT_Dataset(tokenizer=tokenizer, train_percent=train_percent)

# Model PARAMS
block_size = 256
model = NanoGPT(vocab_size=tokenizer.get_vocab_size(), block_size=block_size)

# Load the model PARAMS
filepath = "./checkpoints"
epoch_id = 1
model_checkpoint = ModelCheckpoint(filepath=filepath, epoch_id=epoch_id, model_name=model.get_model_name(), block_size=block_size)
model_checkpoint.load_model_params(model=model)
model = model.to(device)

# Create optimzer
learning_rate = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
optimizer.zero_grad()

# Load the datasets
batch_size = 1024
train_dl, val_dl = dataset.get_train_val_dl(batch_size=batch_size, block_size=block_size)

# Training loop
epochs = 60
train_loss, val_loss = model_checkpoint.load_loss()
for epoch in range(epochs):
    model_checkpoint.update_epoch()
    print(f"Epoch: {model_checkpoint.get_epoch()}")
    
    start_train = time.time()
    train_loss_epoch = 0.0
    for x_batch, y_batch in train_dl:
        # Move to gpu.
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        pred = model(x_batch)
        B, T, C = pred.shape
        loss = F.cross_entropy(pred.view(B*T, C), y_batch.view(B*T))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch += (loss.item() * B)
    end_train = time.time()

    model.eval()
    start_val = time.time()
    val_loss_epoch = 0.0
    with torch.no_grad():
        for x_val, y_val in val_dl:
            # move to gpu
            x_val, y_val = x_val.to(device), y_val.to(device)

            pred = model(x_val)
            B, T, C = pred.shape
            loss = F.cross_entropy(pred.view(B*T, C), y_val.view(B*T))
            val_loss_epoch += (loss.item() * B)
    end_val = time.time()
    model.train()

    train_loss.append(train_loss_epoch / len(train_dl.dataset))
    train_loss.append(val_loss_epoch / len(val_dl.dataset))
    print(f"Train time: {end_train - start_train}, Val time: {end_val - start_val}")
    print(f"Train Loss: {train_loss[-1]}, Val Loss: {val_loss[-1]}")

    model_checkpoint.save(model=model, train_loss=train_loss, val_loss=val_loss)

# from matplotlib import pyplot as plt
# import numpy as np
# plt.plot(np.arange(len(global_train_loss)), global_train_loss, label="Train Loss")
# plt.plot(np.arange(len(global_val_loss)), global_val_loss, label="Val Loss")
# plt.legend()
# plt.show()

# def generate_output(model, x, block_size, num_tokens=100):
#     result = []
#     for _ in range(num_tokens):
#         logits = model(x) # logits is B, T, C
#         logits_last_time_step = logits[:, -1, :]

#         batch_probs = F.softmax(logits_last_time_step, dim=1)
#         batch_next_ch = torch.multinomial(batch_probs, 1)
#         result.append(batch_next_ch.item())
        
#         x = torch.cat((x, batch_next_ch), dim=1) # B, T+1
#         if x.shape[1] > block_size:
#             x = x[:, 1:block_size+1]
            
#     return result

# empty_token = torch.zeros((1, 1), dtype=torch.long, device=device)
# out = generate_output(model=model, x=empty_token, block_size=block_size, num_tokens=400)
# print("".join(decode(out)))
