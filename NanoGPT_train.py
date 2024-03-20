import torch
import torch.nn.functional as F
import time
import pickle

from NanoGPT import NanoGPT
from NanoGPT_Tokenizer import NanoGPT_Tokenizer
from NanoGPT_Dataset import NanoGPT_Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

n_gpus = torch.cuda.device_count()
print(n_gpus)

tokenizer = NanoGPT_Tokenizer("input.txt")
dataset = NanoGPT_Dataset(tokenizer=tokenizer, train_percent=0.9)

block_size = 256
model = NanoGPT(vocab_size=tokenizer.get_vocab_size(), block_size=block_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
optimizer.zero_grad()

global_train_loss = []
global_val_loss = []

filepath = "."
template = "{}/{}_{}_{}.{}"

model_file_template = template.format(filepath, model.get_model_name(), "MODEL", "{}", "pt")
global_train_loss_template = template.format(filepath, model.get_model_name(), "TRAIN_LOSS", "{}", "pkl")
global_val_loss_template = template.format(filepath, model.get_model_name(), "VAL_LOSS", "{}", "pkl")
print(model_file_template)
print(global_train_loss_template)
print(global_val_loss_template)

global_epoch = 1
if not global_epoch is None:
    model.load_state_dict(torch.load(model_file_template.format(global_epoch)))

    with open(global_train_loss_template.format(global_epoch), 'rb') as f:
        global_train_loss = pickle.load(f)

    with open(global_val_loss_template.format(global_epoch), 'rb') as f:
        global_val_loss = pickle.load(f)

    print(global_train_loss)
    print(global_val_loss)


batch_size = 1024
train_dl, val_dl = dataset.get_train_val_dl(batch_size=batch_size, block_size=block_size)

epochs = 60
epoch_start = 0 if global_epoch is None else global_epoch + 1
for epoch in range(epoch_start, epochs+epoch_start):
    print(f"Epoch: {epoch}")
    
    start_train = time.time()
    total_train_loss = 0.0
    for x_batch, y_batch in train_dl:
        # Move to gpu.
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        pred = model(x_batch)
        B, T, C = pred.shape
        loss = F.cross_entropy(pred.view(B*T, C), y_batch.view(B*T))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_loss += (loss.item() * B)
    end_train = time.time()

    model.eval()
    start_val = time.time()
    total_val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_dl:
            # move to gpu
            x_val, y_val = x_val.to(device), y_val.to(device)

            pred = model(x_val)
            B, T, C = pred.shape
            loss = F.cross_entropy(pred.view(B*T, C), y_val.view(B*T))
            total_val_loss += (loss.item() * B)
    end_val = time.time()
    model.train()

    global_train_loss.append(total_train_loss / len(train_dl.dataset))
    global_val_loss.append(total_val_loss / len(val_dl.dataset))
    print(f"Train time: {end_train - start_train}, Val time: {end_val - start_val}")
    print(f"Train Loss: {global_train_loss[-1]}, Val Loss: {global_val_loss[-1]}")
    torch.save(model.state_dict(), model_file_template.format(epoch))
    with open(global_train_loss_template.format(epoch), "wb") as fp:
        pickle.dump(global_train_loss, fp)
    with open(global_val_loss_template.format(epoch), "wb") as fp:
        pickle.dump(global_val_loss, fp)
    global_epoch = epoch

print("Done epoch: ", global_epoch)
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
