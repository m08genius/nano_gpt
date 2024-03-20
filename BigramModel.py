import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, 32)
        self.linear = nn.Linear(32, vocab_size)
        
    def forward(self, x):
        # Create the logits
        embeddings = self.embedding_table(x)
        logits = self.linear(embeddings)
        
        return logits

    def generate(self, x, num_tokens=100):
        for _ in range(num_tokens):
            logits = self.forward(x) # logits is B, T, C
            logits_last_time_step = logits[:, -1, :]
            # print(logits.shape)
            # print(logits_last_time_step.shape)
        
            batch_probs = F.softmax(logits_last_time_step, dim=1)
            batch_next_ch = torch.multinomial(batch_probs, 1)
            x = torch.cat((x, batch_next_ch), dim=1) # B, T+1
        return x
