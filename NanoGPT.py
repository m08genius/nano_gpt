import torch
import torch.nn as nn

#from BigramModel import BigramModel
from AttentionBlock import AttentionBlock

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, 32)

        # Positional embedding
        self.position_embedding = nn.Embedding(block_size, 32)

        # Higher dimension linear layer
        self.layers = nn.Sequential(nn.Linear(32, 256),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    AttentionBlock(num_heads=4, embedding_size=256),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.3),
                                    AttentionBlock(num_heads=4, embedding_size=128),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    AttentionBlock(num_heads=4, embedding_size=64),
                                    nn.Linear(64, vocab_size))


    def forward(self, x):
        B, T = x.shape
        return self.layers(self.embedding(x) + self.position_embedding(torch.arange(T, device=DEVICE)))


    def get_model_name(self):
        return "nx3_4headed_256T_relu_dropout_attn_decoder"
