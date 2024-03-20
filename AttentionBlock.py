import torch
import torch.nn as nn
import math

class SingleHeadAttentionModule(nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.q = nn.Linear(embedding_size, head_size, bias=False)
        self.k = nn.Linear(embedding_size, head_size, bias=False)
        self.v = nn.Linear(embedding_size, head_size, bias=False)
    
    def forward(self, query, key, value):
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)
        
        att_mat = query @ torch.transpose(key, -2, -1)
        att_mat_scaled = torch.div(att_mat, torch.sqrt(torch.tensor(self.head_size)))
        
        att_mat_scaled[torch.tril(torch.ones_like(att_mat_scaled)) == 0.] = -float('inf')
        
        return att_mat_scaled.softmax(-1) @ value


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, num_heads, embedding_size):
        super().__init__()
        assert num_heads != 0
        assert embedding_size % num_heads == 0
        
        self.heads = num_heads
        self.head_size = int(embedding_size / num_heads)
        
        self.q = nn.Linear(embedding_size, embedding_size, bias=False)
        self.k = nn.Linear(embedding_size, embedding_size, bias=False)
        self.v = nn.Linear(embedding_size, embedding_size, bias=False)
        
            
    def forward(self, query, key, value):
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.head_size).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.head_size).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.head_size).permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))
        
        attn_scores[torch.tril(torch.ones_like(attn_scores)) == 0.] = -float('inf')
        
        context = attn_scores.softmax(-1) @ value
        
        return context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.head_size)


class FeedForwardModule(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        
        self.layers = nn.Sequential(nn.Linear(embedding_size, 4 * embedding_size), 
                                    nn.ReLU(), 
                                    nn.Linear(4 * embedding_size, embedding_size), 
                                    nn.ReLU())
        
    def forward(self, x):
        return self.layers(x)

class AttentionBlock(nn.Module):
    def __init__(self, num_heads, embedding_size):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.multi_head_attention = MultiHeadAttentionModule(num_heads=num_heads, embedding_size=embedding_size)
        
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.feed_forward = FeedForwardModule(embedding_size=embedding_size)
    
    def forward(self, x):
        x_layer_norm = self.layer_norm1(x)
        x = x + self.multi_head_attention(x_layer_norm, x_layer_norm, x_layer_norm)
        
        return x + self.feed_forward(self.layer_norm2(x))


# class MultiHeadAttentionModule(nn.Module):
#     def __init__(self, num_heads, embedding_size):
#         super().__init__()
#         assert num_heads != 0
#         assert embedding_size % num_heads == 0
        
#         self.head_size = int(embedding_size / num_heads)
#         self.heads = nn.ModuleList([SingleHeadAttentionModule(embedding_size=embedding_size, head_size=self.head_size) for _ in range(1, num_heads+1)])
    
#     def forward(self, query, key, value):
#         return torch.cat([sahm(query, key, value) for sahm in self.heads], dim=-1)
