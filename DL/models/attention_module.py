"""PyTorch implementation of Attention Module
"""
import torch
import torch.nn as nn


class attention_module(nn.Module):
    def __init__(self, feature_dim, embed_dim, num_heads=4):
        super(attention_module, self).__init__()
        self.num_heads = num_heads
        self.query_projection = nn.Linear(feature_dim, embed_dim)
        self.key_projection = nn.Linear(feature_dim, embed_dim)
        self.value_projection = nn.Linear(feature_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, bias=False)

    @staticmethod
    def get_module_name():
        return "attention_module"

    def forward(self, features1, features2, features3=None):
        # Project query, key, value to the desired dimensions
        query = self.query_projection(features1)
        key = self.key_projection(features2)
        value = self.value_projection(features3)
        # Compute attention (query is from target, key and value are from source)
        output, attn_weights = self.attention(query, key, value)
        return output, attn_weights


if __name__ == '__main__':
    features1 = torch.zeros(2, 10, 128)
    features2 = torch.zeros(2, 6, 128)
    features3 = torch.zeros(2, 6, 128)
    MIXA = attention_module(128, 128)
    output = MIXA(features1, features2, features3)
    print('Job Done')