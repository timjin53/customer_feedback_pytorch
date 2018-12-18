import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(SkipGram, self).__init__()
    self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    self.fc_layer = nn.Linear(embedding_dim, vocab_size)

  def forward(self, x):
    x = self.embedding_layer(x)
    x = self.fc_layer(x)
    return F.log_softmax(x, dim=0)
    