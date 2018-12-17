import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from Models.SkipGram import SkipGram 

df = pd.read_csv('data/processed_data/idx_pairs.csv')
index_pairs = df.values
vocabulary_size = 11343 #todo dynamic import

# split data into train, validation and test (70/15/15)
data_train, data_test = train_test_split(index_pairs, test_size=0.30, random_state=42)
data_test, data_validation = train_test_split(data_test, test_size=0.5, random_state=42)

# Hyper params
embedding_size = 150
num_epochs = 100
learning_rate = 0.01
# batch_size = 1000

model = SkipGram(num_embeddings=vocabulary_size, 
                embedding_dim=embedding_size, 
                output_size=vocabulary_size)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# print(len(data_train))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data_train = torch.tensor(data_train).to(device)

# training
for epoch in range(num_epochs):
  total_loss = 0
  print('epoch: {0}'.format(epoch) )
  for x, y in data_train:
    model.zero_grad()
    x = torch.tensor(x, dtype=torch.long)
    log_probs = model(x)
    loss = loss_function(log_probs.view(1,-1), torch.tensor([y]))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(total_loss)

# for epoch in range(num_epochs):
#   total_loss = 0
#   for batch in range(math.floor(len(data_train)/1000)+1):
#     startIdx = batch*batch_size
#     batch_data = data_train[startIdx: startIdx+1000]
#     model.zero_grad()
#     for x, y in batch_data:
#       x = torch.tensor(x, dtype=torch.long)
#       log_probs = model(x)
#       loss = loss_function(log_probs.view(1,-1), torch.tensor([y]))
  
# validation
# correct_count = 0
# for x, y in data_validation:
#   model.eval()
#   x = torch.tensor(x, dtype=torch.long)
#   output = model(x).detach().numpy()
#   prediction = np.argmax(output)
#   if y == prediction:
#     correct_count += 1

# print(correct_count)
# print('Accuracy is {0}%'.format(correct_count*100/len(data_validation)))
# print('Accuracy on validation set is {0}%'.format(correct_count*100/5000))
