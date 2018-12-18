import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from Models.SkipGram import SkipGram 


def onehot_encode(indices, vocabulary_size):
  result = torch.Tensor().new_zeros(len(indices), vocabulary_size)
  print(result.size())

df = pd.read_csv('data/processed_data/idx_pairs.csv')
index_pairs = df.values
vocabulary_size = 11343 #todo dynamic import

# split data into train, validation and test (70/15/15)
data_train, data_test = train_test_split(index_pairs, test_size=0.20, random_state=42)
data_test, data_validation = train_test_split(data_test, test_size=0.5, random_state=42)

# Hyper params
embedding_size = 300
num_epochs = 2
learning_rate = 0.0001
batch_size = 512

model = SkipGram(vocab_size=vocabulary_size, 
                embedding_dim=embedding_size)
losses = []
loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data_train = torch.tensor(data_train).to(device)
# print(len(data_train)) =>795216
num_batch = int(np.ceil(len(data_train)/batch_size))

# batch training
for epoch in range(num_epochs):
  for batch in range(num_batch):
    print('epoch: {0}/{1}, batch: {2}/{3}'.format(epoch, num_epochs, batch, num_batch))
    batch_loss = 0
    startIndex = batch*batch_size
    x = data_train[startIndex:startIndex+batch_size, 0]
    y = data_train[startIndex:startIndex+batch_size, 1]
    model.zero_grad()
    x = torch.tensor(x, dtype=torch.long)
    log_probs = model(x)
    loss = loss_function(log_probs, torch.tensor(y))
    loss.backward()
    optimizer.step()
    batch_loss += loss.item()
    print(batch_loss)

# training
# for epoch in range(num_epochs):
#   total_loss = 0
#   print('epoch: {0}'.format(epoch) )
#   for x, y in data_train:
#     model.zero_grad()
#     x = torch.tensor(x, dtype=torch.long)
#     log_probs = model(x)
#     loss = loss_function(log_probs.view(1,-1), torch.tensor([y]))
#     loss.backward()
#     optimizer.step()
#     total_loss += loss.item()
#   print(total_loss)
  
# validation
correct_count = 0
for x, y in data_validation:
  model.eval()
  x = torch.tensor(x, dtype=torch.long)
  output = model(x).detach().numpy()
  prediction = np.argmax(output)
  if y == prediction:
    correct_count += 1

print('Accuracy is {0}%'.format(correct_count*100/len(data_validation)))
# print('Accuracy on validation set is {0}%'.format(correct_count*100/5000))
