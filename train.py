import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from Models.SkipGram import SkipGram 

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

df = pd.read_csv('data/processed_data/idx_pairs.csv')
index_pairs = df.values
vocabulary_size = 11343 #todo dynamic import

index_pairs = index_pairs

# split data into train, validation and test (80/10/10)
data_train, data_test = train_test_split(index_pairs, test_size=0.20, random_state=42)
data_test, data_validation = train_test_split(data_test, test_size=0.5, random_state=42)

# Hyper params
embedding_size = 300
num_epochs = 30
learning_rate = 0.01
batch_size = 2048

model = SkipGram(vocab_size=vocabulary_size, 
                embedding_dim=embedding_size)
loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data_train = torch.tensor(data_train)
# print(len(data_train)) =>795216
num_batch = int(np.ceil(len(data_train)/batch_size))
# model.to(device)

# batch training
for epoch in range(num_epochs):
  scheduler.step()
  print(get_lr(optimizer))
  for batch in range(num_batch):
    print('epoch: {0}/{1}, batch: {2}/{3}'.format(epoch+1, num_epochs, batch+1, num_batch))
    batch_loss = 0
    startIndex = batch*batch_size
    x = data_train[startIndex:startIndex+batch_size, 0]
    y = data_train[startIndex:startIndex+batch_size, 1]

    x = x.to(device)
    y = y.to(device)

    model.zero_grad()
    x = torch.tensor(x, dtype=torch.long)
    log_probs = model(x).to(device)
    loss = loss_function(log_probs, torch.tensor(y))
    loss.backward()
    optimizer.step()
    batch_loss += loss.item()
    print('loss: {0}'.format(batch_loss))

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
    print(output)
    print(prediction)
    correct_count += 1

print('Accuracy is {0}%'.format(correct_count*100/len(data_validation)))
# print('Accuracy on validation set is {0}%'.format(correct_count*100/5000))
