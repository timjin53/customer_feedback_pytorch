import pandas as pd
import numpy as np

df = pd.read_csv('raw_data/completion_comments.csv')
#remove punctuations, to lowercase and convert into list
comment_list = df['Task Completion Comments'].str.replace(r'[^\w\s]','').str.lower().tolist()

# tokenize comment and construct index pairs of words
def tokenize_comment(comment_list):
    return [comment.split() for comment in comment_list]

tokenized_comments = tokenize_comment(comment_list)

vocabulary = []
for tokenized_comment in tokenized_comments:
  for word in tokenized_comment:
    if word not in vocabulary:
      vocabulary.append(word)

word_to_index = { word: index for (index, word) in enumerate(vocabulary)}
index_to_word = { index: word for (index, word) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)

window_size = 2
index_pairs = []

for tokenized_comment in tokenized_comments:
  indices = [word_to_index[word] for word in tokenized_comment]
  for center_word_position, word_index in enumerate(indices):
    left = center_word_position - window_size
    right = center_word_position + window_size + 1
    for context_word_position in range(left, right):
      if(context_word_position < 0 or context_word_position >= len(indices) or context_word_position == center_word_position):
        continue
      else:
        index_pairs.append([indices[center_word_position], indices[context_word_position]])

print(vocabulary_size)

index_pairs = np.array(index_pairs)
df = pd.DataFrame({'Center Word Index': index_pairs[:, 0], 'Target Word Index': index_pairs[:, 1]})
df.to_csv('processed_data/idx_pairs.csv', index=False)
