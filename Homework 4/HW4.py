import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

with open('data/train', 'r') as f:
    train_data = f.readlines()

for i in range(len(train_data)):
    train_data[i] = train_data[i].split()

sentences = []
tags = []
current_sentence = []
current_tags = []
for line in train_data:
    if len(line) == 0:
        sentences.append(current_sentence)
        tags.append(current_tags)
        current_sentence = []
        current_tags = []
    else:
        current_sentence.append(line[1])
        current_tags.append(line[2])


word_to_ix = {}
tag_to_ix = {"O": 0, "B-MISC": 1, "I-MISC": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6, "B-PER": 7, "I-PER": 8}
for sentence in sentences:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

X_train = []
for sentence in sentences:
    sentence_indices = [word_to_ix[word] for word in sentence]
    X_train.append(torch.tensor(sentence_indices, dtype=torch.long))


if '<unk>' not in word_to_ix:
    word_to_ix['<unk>'] = len(word_to_ix)


y = []
for tags_for_sentence in tags:
    tag_indices = [tag_to_ix[tag] for tag in tags_for_sentence]
    y.append(torch.tensor(tag_indices, dtype=torch.long))

# Define hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 128
DROPOUT = 0.33
LEARNING_RATE = 0.1
NUM_EPOCHS = 20


class BLSTM(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout):
        super(BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden1tag = nn.Linear(hidden_dim, hidden_dim // 2)
        self.hidden2tag = nn.Linear(hidden_dim // 2, len(tag_to_ix))
        self.dropout = nn.Dropout(p=0.33)
        self.activation = nn.ELU()

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden1tag(lstm_out)
        tag_scores = self.activation(tag_space)
        tag_space1 = self.hidden2tag(tag_scores)
        return tag_space1

# Instantiate the model
model1 = BLSTM(vocab_size=len(word_to_ix), tag_to_ix=tag_to_ix, 
              embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model1.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch in range(NUM_EPOCHS):
    for i in range(len(X_train)):
        sentence = X_train[i]
        tags = y[i]
        
        # Clear accumulated gradients
        model1.zero_grad()
        
        # Forward pass
        tag_scores = model1(sentence)
        
        # Calculate loss and perform backpropagation
        loss = loss_function(tag_scores, tags)
        loss.backward()
        optimizer.step()

    # Print epoch and loss
    print('epoch [{}/{}] ===> loss: {:.4f}'.format(epoch+1, NUM_EPOCHS, loss.item()))

torch.save(model1, 'blstm1.pt')


with open('data/dev', 'r') as f:
    dev_data = f.readlines()
    
for i in range(len(dev_data)):
    dev_data[i] = dev_data[i].split()

dev_sentences = []
current_sentence = []
for line in dev_data:
    if len(line) == 0:
        dev_sentences.append(current_sentence)
        current_sentence = []
    else:
        current_sentence.append(line[1])
dev_sentences.append(current_sentence)

X_dev = []
for sentence in dev_sentences:
    sentence_indices = [word_to_ix.get(word, word_to_ix['<unk>']) for word in sentence]
    X_dev.append(torch.tensor(sentence_indices, dtype=torch.long))

model1.eval()
with torch.no_grad():
    dev_tag_scores = []
    for sentence in X_dev:
        tag_scores = model1(sentence)
        dev_tag_scores.append(tag_scores)

dev_predicted_tags = []
for tag_scores in dev_tag_scores:
    _, predicted_tags = torch.max(tag_scores, dim=1)
    dev_predicted_tags.append([list(tag_to_ix.keys())[i] for i in predicted_tags.tolist()])


total = 0
correct = 0

for pred_tags, line in zip(dev_predicted_tags, dev_data):
    if len(line) > 0:
        total += 1
        if pred_tags[-1] == line[-1]:
            correct += 1

accuracy = correct / total

dev_true_tags = []
current_tags = []
for line in dev_data:
    if len(line) == 0:
        dev_true_tags.append(current_tags)
        current_tags = []
    else:
        current_tags.append(line[2])
dev_true_tags.append(current_tags)

dev_predicted_flat = [tag for tags in dev_predicted_tags for tag in tags]
dev_true_flat = [tag for tags in dev_true_tags for tag in tags]

y_dev = []
current_sentence = []
for line in dev_data:
    if len(line) == 0:
        y_dev.append(current_sentence)
        current_sentence = []
    else:
        current_sentence.append(line[2])
y_dev.append(current_sentence)

dev_predicted_tags_flatten = [tag for sentence_tags in dev_predicted_tags for tag in sentence_tags]
y_dev_flat = [tag for sentence_tags in y_dev for tag in sentence_tags]

Precision = precision_score(y_dev_flat, dev_predicted_tags_flatten, average='weighted')
Recall = recall_score(y_dev_flat, dev_predicted_tags_flatten, average='weighted')
F1_score = f1_score(y_dev_flat, dev_predicted_tags_flatten, average='weighted')

y_dev = []
current_sentence = []
for line in dev_data:
    if len(line) == 0:
        y_dev.append(current_sentence)
        current_sentence = []
    else:
        current_sentence.append(line[2])
y_dev.append(current_sentence)

# Flatten dev_predicted_tags and y_dev
dev_predicted_tags_flatten = [tag for sentence_tags in dev_predicted_tags for tag in sentence_tags]
y_dev_flat = [tag for sentence_tags in y_dev for tag in sentence_tags]

Precision = precision_score(y_dev_flat, dev_predicted_tags_flatten, average='macro')
Recall = recall_score(y_dev_flat, dev_predicted_tags_flatten, average='macro')
F1_score = f1_score(y_dev_flat, dev_predicted_tags_flatten, average='macro')

with open('dev1.out', 'w') as f:
    i = 0
    for index in range(len(dev_data)):
        line = dev_data[index]
        if not line:
            f.write('\n')
        else:
            f.write(f"{line[0]} {line[1]} {dev_predicted_tags_flatten[i]}\n")
            i += 1

with open('data/test', 'r') as file:
    test_datas = file.readlines()
    
for i in range(len(test_datas)):
    test_datas[i] = test_datas[i].split()

test_sentences = []
curr_sentence = []
for line in test_datas:
    if len(line) == 0:
        test_sentences.append(curr_sentence)
        curr_sentence = []
    else:
        curr_sentence.append(line[1])
test_sentences.append(curr_sentence)

X_test = []
for sentence in test_sentences:
    sentence_indices = [word_to_ix.get(word, word_to_ix['<unk>']) for word in sentence]
    X_test.append(torch.tensor(sentence_indices, dtype=torch.long))


model1.eval()
with torch.no_grad():
    test_tag_scores = []
    for sentence in X_test:
        tag_scores = model1(sentence)
        test_tag_scores.append(tag_scores)

test_predicted_tags = []
for tag_scores in test_tag_scores:
    _, predicted_tags = torch.max(tag_scores, dim=1)
    test_predicted_tags.append([list(tag_to_ix.keys())[i] for i in predicted_tags.tolist()])

test_predicted_tags_flat = [tag for sentence_tags in test_predicted_tags for tag in sentence_tags]


with open('test1.out', 'w') as f:
    i = 0
    for index in range(len(test_datas)):
        line = test_datas[index]
        if not line:
            f.write('\n')
        else:
            f.write(f"{line[0]} {line[1]} {test_predicted_tags_flat[i]}\n")
            i += 1

word_to_vec = {}
with open('glove.6B.100d', 'r') as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        vec = np.array(parts[1:], dtype=np.float32)
        word_to_vec[word] = vec


if '<pad>' not in word_to_ix:
    word_to_ix['<pad>'] = len(word_to_ix)


embedding_dim = 100
num_words = len(word_to_ix)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_to_ix.items():
    if word.lower() in word_to_vec:
        embedding_matrix[i] = word_to_vec[word.lower()]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

# Define hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 128
DROPOUT = 0.33
NUM_EPOCHS = 20
VOCAB_SIZE = len(word_to_ix)
TAG_SIZE = len(tag_to_ix)

class GloveBLSTM(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim, dropout, word_to_embedding):
        super(GloveBLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_to_embedding = word_to_embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True, dropout=dropout, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, hidden_dim//2)
        self.dropout = nn.Dropout(p=0.33)
        self.activation = nn.ELU()
        self.hidden3tag = nn.Linear(hidden_dim//2, tag_size)

    def forward(self, sentence):
        embeddings = []
        for word in sentence:
            if word.lower() in self.word_to_embedding:
                embeddings.append(self.word_to_embedding[word.lower()])
            else:
                embeddings.append(np.random.normal(scale=0.6, size=self.embedding_dim))
        embeddings = torch.tensor(embeddings)
        embeddings = embeddings.type(torch.float32)
        embeddings = embeddings.unsqueeze(0)
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = lstm_out.squeeze(0)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.activation(tag_space)
        tag_scores_final = self.hidden3tag(tag_scores)
        return tag_scores_final


model2 = GloveBLSTM(VOCAB_SIZE, TAG_SIZE, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT, word_to_vec)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model2.parameters(), lr=0.1)

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for i in range(len(sentences)):
        sentence = sentences[i]
        tags = y[i]
        model2.zero_grad()
        outputs = model2(sentence)
        loss = loss_function(outputs, tags)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch [{epoch+1}/{NUM_EPOCHS}] ====> loss: {total_loss/len(sentences):.4f}")


dev_sentences = []
current_sentence = []
for line in dev_data:
    if len(line) == 0:
        dev_sentences.append(current_sentence)
        current_sentence = []
    else:
        current_sentence.append(line[1])
dev_sentences.append(current_sentence)


X_dev = []
for sentence in dev_sentences:
    sentence_indices = [word_to_ix.get(word, word_to_ix['<unk>']) for word in sentence]
    X_dev.append(torch.tensor(sentence_indices, dtype=torch.long))



model2.eval()
with torch.no_grad():
    dev_tag_scores = []
    for sentence in dev_sentences:
        tag_scores = model2(sentence)
        dev_tag_scores.append(tag_scores)



dev_predicted_tags = []
for tag_scores in dev_tag_scores:
    _, predicted_tags = torch.max(tag_scores, dim=1)
    dev_predicted_tags.append([list(tag_to_ix.keys())[i] for i in predicted_tags.tolist()])


total = 0
correct = 0

for pred_tags, line in zip(dev_predicted_tags, dev_data):
    if len(line) > 0:
        total += 1
        if pred_tags[-1] == line[-1]:
            correct += 1

accuracy = correct / total

dev_true_tags = []
current_tags = []
for line in dev_data:
    if len(line) == 0:
        dev_true_tags.append(current_tags)
        current_tags = []
    else:
        current_tags.append(line[2])
dev_true_tags.append(current_tags)

dev_predicted_flat = [tag for tags in dev_predicted_tags for tag in tags]
dev_true_flat = [tag for tags in dev_true_tags for tag in tags]


y_dev = []
current_sentence = []
for line in dev_data:
    if len(line) == 0:
        y_dev.append(current_sentence)
        current_sentence = []
    else:
        current_sentence.append(line[2])
y_dev.append(current_sentence)

dev_predicted_tags_flatten = [tag for sentence_tags in dev_predicted_tags for tag in sentence_tags]
y_dev_flat = [tag for sentence_tags in y_dev for tag in sentence_tags]

Precision = precision_score(y_dev_flat, dev_predicted_tags_flatten, average='macro')
Recall = recall_score(y_dev_flat, dev_predicted_tags_flatten, average='macro')
F1_score = f1_score(y_dev_flat, dev_predicted_tags_flatten, average='macro')

with open('dev2.out', 'w') as f:
    i = 0
    for index in range(len(dev_data)):
        line = dev_data[index]
        if not line:
            f.write('\n')
        else:
            f.write(f"{line[0]} {line[1]} {dev_predicted_tags_flatten[i]}\n")
            i += 1


test_sentences = []
curr_sentence = []

for line in test_datas:
    if len(line) == 0:
        test_sentences.append(curr_sentence)
        curr_sentence = []
    else:
        curr_sentence.append(line[1])
        
test_sentences.append(curr_sentence)


X_test = []
for sentence in test_sentences:
    sentence_indices = [word_to_ix.get(word, word_to_ix['<unk>']) for word in sentence]
    X_test.append(torch.tensor(sentence_indices, dtype=torch.long))


model2.eval()
with torch.no_grad():
    test_tag_scores = []
    for sentence in test_sentences:
        tag_scores = model2(sentence)
        test_tag_scores.append(tag_scores)


test_predicted_tags = []
for tag_scores in test_tag_scores:
    _, predicted_tags = torch.max(tag_scores, dim=1)
    test_predicted_tags.append([list(tag_to_ix.keys())[i] for i in predicted_tags.tolist()])

test_predicted_tags_flat = [tag for sentence_tags in test_predicted_tags for tag in sentence_tags]

torch.save(model2,'blstm2.pt')


with open('test2.out', 'w') as f:
    i = 0
    for index in range(len(test_datas)):
        line = test_datas[index]
        if not line:
            f.write('\n')
        else:
            f.write(f"{line[0]} {line[1]} {test_predicted_tags_flat[i]}\n")
            i += 1