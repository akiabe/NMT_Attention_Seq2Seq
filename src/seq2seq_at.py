import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random

# to print to tensorboard
from torch.utils.tensorboard import SummaryWriter

from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

# loading word
spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")

# create tokenizer function
def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

# define vocab form
german = Field(tokenize=tokenizer_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenizer_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

# build vocab
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size

        # embedding shape: (seq_length, N, embedding_size)
        embedding = self.dropout(self.embedding(x))

        # outputs shape: (seq_length, N, hidden_size)
        encoder_states, (hidden, cell) = self.rnn(embedding)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers, dropout=dropout)

        self.energy = nn.Linear(hidden_size*3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):
        # x shape: (N) where N is for batch size, we want it to be (1, N)
        # seq_length is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)

        # embedding shape: (1, N, embedding_size)
        embedding = self.dropout(self.embedding(x))

        sequence_length = encoder_states.shape[0]

        # h_reshaped: (seq_length, N, hidden_size*2)
        h_reshaped = hidden.repeat(sequence_length, 1, 1)

        # energy: (seq_length, N, 1)
        energy = self.relu(self.energy(torch.cat(h_reshaped, encoder_states), dim=2))

        # attention shape : (seq_length, N, 1)
        attention = self.softmax(energy)

        # attention shape: (seq_length, N, 1)
        attention = attention.permute(1, 2, 0)

        # encoder shape: (N, seq_length, hidden_size*2)
        encoder_states = encoder_states.permute(1, 0, 2)

        # (N, 1, hidden_size*2) --> (1, N, hidden_size*2)
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)

        # rnn_input: (1, N, hidden_size*2 + embedding_size)
        rnn_input = torch.cat((context_vector, embedding), dim=2)

        # outputs shape: (1, N, hidden_size)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        target_len = target.shape[0]
        batch_size = source.shape[1] # source shape: (target_len, batch_size)
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        encoder_states, hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # output shape: (N, english_vocab_size)
            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # define next input of decoder
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

### ready to to the training the model ###
# training hyperparameters
num_epochs = 5
learning_rate = 0.001
batch_size = 64

# model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# tensorbord
writer = SummaryWriter(f"runs/loss_plot")
step = 0

# define iterator
# to minimize number of padding, we group similar length of batch
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x:len(x.src),
    device=device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                      hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# stoi: stings to index
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to device
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # forward prop
        output = model(inp_data, target)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("training loss", loss, global_step=step)
        step += 1

        print(f"Epoch {epoch + 1}/{num_epochs}, Train loss: {loss.item():.4f}")

score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score*100:.2f}")