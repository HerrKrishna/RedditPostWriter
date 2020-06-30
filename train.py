import torch
import sys
from torch import nn
import dataset_reader
from model import Model


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def train(model: nn.Module,
          data_directory: str,
          learning_rate: float,
          batch_size: int = 10,
          num_epochs: int = 10,
          val_freq: int = 100,
          summary_freq: int = 10):

    model.apply(init_weights)
    model.train()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    batch_no = 0
    for epoch in range(num_epochs):
        for batch in dataset_reader.get_batches(data_directory, 'train', batch_size):
            batch_input = torch.tensor(batch[:, :-1])
            batch_labels = torch.tensor(batch[:, 1:])
            logits = model(batch_input)
            seq_len = logits.size()[1]
            logits = logits.view(batch_size * seq_len, vocab_size)
            batch_labels = batch_labels.flatten()
            loss = cross_entropy(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_no % summary_freq == 0:
                total_count = batch_no*batch_size
                print("Epoch: {}\tStep: {}\tExamples Seen: {}\tLoss: {} ".format(epoch, batch_no, total_count, loss))

            if batch_no % val_freq == 0 and batch_no != 0:
                model.eval()
                total_val_loss = 0
                count = 0
                print('Validation:')
                for val_batch in dataset_reader.get_batches(data_directory, 'dev', batch_size):
                    print(count)
                    val_batch_input = torch.tensor(val_batch[:, :-1])
                    val_batch_labels = torch.tensor(val_batch[:, 1:])
                    val_logits = model(val_batch_input)
                    seq_len = val_logits.size()[1]
                    val_logits = val_logits.view(batch_size * seq_len, vocab_size)
                    val_batch_labels = val_batch_labels.flatten()
                    val_loss = cross_entropy(val_logits, val_batch_labels)
                    total_val_loss += val_loss
                    count += 1

                average_val_loss = total_val_loss/count
                print("Average val loss: {}: {:.2f}".format(batch_no, average_val_loss))
                model.train()

            batch_no += 1






if __name__=='__main__':

    data_directory = sys.argv[1]
    vocab_size = dataset_reader.get_vocab_size(data_directory)
    model = Model(vocab_size=vocab_size,
                  hidden_size=300,
                  dropout=0.1)

    train(model=model, data_directory=data_directory, learning_rate=0.1, val_freq=10)
