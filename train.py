import torch
import sys
import yaml
from torch import nn
import dataset_reader
from model import Model
import generate


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def train(model: nn.Module,
          data_dir: str,
          savePath: str,
          learning_rate: float,
          batch_size: int = 10,
          num_epochs: int = 10,
          val_freq: int = 100,
          summary_freq: int = 10,
          max_len: int = 1500):
    
    device = model.device
    model.apply(init_weights)
    model.train()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    batch_no = 0
    best_checkpoint = None
    for epoch in range(num_epochs):
        for batch in dataset_reader.get_batches(data_dir, 'train', batch_size, max_len):
            batch_input = torch.tensor(batch[:, :-1])
            logits, lstm_out = model(batch_input)
            del batch_input
            seq_len = logits.size()[1]
            logits = logits.view(batch_size * seq_len, vocab_size)
            batch_labels = torch.tensor(batch[:, 1:])
            batch_labels = batch_labels.flatten()
            batch_labels = batch_labels.to(device)
            loss = cross_entropy(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del batch_labels
            del logits
            if batch_no % summary_freq == 0:
                total_count = batch_no*batch_size
                print("Epoch: {}\tStep: {}\tExamples Seen: {}\tLoss: {} ".format(epoch, batch_no, total_count, loss))
            del loss
            if batch_no % val_freq == 0 and batch_no != 0:
                with torch.no_grad():
                    model.eval()
                    total_val_loss = 0
                    count = 0
                    print('Validation:')
                    for val_batch in dataset_reader.get_batches(data_dir, 'dev', batch_size, max_len):
                        val_batch_input = torch.tensor(val_batch[:, :-1])
                        val_logits, lstm_out = model(val_batch_input)
                        del val_batch_input
                        seq_len = val_logits.size()[1]
                        val_logits = val_logits.view(batch_size * seq_len, vocab_size)
                        val_batch_labels = torch.tensor(val_batch[:, 1:])
                        val_batch_labels = val_batch_labels.flatten()
                        val_batch_labels = val_batch_labels.to(device)
                        val_loss = cross_entropy(val_logits, val_batch_labels)
                        total_val_loss += val_loss
                        del val_logits
                        del val_batch_labels
                        count += 1

                average_val_loss = total_val_loss/count
                print("Average val loss: {}: {:.2f}".format(batch_no, average_val_loss))
                if best_checkpoint is None:
                    print('New best checkpoint. Saving ...')
                    torch.save(model.state_dict(), savePath + '/model_state_dict')
                    best_checkpoint = average_val_loss 

                elif average_val_loss < best_checkpoint:
                    best_checkpoint = average_val_loss
                    print('New best checkpoint. Saving ...')
                    torch.save(model.state_dict(), savePath + '/model_state_dict')

                model.train()

            batch_no += 1
        print('')
        print('---------------------------------------------------------------------------------------')
        print('Epoch over. Generating a sample:')
        print(generate.generate(model=model, vocab=vocab, input_text=''))
        print('----------------------------------------------------------------------------------------')
        print('')
        model.train()

if __name__=='__main__':

    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    config_path = sys.argv[1]
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    data_dir = config['storage']['data_dir']
    savePath = config['storage']['savePath']
    vocab_size = dataset_reader.get_vocab_size(data_dir)
    vocab = dataset_reader.get_vocab(data_dir + '/vocab.txt')
    dataset_reader.write_vocab(savePath + '/vocab.txt', vocab=vocab)

    with open(savePath + '/config.yaml', 'w') as f:
        yaml.dump(config, f)

    model = Model(device=device, vocab_size=vocab_size, **config['model'])
    model.to(device=device)
    train(model=model, **config['training'])
