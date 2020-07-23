import torch
import sys
import yaml
import numpy as np
import dataset_reader
from model import Model
from torch.nn import functional as F
import matplotlib.pyplot as plt
from preprocessing import convertTextIds
from matplotlib import ticker


def plot(x_labels, y_labels, data, title = ''):
    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(data)
    plt.title(title)
    fig.colorbar(im)
    ax.set_xticklabels([''] + x_labels)
    ax.set_yticklabels([''] + y_labels)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

if __name__=='__main__':

    device = torch.device("cpu")
    model_path = sys.argv[1]
    with open(model_path + '/config.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    vocab_size = dataset_reader.get_vocab_size(model_path)
    model = Model(device=device, vocab_size=vocab_size, **config['model'])
    model.load_state_dict(torch.load(model_path + '/model_state_dict', map_location=torch.device('cpu')))
    vocab = dataset_reader.get_vocab(model_path + '/vocab.txt')

    data_dir = config['storage']['data_dir']
    input_seq = dataset_reader.get_random_sequence(directory=data_dir, split='dev', max_len=100)
    input_seq = torch.tensor(input_seq)
    input_seq = input_seq[None, :]
    model_out, lstm_out = model(input_seq)
    model_out = F.log_softmax(model_out, -1)
    model_out = model_out.detach().numpy()[0, :, :]
    model_out = np.exp(model_out)
    model_out = model_out[:, 4:25]
    model_out = np.transpose(model_out)

    neuron_activatons = lstm_out[0, :, :5]
    neuron_activatons = np.transpose(neuron_activatons.detach().numpy())

    x_labels = convertTextIds.id2text(input_seq[0, :].tolist(), vocab)
    y_labels = convertTextIds.id2text(list(range(4, 25)), vocab)
    x_labels = [''] + [char for char in x_labels]
    y_labels = [char for char in y_labels]

    plot(x_labels, y_labels, model_out)

    y_labels = list(range(5))
    plot(x_labels, y_labels, neuron_activatons)



