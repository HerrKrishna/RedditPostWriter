import torch
import sys
import yaml
from torch import nn
from torch.nn import functional as F
from model import Model
from preprocessing.convertTextIds import text2id, id2text
from dataset_reader import get_vocab_size, get_vocab
import numpy as np

def generate(model: nn.Module, vocab: list, input_text: str = '', max_len: int = 1500):

    start_sequence = text2id(input_text, vocab, add_bos=True)
    start_sequence = torch.tensor(start_sequence)
    start_sequence = start_sequence[None, :]

    last_char = start_sequence[0, -1]
    model.eval()
    seq_len = start_sequence.size()[1]
    while int(last_char) != 1 and seq_len < 300:
        model_out = model(start_sequence)
        model_out = F.log_softmax(model_out, -1)
        model_out = model_out.detach().numpy()[0, -1, :]
        sample = np.random.choice(len(vocab), p=np.exp(model_out))
        last_char = torch.tensor(sample).unsqueeze(0).unsqueeze(0)
        start_sequence = torch.cat((start_sequence, last_char), -1)
        seq_len = start_sequence.size()[1]
        if seq_len % 100 == 0:
            print(seq_len)

    output_text = id2text(start_sequence.squeeze().tolist(), vocab)

    return output_text


if __name__=='__main__':

    print('####################################')
    print('Welcome to RedditPostWriter')
    print('####################################')
    print('\n\n')

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

    model_path = sys.argv[1]
    with open(model_path + '/config.yaml' ) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    input_text = sys.argv[2]
    vocab_size = get_vocab_size(model_path)
    model = Model(device=device, vocab_size=vocab_size, **config['model'])
    model.load_state_dict(torch.load(model_path + '/model_state_dict', map_location=torch.device('cpu')))
    print('Your input was: ' + input_text)
    print('\n')

    vocab = get_vocab(model_path + '/vocab.txt')
    max_len = config['training']['max_len']
    output_text = generate(model=model, vocab=vocab, input_text=input_text, max_len=max_len)
    print('Output: ')
    print(output_text)
