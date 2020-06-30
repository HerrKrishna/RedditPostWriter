import torch
import sys
import yaml
from torch import nn
from torch.nn import functional as F
from model import Model
from preprocessing.convertTextIds import text2id, id2text
from dataset_reader import get_vocab_size, get_vocab


def generate(model: nn.Module, vocab: list, input_text: str = '',):

    start_sequence = text2id(input_text, vocab, add_bos=True)
    start_sequence = torch.tensor(start_sequence)
    start_sequence = start_sequence[None, :]

    last_char = start_sequence[0, -1]
    model.eval()
    while int(last_char) != 1:

        model_out = model(start_sequence)
        model_out = F.log_softmax(model_out, -1)
        model_out = torch.argmax(model_out, -1)
        last_char = model_out[0, -1]
        start_sequence = torch.cat((start_sequence, last_char), -1)

    output_text = id2text(start_sequence.squeeze().tolist(), vocab)

    return output_text


if __name__=='__main__':

    print('####################################')
    print('Welcome to RedditPostWriter')
    print('####################################')
    print('\n\n')
    model_path = sys.argv[1]
    with open(model_path + '/config.yaml' ) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    input_text = sys.argv[2]
    vocab_size = get_vocab_size(model_path)
    model = Model(vocab_size=vocab_size, **config['model'])
    print('Your input was: ' + input_text)
    print('\n')
    model = Model()
    vocab = get_vocab(model_path + '/vocab.txt')
    output_text = generate(model=model, vocab=vocab, input_text=input_text)
    print('Output: ')
    print(output_text)
