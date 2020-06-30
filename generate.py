import torch
from torch import nn
from torch.nn import functional as F
import sys
from model import Model
from preprocessing.convertTextIds import text2id, id2text
from dataset_reader import get_vocab


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
    hyperparams = sys.argv[1]
    input_text = sys.argv[2]
    vocab = get_vocab(vocab_path)
    print('Your input was: ' + input_text)
    print('\n')
    model = Model(**hyperparams)
    output_text = generate(model=model, vocab=vocab, input_text=input_text)
    print('Output: ')
    print(output_text)
