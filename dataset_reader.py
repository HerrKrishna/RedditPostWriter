import sys
import itertools
import numpy as np
import random
from preprocessing import convertTextIds

def get_batches(directory: str, split: str, batch_size: int, max_len: int, pseudo: bool = False):
    
    if not pseudo:
        with open(directory + '/' + split + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        sequences = [['0'] + line.strip('\n').split(' ') + ['1'] for line in lines]
        sequences = [[int(character) for character in sequence] for sequence in sequences]
        sequences = [seq for seq in sequences if len(seq) < max_len]

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            if len(batch) < batch_size:
                break
            #this converts batch to a np.array padded with zeros
            batch = np.array(list(itertools.zip_longest(*batch, fillvalue=2))).T
            yield batch

    else:
        for i in range(0, 500):
            batch = [[0] +[i for i in range(4, 20)]+ [1] for x in range(batch_size)]
            #print(batch)
            batch = np.array(batch)
            yield batch

def get_random_sequence(directory: str, split: str, max_len: int, pseudo: bool = False):
    if not pseudo:
        with open(directory + '/' + split + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        short_lines = []
        for line in lines:
            if len(line) < max_len:
                short_lines.append(line)
            else:
                break

        sampled_line = random.sample(short_lines, 1)[0]
        seq = ['0'] + sampled_line.strip('\n').split(' ') + ['1']
        seq = [int(character) for character in seq]
        seq = np.array(seq)
        return seq

def get_vocab(filename:str) -> list:

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip('\n') for line in lines]
    return lines


def write_vocab(filename:str , vocab: list):
    with open(filename, 'w', encoding='utf-8') as f:
        for char in vocab:
            f.write(char + '\n')


def get_vocab_size(directory: str) -> int:

    with open(directory + '/vocab.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return len(lines)

if __name__ == '__main__':

    directory = sys.argv[1]
    split = sys.argv[2]
    vocab = get_vocab(directory+ 'vocab.txt')
    print(vocab)
    for batch in get_batches(directory, split, 1, max_len=1000):
        ids = batch[0]
        text = convertTextIds.id2text(ids, vocab=vocab)
        print(text)
        print('.............................................................................\n')
