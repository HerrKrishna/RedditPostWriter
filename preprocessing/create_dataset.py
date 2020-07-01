from convokit import Corpus, download
import sys
import random

if __name__=='__main__':

    corpus_name = sys.argv[1]
    output_filename = sys.argv[2]
    corpus = Corpus(filename=download(corpus_name))

    char_list = ['<sos>', '<eos>', '<pad>', '<unk>']
    sequences = []
    for convo in corpus.iter_conversations():
        title = convo.meta['title']
        text = convo.get_utterance(convo.get_chronological_utterance_list()[0].conversation_id).text
        if text == '' or text == '[deleted]' or text == '[removed]':
            continue
        else:
            post = title + '\t' + text
            post = post.replace('\n', '').lower()
            sequence = ''
            for character in post:
                if character in char_list:
                    sequence += str(char_list.index(character)) + ' '
                else:
                    char_list.append(character)
                    sequence += str(len(char_list) - 1) + ' '
            sequences.append(sequence[:-1])

    random.shuffle(sequences)
    num_train_examples = (len(sequences) // 10) * 8
    num_dev_examples = len(sequences) // 10

    train_split = sequences[:num_train_examples]
    dev_split = sequences[num_train_examples:num_train_examples + num_dev_examples]
    test_split = sequences[num_train_examples + num_dev_examples:]

    train_split = sorted(train_split, key=len)
    dev_split = sorted(dev_split, key=len)
    test_split = sorted(test_split, key=len)

    with open(output_filename + '/vocab.txt', 'w', encoding='utf-8') as f:
        for character in char_list:
            f.write(character + '\n')

    with open(output_filename + '/train.txt', 'w', encoding='utf-8') as f:
        for sequence in train_split:
            f.write(sequence + '\n')

    with open(output_filename + '/dev.txt', 'w', encoding='utf-8') as f:
        for sequence in dev_split:
            f.write(sequence + '\n')

    with open(output_filename + '/test.txt', 'w', encoding='utf-8') as f:
        for sequence in test_split:
            f.write(sequence + '\n')


