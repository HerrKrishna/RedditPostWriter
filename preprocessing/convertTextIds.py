def text2id(text: str, vocab: list, add_bos: bool = False, add_eos: bool = False) -> list:
    sequence = []
    for character in text:
        try:
            sequence.append(vocab.index(character))
        except ValueError:
            sequence.append(vocab[3])

    if add_bos:
        sequence = [0] + sequence
    if add_eos:
        sequence = sequence.append(1)

    return sequence


def id2text(ids: list, vocab: list) -> str:

    text = ''
    for id in ids:
        if id == 0:
            continue
        elif id == 1:
            break
        elif id == 2:
            continue

        text += vocab[id]

    return text
