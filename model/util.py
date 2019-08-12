import re
import json


def conver_num_to_zero(string):
    return re.sub('\d', '0', string)


def load_data_file(filename, zero=True, lower=True):
    dataset = []
    with open(filename, 'r', encoding='utf8') as r:
        for line in r:
            line = json.loads(line.strip())
            if zero:
                line['sentText'] = conver_num_to_zero(line['sentText'])
            if lower:
                line['sentText'] = line['sentText'].lower()

            for i, rel in enumerate(line['relationMentions']):
                if lower:
                    line['relationMentions'][i]['em1Text'] = rel['em1Text'].lower()
                    line['relationMentions'][i]['em2Text'] = rel['em2Text'].lower()
                if zero:
                    line['relationMentions'][i]['em1Text'] = conver_num_to_zero(rel['em1Text'])
                    line['relationMentions'][i]['em2Text'] = conver_num_to_zero(rel['em2Text'])

            for i, entity in enumerate(line['entityMentions']):
                if lower:
                    line['entityMentions'][i]['text'] = entity['text'].lower()
                if zero:
                    line['entityMentions'][i]['text'] = conver_num_to_zero(entity['text'])
            dataset.append(line)
    return dataset


def get_vocab(datasets, zero=True, lower=True):
    vocab = set()
    type_vocab = set()
    for dataset in datasets:
        for data in dataset:
            sentence = data['sentText'].split(' ')
            for word in sentence:
                vocab.add(word)

    vocab = sorted(list(vocab))
    vocab = {word: i + 1 for i, word in enumerate(vocab)}
    vocab['[UNK]'] = 0
    return vocab


def conver_token_to_id(word, vocab):
    if word not in vocab:
        return 0

    return vocab[word]


def get_typeDict(datasets):
    type_vocab = set()
    for dataset in datasets:
        for data in dataset:
            for entity in data['entityMentions']:
                type_vocab.add(entity['label'])
    type_vocab = sorted(type_vocab)
    type_vocab = {type_: i + 1 for i, type_ in enumerate(type_vocab)}
    type_vocab['None'] = 0
    return type_vocab


def get_labelDict(datasets):
    label_vocab = set()
    for dataset in datasets:
        for data in dataset:
            for rel in data['relationMentions']:
                label_vocab.add(rel['label'])
    label_vocab = sorted(label_vocab)
    label_vocab = {label: i for i, label in enumerate(label_vocab)}
    return label_vocab


def get_positions(sentence, flag, position_max=None):
    # flag: 0 表示 entity1 在前；1 表示 entity2在前
    entity_position = []
    for i, word in enumerate(sentence):
        if word[:7] == '<START>':
            entity_position.append([i])
        if word[-5:] == '<END>':
            entity_position[-1].append(i + 1)
    assert len(entity_position) == 2, [entity_position, sentence]
    positions = []
    for start, end in entity_position:
        end = end - 1
        position = []
        for i in range(len(sentence)):
            if i < start:
                pos = start - i
                if position_max and pos > position_max:
                    pos = position_max
                position.append(pos)
            elif i > end:
                pos = i - end
                if position_max and pos > position_max:
                    pos = position_max
                position.append(pos)
            else:
                position.append(0)
        positions.append(position)
    assert len(positions) == 2
    if flag == 0:
        return [positions[0], positions[1]]
    else:
        return [positions[1], positions[0]]


def get_types(sentence, entity_to_type, type_dict):
    type_ = 'None'
    types = []
    for word in sentence:
        if word[:7] == '<START>' and word[-5:] == '<END>':
            types.append(entity_to_type[word])
        elif word[:7] == '<START>':
            type_ = entity_to_type[word]
            types.append(type_)
        elif word[-5:] == '<END>':
            types.append(type_)
            type_ = 'None'
        else:
            types.append(type_)
    types = [type_dict[t] for t in types]
    return types


def padding(batch, level=0, pad=0):
    if level == 0:
        max_len = max([len(b) for b in batch])
        for i in range(len(batch)):
            batch[i] = batch[i] + [pad] * (max_len - len(batch[i]))
        return batch
    if level == 1:
        max_len_0 = max([len(b) for b in batch])
        max_len_1 = max([max([len(b0) for b0 in b1]) for b1 in batch])

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                batch[i][j] += [pad] * (max_len_1 - len(batch[i][j]))
            batch[i] += [[pad] * max_len_1 for _ in range(max_len_0 - len(batch[i]))]
        return batch

