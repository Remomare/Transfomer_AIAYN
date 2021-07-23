import os
import sys
import inspect
import codecs
import re
import copy
import warnings
from collections import defaultdict, Counter


def update_vocabulary(vocab, file_name, is_dict=False):

    with codecs.open(file_name, encoding='utf-8') as fobj:
        for i, line in enumerate(fobj):
            if is_dict:
                try:
                    word, count = line.strip('\r\n ').split(' ')
                except:
                    print('Failed reading vocabulary file at line {0}: {1}'.format(i, line))
                    sys.exit(1)
                vocab[word] += int(count)
            else:
                for word in line.strip('\r\n ').split(' '):
                    if word:
                        vocab[word] += 1
    return vocab


def update_pair_statistics(pair, changed, stats, indices):
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first+second
    for j, word, old_word, freq in changed:

        i = 0
        while True:
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            if i < len(old_word)-1 and old_word[i+1] == second:
                if i:
                    prev = old_word[i-1:i+1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word)-2:
                    if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second:
                        nex = old_word[i+1:i+3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            try:
                i = word.index(new_pair, i)
            except ValueError:
                break
            if i:
                prev = word[i-1:i+1]
                stats[prev] += freq
                indices[prev][j] += 1
            if i < len(word)-1 and word[i+1] != new_pair:
                nex = word[i:i+2]
                stats[nex] += freq
                indices[nex][j] += 1
            i += 1


def get_pair_statistics(vocab):

    stats = defaultdict(int)

    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            stats[prev_char, char] += freq
            indices[prev_char, char][i] += 1
            prev_char = char

    return stats, indices


def replace_pair(pair, vocab, indices):
    first, second = pair
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\','\\\\')
    changes = []
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split(' '))

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))

    return changes

def prune_stats(stats, big_stats, threshold):
    for item,freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq


def learn_bpe(infile_names, outfile_name, num_symbols, min_frequency=2, verbose=False, is_dict=False, total_symbols=False):
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    vocab = Counter()
    for f in infile_names:
        sys.stderr.write(f'Collecting vocab from {f}\n')
        vocab = update_vocabulary(vocab, f, is_dict)

    vocab = dict([(tuple(x[:-1])+(x[-1]+'</w>',) ,y) for (x,y) in vocab.items()])
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    stats, indices = get_pair_statistics(sorted_vocab)
    big_stats = copy.deepcopy(stats)

    if total_symbols:
        uniq_char_internal = set()
        uniq_char_final = set()
        for word in vocab:
            for char in word[:-1]:
                uniq_char_internal.add(char)
            uniq_char_final.add(word[-1])
        sys.stderr.write('Number of word-internal characters: {0}\n'.format(len(uniq_char_internal)))
        sys.stderr.write('Number of word-final characters: {0}\n'.format(len(uniq_char_final)))
        sys.stderr.write('Reducing number of merge operations by {0}\n'.format(len(uniq_char_internal) + len(uniq_char_final)))
        num_symbols -= len(uniq_char_internal) + len(uniq_char_final)


    sys.stderr.write(f'Write vocab file to {outfile_name}')
    with codecs.open(outfile_name, 'w', encoding='utf-8') as outfile:
        outfile.write('#version: 0.2\n')
        threshold = max(stats.values()) / 10
        for i in range(num_symbols):
            if stats:
                most_frequent = max(stats, key=lambda x: (stats[x], x))

            if not stats or (i and stats[most_frequent] < threshold):
                prune_stats(stats, big_stats, threshold)
                stats = copy.deepcopy(big_stats)
                most_frequent = max(stats, key=lambda x: (stats[x], x))
                threshold = stats[most_frequent] * i/(i+10000.0)
                prune_stats(stats, big_stats, threshold)

            if stats[most_frequent] < min_frequency:
                sys.stderr.write(f'no pair has frequency >= {min_frequency}. Stopping\n')
                break

            if verbose:
                sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(
                    i, most_frequent[0], most_frequent[1], stats[most_frequent]))
            outfile.write('{0} {1}\n'.format(*most_frequent))
            changes = replace_pair(most_frequent, sorted_vocab, indices)
            update_pair_statistics(most_frequent, changes, stats, indices)
            stats[most_frequent] = 0
            if not i % 100:
                prune_stats(stats, big_stats, threshold)