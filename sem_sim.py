from collections import Counter
import math
import pprint
import re


def prints(s):
    n = len(s)
    print()
    print('*' * n)
    print(s)
    print('*' * n)


def tokenize(text):
    return re.findall('\w+', text.lower())


def bag_of_words(text):
    return Counter(tokenize(text))


def get_total_counts(counts):
    """counts is a dict of dicts"""
    return sum([sum(vals.values()) for vals in counts.values()])


def add_2_sm(counts):
    """counts is a dict of dicts"""
    new = {}
    for w in counts:
        new[w] = {}
        for k, v in counts[w].items():
            new[w][k] = counts[w][k] + 2
    return new

    
def p_wc(word, context_word, counts, total_counts):
    return counts[word][context_word] / total_counts
    
    
def p_w(word, counts, total_counts):
    return sum(counts[word].values()) / total_counts
    
    
def p_c(context_word, counts, total_counts):
    return sum(freqs.get(context_word, 0) for freqs in counts.values()) / total_counts

    
def pmi(word, context_word, counts, total_counts):
    ans = (p_wc(word, context_word, counts, total_counts) /
          (p_w(word, counts, total_counts) * p_c(context_word, counts, total_counts)))
    if ans:
        return math.log2(ans)
    else:
        return 0
    
    
def ppmi(word, context_word, counts, total_counts):
    ans = pmi(word, context_word, counts, total_counts)
    return ans if ans > 0 else 0

    
def cos_sim_old(vec1, vec2):
    assert len(vec1) == len(vec2), 'Vector length must match'
    numerator = sum(vec1[i]*vec2[i] for i in range(len(vec1)))
    denominator = math.sqrt(sum(a ** 2 for a in vec1)) * math.sqrt(sum(b ** 2 for b in vec2))
    return numerator / denominator
    
    
def cos_sim(counts_dict1, counts_dict2):
    words = counts_dict1 | counts_dict2  # Counter supports |
    num = sum(counts_dict1[k] * counts_dict2[k] for k in words)
    denom = (math.sqrt(sum(counts_dict1[k] ** 2 for k in words)) * 
             math.sqrt(sum(counts_dict2[k] ** 2 for k in words)))
    return num / denom
    
    
def build_matrix():
    #lines = open('train_lem.txt', encoding='utf-8').read().split('\n')
    lines = open('Training.txt', encoding='utf-8').read().split('\n')
    vocab = []
    labels = []
    for line in lines:
        q, c = line.split('\t')
        words = re.findall('\w+', q.lower())
        vocab.extend(words)
        labels.append(c)
    vocab = set(vocab)
    labels = set(labels)
    
    counts = {w: {c: 0 for c in labels} for w in vocab}  # dict of dicts
    
    for line in lines:  # second pass, not pretty
        q, c = line.split('\t')
        words = re.findall('\w+', q.lower())
        for w in words:
            context = counts[w]
            context[c] += 1
            
    print(counts, file=open('counts matrix no lem raw.txt', 'w', encoding='utf-8'))
    counts2 = add_2_sm(counts)
    print(counts2, file=open('counts matrix no lem add 2.txt', 'w', encoding='utf-8'))
    
    total_counts = get_total_counts(counts)
    total_words = len(counts)
    i = 0
    
    print('working out PPMI')
    for w, contexts in counts.items():
        i += 1
        if not i % 100:
            print(i, '/', total_words)
        for c in contexts:
            counts[w][c] = ppmi(w, c, counts, total_counts)
    print('done with PPMI, saving')
    
    print(counts, file=open('counts matrix no lem PPMI.txt', 'w', encoding='utf-8'))
    
    
    total_counts = get_total_counts(counts2)
    total_words = len(counts2)
    i = 0
    
    print('working out PPMI with smoothing')
    for w, contexts in counts2.items():
        i += 1
        if not i % 100:
            print(i, '/', total_words)
        for c in contexts:
            counts2[w][c] = ppmi(w, c, counts2, total_counts)
    print('done with PPMI, saving')
    
    print(counts2, file=open('counts matrix no lem PPMI add 2.txt', 'w', encoding='utf-8'))
    
    
def rank_words():
    counts = eval(open('counts matrix no lem PPMI add 2.txt', 'r', encoding='utf-8').read())
    mean_ppmi = {}
    top_ppmi = {}
    for w, contexts in counts.items():
        vals = list(contexts.values())
        mean_ppmi[w] = sum(vals) / len(vals)
        top_ppmi[w] = max(vals)
    order_mean = sorted(mean_ppmi, key=mean_ppmi.get, reverse=True)
    order_top = sorted(top_ppmi, key=top_ppmi.get, reverse=True)
    print(order_mean, file=open('words by mean ppmi no lem.txt', 'w', encoding='utf-8'))
    print(order_top, file=open('words by top ppmi no lem.txt', 'w', encoding='utf-8'))
    
    
def main():
    #build_matrix()
    rank_words()
    
    
if __name__ == '__main__':
    main()
    
