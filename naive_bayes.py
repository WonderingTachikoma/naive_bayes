#! python3

from collections import Counter
import math
import random
import re
import time
from evaluate import evaluate
    
    
def extract_features(text, features):
    """Return feature vector for text."""
    tokens = re.findall('\w+', text.lower())
    d = {f: 0 for f in features}
    for t in tokens:
        if t in d:
            d[t] += 1
    vec = []
    for f in features:
        vec.append(d[f])
    return vec
    
    
def add1_sm(counts):
    """Simple Add-1 smoothing"""
    new = []
    for f in counts:
        new.append(f + 1)
    return new
    

def train_nbc(features, corpora, smoothing=True):
    print('training NBC')
    f_set = set(features)
    Params = []  # list of lists (matrix): param vector for each class in order
    for i, corpus in enumerate(corpora):
        counts = extract_features(corpus, features)
        if smoothing:
            counts = add1_sm(counts)
        N = sum(counts)
        param_vec = []
        for f in counts:
            param_vec.append(f / N)
        Params.append(param_vec)
    print('done training')
    return Params
    
    
def classify_prod(text, features, classes, priors,
             Params, legend=False):
    results = []
    for i, cla in enumerate(classes):
        prod = priors[i]
        feature_vec = extract_features(text, features)
        for j, f_val in enumerate(feature_vec):
            p_val = Params[i][j]
            if legend:
                print('feature:', features[j],
                      'f_val:', f_val,
                      'p_val:', p_val)
            prod *= p_val ** f_val
        results.append(prod)
    max_prod = max(results)
    c_i = results.index(max_prod)
    if legend:
        print('classes:', classes)
        print('results:', results)
        print('choosing', classes[c_i])
    return classes[c_i]
    
    
def classify_log(text, features, classes, priors,
             Params, legend=False):
    results = []
    for i, cla in enumerate(classes):
        res = math.log(priors[i])
        feature_vec = extract_features(text, features)
        for j, f_val in enumerate(feature_vec):
            p_val = Params[i][j]
            if legend:
                print('feature:', features[j],
                      'f_val:', f_val,
                      'p_val:', p_val)
            res += math.log(p_val ** f_val)
        results.append(res)
    max_prod = max(results)
    c_i = results.index(max_prod)
    if legend:
        print('classes:', classes)
        print('results:', results)
        print('choosing', classes[c_i])
    return classes[c_i]
    
    
def classify_prior(text, features, classes, priors,
             Params, legend=False):
    results = []
    for i, cla in enumerate(classes):
        res = math.log(priors[i])
        results.append(res)
    max_prod = max(results)
    c_i = results.index(max_prod)
    if legend:
        print('classes:', classes)
        print('results:', results)
        print('choosing', classes[c_i])
    return classes[c_i]

    
def classify_random(text, features, classes, priors,
             Params, legend=False):
    results = []
    for i, cla in enumerate(classes):
        res = random.random()
        results.append(res)
    max_prod = max(results)
    c_i = results.index(max_prod)
    if legend:
        print('classes:', classes)
        print('results:', results)
        print('choosing', classes[c_i])
    return classes[c_i]

    
def test(features, corpora, classes, priors, Params, legend=False, classify=classify_log):
    print('testing')
    h = []
    y = []
    counter = 1
    log = []
    for j, corp in enumerate(corpora):
        ground_truth = classes[j]
        for t in corp.split('\n'):
            h_i = classify(t, features, classes, priors, Params)
            if legend:
                log.append('#{} - {} - labeled as {} (expected {})'.format(counter, t, h_i, ground_truth))
            counter += 1
            h.append(h_i)
            y.append(ground_truth)
    micro, acc, p, r, f1 = [round(v, 4) for v in evaluate(h, y)]
    print('Micro: {} Acc: {}, Prc: {}, Rec: {}, F1: {}'.format(micro, acc, p, r, f1))
    s = 'Micro: {} Acc: {}, Prc: {}, Rec: {}, F1: {}'.format(micro, acc, p, r, f1)
    s += '\n\nLEMMATIZED\n'
    s += '\n\nNUMBER OF FEATURES:\n{}'.format(len(features))
    s += '\n\nFEATURES:\n{}'.format(features)
    s += '\n\nPRIORS:\n{}'.format(priors)
    s += '\n\nCLASSES:\n{}'.format(classes)
    s += '\n\nLOG:\n{}'.format('\n'.join(log))
    print(s, file=open('res {}.txt'.format(time.time()), 'w', encoding='utf-8'))
    # print(evaluate(h, y))
    return acc, p, r, f1
    
    
def main():
    #lines = open('train_lem.txt', encoding='utf-8').read().split('\n')
    lines = open('train.txt', encoding='utf-8').read().split('\n')
    corp_dict = {}
    for line in lines:
        q, c = line.split('\t')
        if c not in corp_dict:
            corp_dict[c] = []
        corp_dict[c].append(q)
    corpora = []
    classes = []
    for c, qs in corp_dict.items():
        classes.append(c)
        corpora.append('\n'.join(qs))
    features = []
    
    for corp in corpora:
        features.extend(re.findall('\w+', corp.lower()))
    ctr = Counter(features)
    '''
    # build word freq table
    order = sorted(ctr, key=ctr.get, reverse=True)
    lines = []
    for w in order:
        lines.append('{} {}'.format(w, ctr[w]))
    print('\n'.join(lines), file=open('word count.txt', 'w', encoding='utf-8'))
    '''
    #features = list(f for f in ctr if ctr[f] > 1)  # only words with freq > 1
    #features = list(f for f in ctr if 100 > ctr[f] > 1)  # only words with 100 > freq > 1
    #features = list(f for f in ctr if ctr[f] > 1 and len(f) > 2)  # only words with freq > 1 min 3 characters long
    #features = list(ctr)  # all word features
    #features = eval(open('words by mean ppmi.txt', encoding='utf-8').read())  # mean ppmi
    #features = eval(open('words by top ppmi.txt', encoding='utf-8').read())  # top ppmi
    features = eval(open('words by top ppmi no lem.txt', encoding='utf-8').read())  # top ppmi
    features = features[:4000]
    
    Params = train_nbc(features, corpora)
    
    total = sum(len(corp_dict[c]) for c in corp_dict)
    #priors = [len(corp_dict[c]) / total for c in classes]  # true priors
    priors = [1 / len(classes) for c in classes]  # equal priors
    
    #t_lines = open('test_lem.txt', encoding='utf-8').read().split('\n')
    t_lines = open('test.txt', encoding='utf-8').read().split('\n')
    t_corp_dict = {}
    for line in t_lines:
        q, c = line.split('\t')
        if c not in t_corp_dict:
            t_corp_dict[c] = []
        t_corp_dict[c].append(q)
    t_corpora = []
    for c in classes:
        t_corpora.append('\n'.join(t_corp_dict[c]))
    
    test(features, t_corpora, classes, priors, Params, legend=True)

    
if __name__ == '__main__':
    main()
