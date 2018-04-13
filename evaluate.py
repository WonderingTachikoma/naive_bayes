import re
    
    
def evaluate(h, y, klass='all'):
    """
    Оценивает результаты по метрикам accuracy/precision/recall/F1
    :param h: список гипотетических ответов (ответы классификатора)
    :param y: список правильных ответов (из тестового корпуса)
    :return: кортеж (acc, p, r, f1), т.е. accuracy, precision, recall, F1
    В случае, если количество положительных ответов (tp + fp) для класса равно нулю,
    точность для данного класса тоже считается нулевой. 
    Следовательно, precision + recall для класса тоже равно нулю,
    поэтому мы считаем F-меру для данного класса также нулевой.
    
    """
    assert len(h) == len(y), Exception('The length of h ({}) and y ({}) should match'.format(len(h), len(y)))
    len_y = len(y)
    if klass == 'all':
        classes = set(y)
        class_dist = {c: y.count(c) / len_y for c in classes}
    else:
        classes = (klass, )
        class_dist = {c: 1 for c in classes}
    
    results = {'accuracy': 0, 'precision': 0, 'recall': 0, 'F1':0}
    
    
    for c in classes:

        tp = set()
        fp = set()
        tn = set()
        fn = set()

        for i, h_i in enumerate(h):
            y_i = y[i]
            if h_i == c:
                if y_i == c:
                    tp.add((i, h_i))
                else:
                    fp.add((i, h_i))
            else:
                if y_i == c:
                    fn.add((i, h_i))
                else:
                    tn.add((i, h_i))

        acc = len(tp | tn) / len(tp | tn | fp | fn)
        p_denom = len(tp | fp)
        if p_denom:
            p = len(tp) / p_denom
        else:
            p = 0
        r = len(tp) / len(tp | fn)
        
        results['accuracy'] += acc * class_dist[c]
        results['precision'] += p * class_dist[c]
        results['recall'] += r * class_dist[c]
        f1_denom = p + r
        if f1_denom:
            results['F1'] += (2 * p * r / f1_denom) * class_dist[c]
        else:
            results['F1'] += 0
    
    micro = len([True for i in range(len(y)) if h[i] == y[i]]) / len(y)
    
    return (micro,
            round(results['accuracy'], 3),
            round(results['precision'], 3),
            round(results['recall'], 3),
            round(results['F1'], 3))
    
    
def main():
    for h, y in ((('pos', 'neg', 'pos', 'pos', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos'),
                  ('pos', 'neg', 'pos', 'neg', 'neg', 'neg', 'neg', 'neg', 'pos', 'neg')),
                 ((1, 1, 0, 0, 1, 1, 0, 0, 1, 1), (1, 1, 1, 1, 1, 0, 0, 0, 0, 0)),
                 ((1, 1, 0, 0, 0, 0, 0, 0, 1, 1), (1, 1, 1, 1, 1, 0, 0, 0, 0, 0)),
                 ((1, 0, 0, 0, 0, 0, 0, 0, 1, 1), (1, 1, 0, 0, 0, 0, 0, 0, 0, 0)),
                 ((1, 1, 1, 1, 1, 1, 0, 0, 1, 1), (1, 1, 0, 0, 0, 0, 0, 0, 0, 0)),
                 ((1, 1, 2, 2, 3, 3, 1, 2, 3, 3), (1, 2, 3, 1, 2, 3, 1, 2, 3, 1))
                 ):
        print('-' * 40)
        print('h = ', h, '\n', 'y = ', y, sep='', end='\n\n')
        print('class = {}: \nacc {} \nprec {} \nrec {} \nf1 {}'.
              format(y[0], *evaluate(h, y, klass=y[0])), end='\n\n')
        print('all classes:\nacc {} \nprec {} \nrec {} \nf1 {}'.
              format(*evaluate(h, y)), end='\n\n')
    input('Press Enter to exit')
    
if __name__ == '__main__':
    main()
