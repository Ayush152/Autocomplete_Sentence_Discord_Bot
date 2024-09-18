import string
import random
import math
import os
import dotenv
from collections import defaultdict, Counter

dotenv.load_dotenv()
DATA = os.getenv('TRAINING_DATA')

uni_cnts = defaultdict(int)
bi_cnts = defaultdict(int)
tri_cnts = defaultdict(int)

vocabulary_size = 0

def read_data(fp):
    with open(fp, 'r') as f:
        return f.read()

def tokenize(txt, freq_thresh=0):
    txt = txt.lower()

    txt = txt.translate(str.maketrans("", "", string.punctuation))

    tokens = txt.split()

    if freq_thresh > 0:
        token_cnt = Counter(tokens)
        tokens = [token for token in tokens if token_cnt[token] >= freq_thresh]
        
    return tokens

def build_ngram(tokens):
    global vocabulary_size

    for i in range(len(tokens)):
        uni_cnts[tokens[i]] += 1
        if(i > 0):
            bi_cnts[(tokens[i-1], tokens[i])] += 1
        if(i > 1):
            tri_cnts[(tokens[i-2], tokens[i-1], tokens[i])] += 1

    vocabulary_size = len(uni_cnts)

def uni_prob(w):
    return (uni_cnts[w] + 1) / (sum(uni_cnts.values()) + vocabulary_size)

def bi_prob(w1, w2):
    bi_cnt = bi_cnts[(w1, w2)]
    uni_cnt = uni_cnts[w1]

    if uni_cnt == 0:
        return uni_prob(w2)
    
    return (bi_cnt + 1) / (uni_cnt + vocabulary_size)

def tri_prob(w1, w2, w3):
    bi_cnt = bi_cnts[(w1, w2)]
    tri_cnt = tri_cnts[(w1, w2, w3)]

    if bi_cnt == 0:
        return uni_prob(w3)
    
    if tri_cnt > 0:
        return (tri_cnt + 1) / (bi_cnt + vocabulary_size)
    
    return bi_prob(w2, w3)

def weighted_nearest_nbr(word, prob):
    return random.choices(word, prob, k=1)[0]

def gen_bigram(tokens, n):
    curr = tokens[-1]
    res = []

    while len(res) < n:
        next_candidate = [(w1, w2) for (w1, w2) in bi_cnts if w1 == curr]

        if not next_candidate:
            break

        words, counts = zip(*[(w2, bi_cnts[(w1, w2)]) for w1, w2 in next_candidate])
        probs = [cnt/sum(counts) for cnt in counts]

        curr = weighted_nearest_nbr(words, probs)
        res.append(curr)

    return res

def gen_trigram(tokens, n):
    if len(tokens) < 2:
        return gen_bigram(tokens, n)
    
    curr_bi = (tokens[-2], tokens[-1])
    res = []

    while len(res) < n:
        next_candidate = [(w1, w2, w3) for (w1, w2, w3) in tri_cnts if (w1, w2)==curr_bi]

        if not next_candidate:
            return gen_bigram(tokens + res, n - len(res))
        
        words, counts = zip(*[(w3, tri_cnts[(w1, w2, w3)]) for w1, w2, w3 in next_candidate])
        probs = [cnt/sum(counts) for cnt in counts]

        next_word = weighted_nearest_nbr(words, probs)
        res.append(next_word)

        curr_bi = (curr_bi[1], next_word)

    return res

def log_bi_prob(words):
    log_prob = 0.0

    for i in range(1, len(words)):
        prob = bi_prob(words[i-1], words[i])
        log_prob += math.log(max(prob, 1e-10))

    return log_prob

def log_tri_prob(words):
    log_prob = 0.0

    for i in range(len(words)):
        if(i < 2):
            continue

        prob = tri_prob(words[i-2], words[i-1], words[i])
        log_prob += math.log(max(prob, 1e-10))

    return log_prob

def prepare_ngram():
    training_data = read_data(DATA)

    tokens = tokenize(training_data, 2)
    print('Training data tokenized!')

    build_ngram(tokens)
    print('N-gram models built!')

def get_response(user_input):
    input_tokens = tokenize(user_input)
    print(input_tokens)

    predicted_bi_words = gen_bigram(input_tokens, 10)
    log_prob_bigram = log_bi_prob(input_tokens + predicted_bi_words)

    predicted_tri_words = gen_trigram(input_tokens, 10)
    log_prob_trigram = log_tri_prob(input_tokens + predicted_tri_words)

    return (
        f"Bigram Result:\n{user_input} {' '.join(predicted_bi_words)} \n(Log Probability: {log_prob_bigram})\n"
        f"Trigram Result:\n{user_input} {' '.join(predicted_tri_words)} \n(Log Probability: {log_prob_trigram})"
    )