# python hmmlearn.py /path/to/input
from collections import Counter
import sys 
import random 
import math
import numpy as np
import pandas as pd

# TODO:
# - smooth transition matrix 
# - prior probabilities of tags
#
# TRAINING: 
# - unseen words rely on transition probabilities
# - need smoothing to compensate for unseen transitions 

# output needs enough info for decode tag new data
# human readable to inspect model params  
OUT_PATH = "hmmmodel.txt"

CORPUS = None
TAGS = None
T_MAT = None
E_MAT = None
T_i = None
T_c = None

def read_file(f_path=sys.argv[1]):
   with open(f_path, 'r') as f: 
      lines = f.readlines()

   return [line.lower().strip().split(" ") for line in lines]

def word_tuples(lines): 
   tagged_tupes = []
   for line in lines: 
      line_tupes = []
      for token in line: 
         tok = token.strip()
         tag = tok.split("/")[-1]
         w = tok.removesuffix(f"/{tag}")
         line_tupes.append((w, tag))
      tagged_tupes.append(line_tupes)
   return tagged_tupes

def train_test_split(lines, test_frac=0.2): 

   tagged_tuples = word_tuples(lines)
   random.shuffle(tagged_tuples)

   num_states = len(tagged_tuples)

   test_len = math.floor(num_states * test_frac)
   test_set = tagged_tuples[:test_len]
   train_set = tagged_tuples[test_len:]

   return train_set, test_set

def get_tag_set(train_set, test=False): 
   
   return {tag for _, tag in train_set} if test else {tag for line in train_set for _, tag in line}

def get_vocab(train_set, test=False): 
   return [w for w, _ in train_set] if test else {w for line in train_set for w, _ in line}

def get_tag_counts(line_tuples): 
   tag_counts = Counter(t for l in line_tuples for _,t in l)
   tag_counts[''] = len(line_tuples)
   tag_bigram = {}
   word_tag_counts = {}

   for line in line_tuples: 
      prev_tag = None
      for word, tag in line: 
         if word not in word_tag_counts.keys(): 
            word_tag_counts[word] = {}
         if tag not in word_tag_counts[word].keys():
            word_tag_counts[word][tag] = 0
         word_tag_counts[word][tag] +=1

         if prev_tag is not None: 
            bi_tags = (prev_tag, tag)
            if bi_tags not in tag_bigram.keys(): 
               tag_bigram[bi_tags] = 0
         elif (bi_tags:= ('', tag)) not in tag_bigram.keys():
            tag_bigram[bi_tags] = 0
         
         tag_bigram[bi_tags]+=1
         prev_tag = tag

   return tag_bigram, tag_counts, word_tag_counts

def build_transition_matrix(tag_bigram, tag_counts): 
   global T_i, T_c
   n_tags = len(T_i)
   t_mat = np.zeros((n_tags, n_tags-1), dtype=np.float64) # prev tag by cur tag 

   for i, tag1 in enumerate(T_i): 
      for j, tag2 in enumerate(T_c): 
         if tag2=='': continue 
         bi_count = tag_bigram[(tag1, tag2)] if (tag1, tag2) in tag_bigram.keys() else 0
         t_mat[i, j] = bi_count / tag_counts[tag1]

   return t_mat / np.linalg.norm(t_mat)

def build_emission_matrix(tag_counts, word_tag_counts): 
   global T_i, T_c, CORPUS

   n_tags = len(T_c)
   n_words = len(CORPUS)

   emission_mat = np.zeros((n_words, n_tags), dtype=np.float64)
   
   for i, tag in enumerate(T_c): 
      for j, word in enumerate(CORPUS): 
         t_count = word_tag_counts[word][tag] if tag in word_tag_counts[word].keys() else 0
         emission_mat[j, i] = t_count / tag_counts[tag]

   return emission_mat   

def viterbi_algo(word_seq): 
   global CORPUS, TAGS, E_MAT, T_MAT, T_i
   states = []
   last_state = T_i.index('')

   for i, w in enumerate(word_seq): 
      p = []
      unseen = False

      try: 
         c_i = CORPUS.index(w)

      except ValueError: 
         unseen = True

      for j, tag in enumerate(T_c):
         e_p = 1 if unseen or i == 0 else E_MAT[c_i, j]
         t_p = T_MAT[last_state,j]

         p.append(e_p * t_p)

      p_max = max(p)
      best_tag_index = p.index(p_max)
      best_state = T_c[best_tag_index]
      states.append(best_state)
      last_state = T_i.index(best_state)

   return list(zip(word_seq, states))

def train_model(train_set):
   global CORPUS, TAGS, E_MAT, T_MAT, T_c, T_i

   tag_bigram, tag_counts, word_tag_counts = get_tag_counts(train_set)

   CORPUS = list(set(word_tag_counts.keys()))
   TAGS = list(set(tag_counts.keys()))
   
   T_i = TAGS
   T_c = TAGS.copy()
   T_c.remove('')

   T_MAT = build_transition_matrix(tag_bigram, tag_counts)
   E_MAT = build_emission_matrix(tag_counts, word_tag_counts)

def eval_model(test_set):

   random.seed(1234)
   rndom = [random.randint(1,len(test_set)) for _ in range(100)]

   test_run = [test_set[i] for i in rndom]

   test_tagged_words = [[tup[0] for tup in sent] for sent in test_run]

   preds = [viterbi_algo(samp) for samp in test_tagged_words]

   check = []
   total = 0

   for p_line, a_line in zip(preds, test_run): 
      for p_tup, a_tup in zip(p_line, a_line): 
         if p_tup == a_tup: check.append(p_tup)
         total +=1
      write_out(p_line, a_line)

   acc = len(check)/total
   print(f"Viterbi Algorithm Accuracy: {acc}")

def test_model(f_path=sys.argv[2], a_f_path=sys.argv[3]): 

   lines = read_file(f_path)
   a_tupes = word_tuples(read_file(a_f_path))

   preds = [viterbi_algo(samp) for samp in lines]

   check = []
   total = 0

   with open(OUT_PATH, 'w') as f: 
      for p_line, a_line in zip(preds, a_tupes): 
         for p_tup, a_tup in zip(p_line, a_line): 
            if p_tup == a_tup: check.append(p_tup)
            total +=1
            f.write(f"{p_tup[0]}\t{p_tup[1]}\t{a_tup[1]}\n")

   acc = len(check)/total
   print(f"Viterbi Algorithm Accuracy: {acc}")

# def write_out(tagged_seq, actual_seq, out_path=OUT_PATH): 
#    with open(out_path, 'w+') as f: 

#       for pred, actual in zip(tagged_seq, actual_seq): 
#          f.write(f"{pred[0]}\t{pred[1]}\t{actual[1]}\n")

def main(): 
   train = word_tuples(read_file())

   train_model(train)
   test_model()
   # t_df = pd.DataFrame(T_MAT, index=TAGS, columns=T_c)
   # print(t_df)
   return (0)

if __name__ == '__main__': 
   main()
