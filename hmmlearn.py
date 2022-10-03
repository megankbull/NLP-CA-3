# python hmmlearn.py /path/to/input
from collections import Counter
import sys 
import random 
import math
import numpy as np
 
OUT_PATH = "hmmmodel.txt"

CORPUS = None
TAGS = None
T_MAT = None
E_MAT = None
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
         w = tok.replace(f"/{tag}", '')
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
         else:
            bi_tags = ('', tag)
            if bi_tags not in tag_bigram.keys():
                tag_bigram[bi_tags] = 0
         
         tag_bigram[bi_tags]+=1
         prev_tag = tag

   return tag_bigram, tag_counts, word_tag_counts

def build_transition_matrix(tag_bigram, tag_counts): 
   global TAGS, T_c
   n_tags = len(TAGS)
   t_mat = np.zeros((n_tags, n_tags-1), dtype=np.float64) # prev tag by cur tag 

   for i, tag1 in enumerate(TAGS): 
      for j, tag2 in enumerate(T_c): 
         if tag2=='': continue 
         bi_count = tag_bigram[(tag1, tag2)] if (tag1, tag2) in tag_bigram.keys() else 0
         t_mat[i, j] = bi_count / tag_counts[tag1]

   return t_mat / np.linalg.norm(t_mat)

def build_emission_matrix(tag_counts, word_tag_counts): 
   global TAGS, T_c, CORPUS

   n_tags = len(T_c)
   n_words = len(CORPUS)

   emission_mat = np.zeros((n_words, n_tags), dtype=np.float64)
   
   for i, tag in enumerate(T_c): 
      for j, word in enumerate(CORPUS): 
         t_count = word_tag_counts[word][tag] if tag in word_tag_counts[word].keys() else 0
         emission_mat[j, i] = t_count / tag_counts[tag]

   return emission_mat   

def train_model():
   global CORPUS, TAGS, E_MAT, T_MAT, T_c
   
   train_set = word_tuples(read_file())

   tag_bigram, tag_counts, word_tag_counts = get_tag_counts(train_set)

   CORPUS = list(set(word_tag_counts.keys()))
   TAGS = list(set(tag_counts.keys()))
   
   T_c = TAGS.copy()
   T_c.remove('')

   T_MAT = build_transition_matrix(tag_bigram, tag_counts)
   E_MAT = build_emission_matrix(tag_counts, word_tag_counts)

def save_model(): 
   global CORPUS, TAGS, E_MAT, T_MAT, T_c
   vocab=" ".join(CORPUS)
   tags = " ".join(TAGS)
   np.set_printoptions(threshold=sys.maxsize)
   e_mat_str = np.array2string(E_MAT)
   t_mat_str = np.array2string(T_MAT)

   with open(OUT_PATH, "w") as f: 
      f.write(f"corpus: {vocab}\n\n")
      f.write(f"tags: {tags}\n\n")

      f.write(f"Emission Matrix (dimensions: words x tags)\n\n{e_mat_str}\n\n")
      f.write(f"Transition Matrix (dimensions: tags+1 x tags)\n\n{t_mat_str}")

def main(): 

   train_model()
   save_model()

   return (0)

if __name__ == '__main__': 
   main()