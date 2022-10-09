# python hmmdecode.py /path/to/input
import sys 
import json 
import numpy as np

OUT_PATH = "hmmoutput.txt"
MODEL_PATH = "hmmmodel.txt"

CORPUS = None
TAGS = None
T_MAT = None
E_MAT = None
T_c = None
LOWER_CORP = None 
SET_CORP = None
COMMON_TAGS = None
SUFF_TAGS = None

def read_file(f_path=sys.argv[1]):
   with open(f_path, 'r') as f: 
      return [l.strip().split(" ") for l in f.readlines()]

def get_closest_index(word, last_state, prev_p, last_word=False): 
   global CORPUS, T_c, E_MAT, T_MAT
   boundary = T_c.index('/')

   try:
      return CORPUS.index(word)

   except ValueError: 
      indices = [i for i, vocab_w in enumerate(CORPUS) if word.lower() == vocab_w.lower()]
      cur_p = 0

      for loc in indices: 
         s = []
         for j, tag in enumerate(T_c): 
            if tag == '/': 
               s.append(0)
               continue
            e_p = E_MAT[loc, j]
            t_p = T_MAT[last_state, j] 
            last_tag_p = T_MAT[j, boundary] * 0.05 if last_word else 1

            s.append(e_p * t_p * prev_p * last_tag_p)

         state_p = max(s) 

         if state_p > cur_p: 
            cur_p = state_p 
            best_tag = T_c[s.index(cur_p)]

      return best_tag, cur_p

def suffix_tag_p(word, tag):
   global SUFF_TAGS

   if word[-1:] not in SUFF_TAGS.keys(): return 1
   if tag not in SUFF_TAGS[word[-1:]].keys(): return 0

   return SUFF_TAGS[word[-1:]][tag] / sum(SUFF_TAGS[word[-1:]].values())

def viterbi_algo(word_seq):
   global CORPUS, TAGS, E_MAT, T_MAT, T_c, LOWER_CORP, SET_CORP, COMMON_TAGS

   states = []
   last_state = TAGS.index('\\')
   boundary = T_c.index('/')
   prev_p = 1

   for i, w in enumerate(word_seq): 
      p = []
      last_word = i ==len(word_seq) - 2

      unseen = w not in SET_CORP and w.lower() not in LOWER_CORP
      c_i = 0 if unseen else get_closest_index(w, last_state, prev_p, last_word)

      if type(c_i) is tuple: 
         states.append(c_i[0])
         last_state = TAGS.index(c_i[0])
         prev_p = c_i[1]
         continue 

      for j, tag in enumerate(T_c):
         if tag == '/': 
            p.append(0)
            continue

         suff_p = suffix_tag_p(w, tag) if unseen else 1
         e_p = 1 if unseen else E_MAT[c_i, j]
         t_p = 0 if unseen and tag not in COMMON_TAGS else T_MAT[last_state, j]
         last_tag_p = T_MAT[j, boundary] * 0.05 if last_word else 1

         p.append(e_p * t_p * prev_p * suff_p * last_tag_p)

      p_max = max(p)
      best_state = T_c[p.index(p_max)]
      states.append(best_state)
      last_state = TAGS.index(best_state)

   return list(zip(word_seq, states))

def tag_input(): 
   lines = read_file()

   preds = [viterbi_algo(samp) for samp in lines]

   with open(OUT_PATH, 'w') as f: 
      for p_line in preds: 
         for i, p_tup in enumerate(p_line):
            if i == 0: 
               f.write(f"{p_tup[0]}/{p_tup[1]}")
            else: f.write(f" {p_tup[0]}/{p_tup[1]}")
         f.write("\n")

def clean_matrix(matr_list):

   matr = []

   for l in matr_list:

      vals = [w.replace('[', '').strip() for w in l.split()]
      matr.append(np.fromiter(vals, dtype=np.float64))
      
   return np.asmatrix(matr)

def set_model(): 
   global CORPUS, TAGS, T_c, LOWER_CORP, SET_CORP, COMMON_TAGS, SUFF_TAGS, E_MAT, T_MAT

   with open(MODEL_PATH, "r") as f: 
      raw_params = f.read().split("\n\n")
      CORPUS = raw_params[1].strip().split(" ")
      SUFF_TAGS = json.loads(raw_params[3])
      COMMON_TAGS = set(raw_params[5].strip().split(" "))
      TAGS = json.loads(raw_params[7])

      e_mat = raw_params[9].strip().split(']')[:-2]
      t_mat = raw_params[11].strip().split(']')[:-2]
   
   T_c = TAGS.copy()
   T_c.remove('\\')

   E_MAT = clean_matrix(e_mat)
   T_MAT = clean_matrix(t_mat)

   LOWER_CORP = {w.lower() for w in CORPUS}
   SET_CORP = set(CORPUS)

def main(): 

   set_model()
   tag_input()

   return (0)

if __name__ == '__main__': 
   main()