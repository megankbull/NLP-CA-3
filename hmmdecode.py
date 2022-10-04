# python hmmdecode.py /path/to/input
import sys 
import json 
import heapq
import operator
import numpy as np

OUT_PATH = "hmmoutput.txt"
MODEL_PATH = "hmmmodel.txt"

CORPUS = None
TAGS = None
N_UNIQ_TAG = None
T_MAT = None
E_MAT = None
T_c = None

def read_file(f_path=sys.argv[1]):
   with open(f_path, 'r') as f: 
      lines = f.readlines()

   return [line.strip().split(" ") for line in lines]

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

def viterbi_algo(word_seq):
   global CORPUS, TAGS, N_UNIQ_TAG, E_MAT, T_MAT, T_c
   
   states = []
   last_state = TAGS.index('')
   common_tags_p = heapq.nlargest(6, N_UNIQ_TAG.items(), key=operator.itemgetter(1))
   common_tags = {t for t,_ in common_tags_p}

   for i, w in enumerate(word_seq): 
      p = []
      unseen = False if w.lower() in set(CORPUS) else True
      c_i = CORPUS.index(w.lower()) if not unseen else 0 


      for j, tag in enumerate(T_c):
         
         e_p = 1 if unseen or i == 0 else E_MAT[c_i, j]
         
         if unseen and tag.lower() not in common_tags: t_p = 0
         else: t_p = T_MAT[last_state,j]

         p.append(e_p * t_p)

      p_max = max(p)
      best_tag_index = p.index(p_max)
      best_state = T_c[best_tag_index]
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
               f.write(f"{p_tup[0]}/{p_tup[1].upper()}")
            else: f.write(f" {p_tup[0]}/{p_tup[1].upper()}")
         f.write("\n")

def clean_matrix(matr_list):

   matr = []

   for l in matr_list:

      vals = [w.replace('[', '').strip() for w in l.split()]
      matr.append(np.fromiter(vals, dtype=np.float64))
      
   return np.asmatrix(matr)

def set_model(): 
   global CORPUS, TAGS, N_UNIQ_TAG, E_MAT, T_MAT, T_c

   with open(MODEL_PATH, "r") as f: 
      raw_params = f.read().split("\n\n")
      CORPUS = raw_params[1].strip().split(" ")
      N_UNIQ_TAG = json.loads(raw_params[3])
      TAGS = json.loads(raw_params[5])

      e_mat = raw_params[7].strip().split(']')[:-2]
      t_mat = raw_params[9].strip().split(']')[:-2]
   
   T_c = TAGS.copy()
   T_c.remove('')

   E_MAT = clean_matrix(e_mat)
   T_MAT = clean_matrix(t_mat)

def main(): 

   set_model()
   tag_input()

   return (0)

if __name__ == '__main__': 
   main()