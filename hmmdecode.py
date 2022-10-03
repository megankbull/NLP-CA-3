# python hmmlearn.py /path/to/input
from collections import Counter
import sys 
import random 
import math
import numpy as np

OUT_PATH = "hmmoutput.txt"
MODEL_PATH = "hmmmodel.txt"

CORPUS = None
TAGS = None
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
   global CORPUS, TAGS, E_MAT, T_MAT, T_c
   states = []
   last_state = TAGS.index('')

   for i, w in enumerate(word_seq): 
      p = []
      unseen = False

      try: 
         c_i = CORPUS.index(w.lower())

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
   global CORPUS, TAGS, E_MAT, T_MAT, T_c

   with open(MODEL_PATH, "r") as f: 
      raw_params = f.read().split("\n\n")
      CORPUS = raw_params[0].strip().split(" ")[1:]
      TAGS = raw_params[1].strip().split(" ")[1:]
      e_mat = raw_params[3].strip().split(']')[:-2]
      t_mat = raw_params[5].strip().split(']')[:-2]
   
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