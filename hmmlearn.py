# python hmmlearn.py /path/to/input
import sys 
import json
import numpy as np

OUT_PATH = "hmmmodel.txt"

CORPUS = None
N_UNIQ_TAG = None
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

def get_tag_counts(line_tuples): 

   tag_counts = {'':  []}
   tag_bigram = {}
   word_tag_counts = {}

   for line in line_tuples: 
      prev_tag = None
      for word, tag in line: 
         if prev_tag is None: tag_counts[''].append(word)

         if tag not in tag_counts.keys(): tag_counts[tag] = []
         tag_counts[tag].append(word)

         if word not in word_tag_counts.keys(): word_tag_counts[word] = {}
         if tag not in word_tag_counts[word].keys(): word_tag_counts[word][tag] = 0
         
         word_tag_counts[word][tag] +=1

         bi_tags = (prev_tag, tag) if prev_tag is not None else ('', tag)
         
         if bi_tags not in tag_bigram.keys(): tag_bigram[bi_tags] = 0
         
         tag_bigram[bi_tags]+=1
         prev_tag = tag

   return tag_bigram, tag_counts, word_tag_counts

def build_transition_matrix(tag_bigram, tag_counts): 
   global TAGS, T_c
   n_tags = len(TAGS)
   t_mat = np.zeros((n_tags, n_tags-1), dtype=np.float64) # prev tag by cur tag 

   for i, tag1 in enumerate(TAGS): 
      for j, tag2 in enumerate(T_c): 

         bi_count = tag_bigram[(tag1, tag2)] if (tag1, tag2) in tag_bigram.keys() else 0
         t_mat[i, j] = (bi_count + 1)/ (len(tag_counts[tag1]) + n_tags) # add one smoothing 
         # denom: add tag_counts[tag1] + num_unique bigrams for tag1
   
   return t_mat 

def build_emission_matrix(tag_counts, word_tag_counts): 
   global T_c, CORPUS

   n_tags = len(T_c)
   n_words = len(CORPUS)

   emission_mat = np.zeros((n_words, n_tags), dtype=np.float64)
   
   for i, tag in enumerate(T_c): 
      for j, word in enumerate(CORPUS): 
         t_count = word_tag_counts[word][tag] if tag in word_tag_counts[word].keys() else 0
         emission_mat[j, i] = t_count / len(tag_counts[tag])

   return emission_mat   

def train_model():
   global CORPUS, TAGS, N_UNIQ_TAG, E_MAT, T_MAT, T_c
   
   train_set = word_tuples(read_file())

   tag_bigram, tag_counts, word_tag_counts = get_tag_counts(train_set)

   CORPUS = list(set(word_tag_counts.keys()))
   N_UNIQ_TAG = {k:len(set(v)) for k,v in tag_counts.items()}
   TAGS = list(set(N_UNIQ_TAG.keys()))
   T_c = TAGS.copy()
   T_c.remove('')

   T_MAT = build_transition_matrix(tag_bigram, tag_counts)
   E_MAT = build_emission_matrix(tag_counts, word_tag_counts)

def save_model(): 
   global CORPUS, N_UNIQ_TAG, E_MAT, T_MAT, T_c

   vocab=" ".join(CORPUS)
   np.set_printoptions(threshold=sys.maxsize)
   e_mat_str = np.array2string(E_MAT)
   t_mat_str = np.array2string(T_MAT)

   with open(OUT_PATH, "w") as f: 
      f.write(f"corpus:\n\n{vocab}\n\n")
      f.write(f"tags:\n\n{json.dumps(N_UNIQ_TAG)}\n\n")
      f.write(f"matrix tag order:\n\n{json.dumps(TAGS)}\n\n")
      f.write(f"Emission Matrix (dimensions: words x tags)\n\n{e_mat_str}\n\n")
      f.write(f"Transition Matrix (dimensions: tags+1 x tags)\n\n{t_mat_str}")

def main(): 

   train_model()
   save_model()

   return (0)

if __name__ == '__main__': 
   main()
