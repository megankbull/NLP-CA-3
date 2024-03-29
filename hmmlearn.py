# python hmmlearn.py /path/to/input
import heapq
import operator
import sys 
import json
import numpy as np

OUT_PATH = "hmmmodel.txt"

CORPUS = None
COMMON_TAGS = None
TAGS = None
T_MAT = None
E_MAT = None
T_c = None
SUFF_TAG_COUNTS = None 

def read_file(f_path=sys.argv[1]):
   with open(f_path, 'r') as f: 
      return [l.strip().split(" ") for l in f.readlines()]

def word_tuples(lines):

   tagged_tupes = []
   
   for line in lines: 
      line_tupes = []
      for token in line: 
         tok = token.strip()
         tag = tok.split("/")[-1]

         w = tok.replace(f"/{tag}", '')
         line_tupes.append((w, tag.upper()))

      tagged_tupes.append(line_tupes)
      
   return tagged_tupes

def get_tag_counts(): 

   line_tuples = word_tuples(read_file())

   tag_counts = {'\\':[], '/':[]}
   tag_bigram = {}
   word_tag_counts = {}
   suff_tag_counts = {}

   for line in line_tuples: 
      prev_tag = '\\'
      for i, tup in enumerate(line): 
         if not tup: continue 
         word = tup[0]
         tag = tup[1]
         
         suff = word[-1:]

         if prev_tag == '\\': tag_counts['\\'].append(word)
         if tag not in tag_counts.keys(): tag_counts[tag] = []

         if suff not in suff_tag_counts.keys(): suff_tag_counts[suff] = {}
         if tag not in suff_tag_counts[suff].keys(): suff_tag_counts[suff][tag] = 0

         if word not in word_tag_counts.keys(): word_tag_counts[word] = {}
         if tag not in word_tag_counts[word].keys(): word_tag_counts[word][tag] = 0

         bi_tags = (prev_tag, tag)
         if bi_tags not in tag_bigram.keys(): tag_bigram[bi_tags] = 0
         
         if i == len(line)-2 :
            bi_tag2 = (tag, '/')
            if bi_tag2 in tag_bigram: tag_bigram[bi_tag2] += 1
            else: tag_bigram[bi_tag2] = 1
            
            tag_counts['/'].append(word)

         tag_counts[tag].append(word)
         word_tag_counts[word][tag] += 1
         suff_tag_counts[suff][tag] += 1
         tag_bigram[bi_tags] += 1

         prev_tag = tag

   return tag_bigram, tag_counts, word_tag_counts, suff_tag_counts

def build_transition_matrix(tag_bigram, tag_counts): 
   global TAGS, T_c, T_MAT

   n_tags = len(TAGS)
   T_MAT = np.zeros((n_tags, n_tags), dtype=np.float64)

   for i, tag1 in enumerate(TAGS): 
      for j, tag2 in enumerate(T_c): 
         bi_count = tag_bigram[(tag1, tag2)] if (tag1, tag2) in tag_bigram.keys() else 0
         T_MAT[i, j] = (bi_count + 1) / (len(tag_counts[tag1]) + n_tags) # add one smoothing 
         # denom: add tag_counts[tag1] + num_unique bigrams for tag1
   
def build_emission_matrix(tag_counts, word_tag_counts): 
   global T_c, CORPUS, E_MAT

   n_tags = len(T_c)
   n_words = len(CORPUS)

   E_MAT = np.zeros((n_words, n_tags), dtype=np.float64)
   
   for i, word in enumerate(CORPUS): 
      for j, tag in enumerate(T_c): 
         if tag == '/' : continue 
         t_count = word_tag_counts[word][tag] if tag in word_tag_counts[word].keys() else 0
         E_MAT[i, j] = t_count / len(tag_counts[tag])
   
def train_model():
   global CORPUS, TAGS, T_c, SUFF_TAG_COUNTS, COMMON_TAGS

   tag_bigram, tag_counts, word_tag_counts, SUFF_TAG_COUNTS = get_tag_counts()
   
   CORPUS = list(set(word_tag_counts.keys()))
   n_uniq_tag = {k:len(set(v)) for k,v in tag_counts.items()}
   TAGS = list(set(n_uniq_tag.keys()))

   T_c = TAGS.copy()
   T_c.remove('\\')

   common_tags_p = heapq.nlargest(7, n_uniq_tag.items(), key=operator.itemgetter(1))
   COMMON_TAGS = {t for t,_ in common_tags_p if t not in ['\\', '/']}
   
   build_transition_matrix(tag_bigram, tag_counts)
   build_emission_matrix(tag_counts, word_tag_counts)

def save_model(): 
   global CORPUS, TAGS, COMMON_TAGS, SUFF_TAG_COUNTS, E_MAT, T_MAT 

   vocab = " ".join(CORPUS) # list of unique words in training data 
   c_tags = " ".join(list(COMMON_TAGS)) # top 7 most common tags 
   np.set_printoptions(threshold=sys.maxsize)
   e_mat_str = np.array2string(E_MAT)
   t_mat_str = np.array2string(T_MAT)

   with open(OUT_PATH, "w") as f: 
      f.write(f"corpus:\n\n{vocab}\n\n")
      f.write(f"possible tags given suffix:\n\n{json.dumps(SUFF_TAG_COUNTS)}\n\n")

      f.write(f"common tags:\n\n{c_tags}\n\n")
      f.write(f"matrix tag order:\n\n{json.dumps(TAGS)}\n\n")

      f.write(f"Emission Matrix (dimensions: words x tags)\n\n{e_mat_str}\n\n")
      f.write(f"Transition Matrix (dimensions: tags+1 x tags)\n\n{t_mat_str}\n\n")

def main(): 

   train_model()
   save_model()

   return (0)

if __name__ == '__main__': 
   main()