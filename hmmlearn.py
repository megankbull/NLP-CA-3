# python hmmlearn.py /path/to/input
import sys 
import random 
import math
import numpy as np

# output needs enough info for decode tag new data
# human readable to inspect model params  
OUT_PATH = "hmmmodel.txt"

def read_file(f_path=sys.argv[1]):
   with open(f_path, 'r') as f: 
      lines = f.readlines()
   
   words = []
   for line in lines: 
      words.extend(line.split(" "))

   return words 

def word_tuples(): 
   words = read_file()
   tagged_tupes = []

   for token in words: 
      tok = token.strip()
      tag = tok.split("/")[-1].upper()
      w = tok.removesuffix(f"/{tag}").lower()
      tagged_tupes.append((w, tag))

   return tagged_tupes

def train_test_split(test_frac=0.2): 
   tagged_tuples = word_tuples()
   random.shuffle(tagged_tuples)

   num_states = len(tagged_tuples)

   test_len = math.floor(num_states * test_frac)
   test_set = tagged_tuples[:test_len]
   train_set = tagged_tuples[test_len:]

   return train_set, test_set

def get_tag_set(train_set): 
   return {tag for _, tag in train_set}

def get_vocab(train_set): 
   return {word for word, _ in train_set}

def get_tag_counts(word_tuples): 
   tag_bigram = {}
   tag_counts = {}
   word_tag_counts = {}

   prev_tag = None

   for word, tag in word_tuples: 
      if word not in word_tag_counts.keys(): 
         word_tag_counts[word] = {}
      if tag not in word_tag_counts[word].keys():
         word_tag_counts[word][tag] = 0
      word_tag_counts[word][tag] +=1

      if tag not in tag_counts.keys(): tag_counts[tag] = 0 
      tag_counts[tag] += 1

      if prev_tag is not None: 
         bi_tags = (prev_tag, tag)
         if bi_tags not in tag_bigram.keys(): 
            tag_bigram[bi_tags] = 0
         tag_bigram[bi_tags]+=1
      prev_tag = tag

   return tag_bigram, tag_counts, word_tag_counts

def build_transition_matrix(tag_bigram, tag_counts): 
   n_tags = len(tag_counts.keys())

   tag_matrix = np.zeros((n_tags, n_tags), dtype=np.float64)
   
   for i, tag1 in enumerate(tag_counts.keys()): 
      for j, tag2 in enumerate(tag_counts.keys()): 
         bi_count = tag_bigram[(tag1, tag2)] if (tag1, tag2) in tag_bigram.keys() else 0
         tag_matrix[i, j] = bi_count/ tag_counts[tag1]

   return tag_matrix

def build_emission_matrix(tag_counts, word_tag_counts): 
   n_tags = len(tag_counts.keys())
   n_words = len(word_tag_counts.keys())

   emission_mat = np.zeros((n_tags, n_words), dtype=np.float64)
   
   for i, tag in enumerate(tag_counts.keys()): 
      for j, word in enumerate(word_tag_counts.keys()): 
         t_count = word_tag_counts[word][tag] if tag in word_tag_counts[word].keys() else 0
         emission_mat[i, j] = t_count / tag_counts[tag]

   return emission_mat   

def main(): 
   train, test = train_test_split()
   tag_bigram, tag_counts, word_tag_counts = get_tag_counts(train)

   t_mat = build_transition_matrix(tag_bigram, tag_counts)
   e_mat = build_emission_matrix(tag_counts, word_tag_counts)

   return (0)

if __name__ == '__main__': 
   main()
