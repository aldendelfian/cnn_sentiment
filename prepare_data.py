from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import random
import re
import time
import torchtext
from read_data import ReadDataTopik

class prepare_data:

  def __init__(self):
    self.emb_init_value = None
    self.vocab_to_idx   = None
    self.idx_to_vocab   = None
    self.label_to_idx   = None
    self.idx_to_label   = None
    self.embed_num      = 0
    self.label_num      = 0
    self.train_set      = None
    self.dev_set        = None
  
  def polarity_to_label(self, polarity):
    # Di sini musti ditambain lagi untuk subtask A dan B
    if int(polarity) == -2:
      label = 'strong negative'
    elif int(polarity) == -1:
      label = 'negative'
    elif int(polarity) == 1:
      label = 'positive'
    elif int(polarity) == 2:
      label = 'strong positive'
    else:
      label = 'neutral'

    return label
    # return 'negative' if int(polarity) < 0 else 'positive' if int(polarity) > 0 else 'neutral'
  
  def read_dataset(self):
    # pos_kuat  = []
    # pos_lemah = []
    # netral    = []
    # neg_lemah = []
    # neg_kuat  = []

    # with open('data/example_tagging.txt', 'r', encoding='utf8') as f:
    #   for line in f:
    #     id_tweet, topik, lbl_sntmnt, lbl_topik, lbl_sarkas, message = self.divide_line(line)
    #     # message = ReadDataTopik.get_primes(message)
    #     readtopik = ReadDataTopik(message)
    #     message   = readtopik.get_value()

        # message  = self.clean_str(message)
        # print('Nilai : {}'.format(message))

        # if lbl_sntmnt == '-2':
        #   neg_kuat.append(message)
        # elif lbl_sntmnt == '-1':
        #   neg_lemah.append(message)
        # elif lbl_sntmnt == '0':
        #   netral.append(message)
        # elif lbl_sntmnt == '1':
        #   pos_lemah.append(message)
        # elif lbl_sntmnt == '2':
        #   pos_kuat.append(message)
    
    # print('Jumlah masing-masing data : ')
    # print('Negatif Kuat : {}'.format(len(neg_kuat)))
    # print('Negatif Lemah : {}'.format(len(neg_lemah)))
    # print('Netral : {}'.format(len(netral)))
    # print('Positif Lemah : {}'.format(len(pos_lemah)))
    # print('Positif Kuat : {}'.format(len(pos_kuat)))

    file_name = 'data/example_tagging.txt'

    indo_ecommerce_data = ReadDataTopik(file_name)

    twt_id_field = torchtext.data.Field(use_vocab=False, sequential=False)
    label_field = torchtext.data.Field(sequential=False)
    text_field = torchtext.data.Field()

    fields = [('twt_id', twt_id_field), ('label', label_field), ('text', text_field)]

    # examples = 	[torchtext.data.Example.fromlist([line, 'negative'], fields) for line in neg]
    # examples += [torchtext.data.Example.fromlist([line, 'positive'], fields) for line in pos]
    
    self.fields = fields
    
    examples = [torchtext.data.Example.fromlist([twt_id, self.polarity_to_label(polarity), text], fields)
                    for twt_id, polarity, text in indo_ecommerce_data]
                    
    self.examples = examples

    # random.seed(4321)
    # random.shuffle(examples)

    # print('Jumlah data : ', len(examples))

    # dev_ratio = 0.1
    # dev_index = -1*int(dev_ratio*len(examples))

    # self.train_set = torchtext.data.Dataset(examples=examples[:dev_index], fields=fields)
    # self.dev_set   = torchtext.data.Dataset(examples=examples[dev_index:], fields=fields)

    # print('Jumlah data Training : ',len(self.train_set))
    # print('Jumlah data Testing : ',len(self.dev_set))

    # text_field.build_vocab(self.train_set)
    # label_field.build_vocab(self.train_set)

    # print("loading word2vec...")
    # start = time.time()
    # word_vectors = KeyedVectors.load_word2vec_format("google.bin", binary="False")
    # end = time.time()
    # print("word2vec loading done in {} seconds".format(end-start))


    # # print('Coba liat Vocab :')
    # # print(text_field.vocab.text)

    # self.vocab_to_idx = dict(text_field.vocab.stoi)
    # self.idx_to_vocab = { v: k for k, v in self.vocab_to_idx.items() }

    # print('Jumlah vocab : ',len(self.idx_to_vocab))
    
    # print('---------------------------------')
    
    # print("loading vocab...")

    # emb_init_values = []
    
    # for x in range(0,len(self.idx_to_vocab)):
    #   word = self.idx_to_vocab[x]
    #   # print(word)
    #   if word == '<unk>':
    #     emb_init_values.append(np.random.uniform(-0.01, 0.01, 300).astype('float64'))
    #   elif word == '<pad>':
    #     emb_init_values.append(np.zeros(300).astype('float64'))
    #   elif word in word_vectors.vocab:
    #     emb_init_values.append(word_vectors.word_vec(word))
    #   else:
    #     emb_init_values.append(np.random.uniform(-0.01, 0.01, 300).astype('float64'))
    
    # # print('Nilai sebelum array ######')
    # # for x in range(1, 4):
    # #   print(emb_init_values[x])
    # # emb_init_values = np.float64(emb_init_values)

    # self.emb_init_values = np.array(emb_init_values)

    # # print('Nilai setelah array -----------')
    # # for x in range(1, 4):
    # #   print(self.emb_init_values[x])


    # self.embed_num = len(text_field.vocab)
    # self.label_num = len(label_field.vocab)
    
    # self.label_to_idx = dict(label_field.vocab.stoi)
    # self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

  def create_fold_embeddings(self, embeddings, args):
    emb_init_values = []

    if args.embeddings_source == 'none':
      for i in range(self.idx_to_vocab.__len__()):  # Untuk memastikan bahwa urut
        word = self.idx_to_vocab.get(i)
        if word == '<unk>':
          emb_init_values.append(np.random.uniform(-0.01, 0.01, args.embeddings_dim).astype('float32'))
        elif word == '<pad>':
          emb_init_values.append(np.zeros(args.embeddings_dim).astype('float32'))
        else:
          emb_init_values.append(np.random.uniform(-0.01, 0.01, args.embeddings_dim).astype('float32'))

    self.emb_init_values = emb_init_values
    return emb_init_values
    

# if __name__ == '__main__':
#   # with open('data/example_tagging_2.txt', 'r') as f:
#   #   for line in f:
#   #     print(line.strip())
#   df = prepare_data()
#   df.read_dataset()