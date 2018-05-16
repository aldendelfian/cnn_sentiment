import time
from gensim.models.keyedvectors import KeyedVectors

class Embeddings():

  def __init__(self):
    self.embed_type  = None
    self.embed_dim   = 0
  
  def read_social_media(self):
    embeddings_social_media = ''
    embeddings_path = embeddings_social_media
    
    print('Loading Indonesian Social Media Word Embedding ...')

    start = time.time()
    word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path)
    end = time.time()
    print("Indonesian Social Media Word Embedding loading done in {} seconds".format(end - start))
    self.word2vec = word2vec_model
    self.embed_dim = 300
  
  # Method ini nanti dihapus soalnya modelnya dibuat pake bahasa inggris
  def read_google(self):
    # embeddings_google = 'google.bin'    
    embeddings_google = 'socmed_w2v_model_1.bin'
    embeddings_path = embeddings_google
    
    print('Loading Google Word Embedding ...')

    start = time.time()
    word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path, binary="False", unicode_errors='ignore')
    end = time.time()
    print("Google Word Embedding loading done in {} seconds".format(end - start))
    self.word2vec = word2vec_model
    self.embed_dim = 300

  def read_godin(self):
    embeddings_godin = ''
    embeddings_path = embeddings_godin

    print('Loading Godin Word Embedding ...')

    start = time.time()
    word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path)
    end = time.time()
    print("Godin Word Embedding loading done in {} seconds".format(end - start))
    self.word2vec = word2vec_model
    self.embed_dim = 400