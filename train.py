import argparse
import model
import prepare_data
import torch
import numpy as np
import read_embed
import inspect
import predict
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data
from collections import defaultdict
from math import ceil

# Create Parser
parser  = argparse.ArgumentParser(description='Hopefully somewhat generalized classifier for Sentiment Analysis')

parser.add_argument('-embeddings_dim', type=int, default=300, help='the default size of embeddings. '
                                                                   'When mikolov2013 is used, must use a size of 300, '
                                                                   'when godin2015 is used, must use a size of 400, '
                                                                   'and when grave2018 is used, must use 300 ')
parser.add_argument('-epoch_num', type=int, default=1, help='number of epochs')
parser.add_argument('-embeddings_mode', type=str, default='random', help='random, static,non-static, or multichannel')
parser.add_argument('-model_type', type=str, default='kim2014', help='kim2014')
parser.add_argument('-batch_size', type=int, default=50, help='size of the minibatches')
parser.add_argument('-kim2014_embeddings_mode', type=str, default='random', help='random, static, non-static, or multichannel')
# ???
# parser.add_argument('-conv_mode', type=str, default='wide', help='convolution mode')

args  = parser.parse_args()

def cross_validate(fold, data, embeddings, args):
  
  # defaultdict buat apa (?)
  actual_counts     = defaultdict(int)
  predicted_counts  = defaultdict(int)
  match_counts      = defaultdict(int)

  # Buat apa ini ???
  # Penggunaan ceil buat apa (?)
  split_width = int(ceil(len(data.examples)/fold))

  for i in range(fold):
    print('###=============FOLD [{}]=============###'.format(i + 1))

    # Maksudnya kayak gimana ini bentuk outputnya (?)
    train_examples  = data.examples[:]
    # Ini apa lagi cara bacanya (?)
    del train_examples[i*split_width:min(len(data.examples), (i+1)*split_width)]
    # Ini juga kayak gimana coba (?)
    test_examples   = data.examples[i*split_width:min(len(data.examples), (i+1)*split_width)]


    # Print count data in this fold
    print('###---------------Counts--------------###')
    total_len = len(data.examples)
    train_len = len(train_examples)
    test_len  = len(test_examples)

    # defaultdict buat apa (?)
    train_counts = defaultdict(int)
    test_counts  = defaultdict(int)

    for example in train_examples:
      # Nanti dicoba print disini buat liat apa isinya
      train_counts[example.label] += 1
    
    for example in test_examples:
      # Ini juga coba di print liat apa isinya
      test_counts[example.label] += 1
    
    # Print Persebaran Data
    # bangun if untuk subtask A, B dan C

    print('\tTotal Number of Examples : {}\n'.format(total_len))
    print('\tNumber of Train Examples : {}'.format(train_len))
    strg_pos  = train_counts['strong positive']
    pos       = train_counts['positive']
    neu       = train_counts['neutral']
    neg       = train_counts['negative']
    strg_neg  = train_counts['strong negative']
    print('\t\tTrain-Strong Positive  = {} ({}%)'.format(strg_pos, 100*strg_pos/train_len))
    print('\t\tTrain-Positive         = {} ({}%)'.format(pos, 100*pos/train_len))
    print('\t\tTrain-Neutral          = {} ({}%)'.format(neu, 100*neu/train_len))
    print('\t\tTrain-Negative         = {} ({}%)'.format(neg, 100*neg/train_len))
    print('\t\tTrain-Strong Negative  = {} ({}%)\n'.format(strg_neg, 100*strg_neg/train_len))
    
    print('\tNumber of Test Examples : {}'.format(test_len))
    strg_pos  = test_counts['strong positive']
    pos       = test_counts['positive']
    neu       = test_counts['neutral']
    neg       = test_counts['negative']
    strg_neg  = test_counts['strong negative']
    print('\t\tTest-Strong Positive  = {} ({}%)'.format(strg_pos, 100*strg_pos/test_len))
    print('\t\tTest-Positive         = {} ({}%)'.format(pos, 100*pos/test_len))
    print('\t\tTest-Neutral          = {} ({}%)'.format(neu, 100*neu/test_len))
    print('\t\tTest-Negative         = {} ({}%)'.format(neg, 100*neg/test_len))
    print('\t\tTrTestain-Strong Negative  = {} ({}%)\n'.format(strg_neg, 100*strg_neg/test_len))
    print('###-----------------------------------###')

    # What it is ???

    fields = data.fields
    train_set = torchtext.data.Dataset(examples=train_examples, fields=fields)
    test_set  = torchtext.data.Dataset(examples=test_examples, fields=fields)

    #Building Vocab
    # Masih belum dipahami di proses ini

    text_field  = None
    label_field = None

    for field_name, field_object in fields:
      if field_name == 'text':
        text_field = field_object
      elif field_name == 'label':
        label_field = field_object
    
    text_field.build_vocab(train_set)
    label_field.build_vocab(train_set)

    data.vocab_to_idx = dict(text_field.vocab.stoi)
    data.idx_to_vocab = {v: k for k, v in data.vocab_to_idx.items()}

    data.label_to_idx = dict(label_field.vocab.stoi)
    data.idx_to_label = {v: k for k, v in data.label_to_idx.items()}

    embed_num = len(text_field.vocab)
    label_num = len(label_field.vocab)

    # Loading pre-trained embedding
    emb_init_values = np.array(data.create_fold_embeddings(embeddings, args))

    train_iter, test_iter = torchtext.data.Iterator.splits((train_set, test_set),
                                                            batch_sizes=(args.batch_size, len(test_set)),
                                                            device=-1,repeat=False)
    
    # Ini buat ngukur performance model tapi belum ngerti ngukur performa keseluruhan gimana
    train_bulk_dataset = train_set,
    train_bulk__size   = len(train_set),
    train_bulk_iter    = torchtext.data.Iterator.splits(datasets=train_bulk_dataset, 
                                                        batch_sizes=train_bulk__size,
                                                        device=-1, repeat=False)[0]
    
    kim2014 = model.CNN_Kim2014(embed_num, label_num - 1,args.embeddings_dim, 
                                args.kim2014_embeddings_mode,emb_init_values)
    
    trained_model = train(kim2014, train_iter, test_iter, data.label_to_idx, data.idx_to_label, train_bulk_iter)


def train(model, train_iter, test_iter, label_to_idx, idx_to_label, train_bulk_iter):
  
  parameters = filter(lambda p: p.requires_grad, model.parameters())

  optimizer = torch.optim.Adadelta(parameters)
  # optimizer = torch.optim.Adam(parameters)

  model.train()
  
  # corrects_sum, accuracy_sum  = 0, 0
  
  for epoch in range(1, args.epoch_num+1):
    
    print("###__________EPOCH[{}]__________###".format(epoch))
    steps = 0
    corrects_sum  = 0
    
    for batch in train_iter:
      text_numerical, target = batch.text, batch.label

      text_numerical.data.t_()
      target.data.sub_(1)

      optimizer.zero_grad()

      forward = model(text_numerical)
      loss = F.cross_entropy(forward, target)
      loss.backward()
      # ???
      optimizer.step()
      steps += 1

      corrects = (torch.max(forward, 1)[1].view(target.size()).data == target.data).sum()
      # corrects_sum += (torch.data.max(forward, 1)[1].view(target.size()).data == target.data).sum()
      
      accuracy = 100.0 * corrects / batch.batch_size
    

      # print('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})\n'.format(steps, loss.data[0],
      #                                                                        accuracy, corrects,
      #                                                                        batch.batch_size))
    
    print('\n###____Performance on Train Data____###')
    actual, predicted = evaluate(model, train_bulk_iter)
    # I no have idea ????
    epoch_actual_counts, epoch_predicted_counts, epoch_match_counts = calculate_fold_counts(actual, predicted, label_to_idx, idx_to_label)
    calculate_and_display_SemEval_metrics(epoch_actual_counts, epoch_predicted_counts, epoch_match_counts)
    
    print('\n###____Performance on Test Data____###')
    actual, predicted = evaluate(model, test_iter)
    # I no have idea ???
    epoch_actual_counts, epoch_predicted_counts, epoch_match_counts = \
        calculate_fold_counts(actual, predicted, label_to_idx, idx_to_label)
    calculate_and_display_SemEval_metrics(
        epoch_actual_counts, epoch_predicted_counts, epoch_match_counts)
  
  return model
    
def evaluate(model, data_iter):

  model.eval()
  corrects, avg_loss = 0, 0
  
  data_iter.sort_key = lambda x: len(x.text)

  for batch in data_iter:
    
    text_numerical, target = batch.text, batch.label
    text_numerical.data.t_()
    target.data.sub_(1)

    forward = model(text_numerical)
    loss = F.cross_entropy(forward, target, size_average=False)

    avg_loss += loss.data[0]
    # avg_loss += loss.item([0])
    # avg_loss += loss.item()
    corrects += (torch.max(forward, 1)[1].view(target.size()).data == target.data).sum()

  size = len(data_iter.dataset)
  # avg_loss = avg_loss/size
  accuracy = 100.0 * corrects/size

  return target.data, torch.max(forward, 1)[1].view(target.size()).data

def calculate_fold_counts(actual, predicted, label_to_idx, idx_to_label):
  assert len(actual)  ==  len(predicted)

  fold_actual_counts    = defaultdict(int)
  fold_predicted_counts = defaultdict(int)
  fold_match_counts     = defaultdict(int)

  for i in range(len(actual)):

    idx   = actual[i] + 1
    label = idx_to_label[idx]
    fold_actual_counts[label] += 1

    # Ini buat kayak gimana sih ??

    if actual[i] == predicted[i]:
      fold_match_counts[label] += 1
  
  # Ini juga gimana lagi 
  for i in range(len(predicted)):
    idx = predicted[i] + 1
    label = idx_to_label[idx]
    fold_predicted_counts[label] += 1
  
  return fold_actual_counts, fold_predicted_counts, fold_match_counts

def calculate_and_display_SemEval_metrics(actual_counts, predicted_counts, match_counts):
  
  precisions = defaultdict(float)
  recalls    = defaultdict(float)
  f_measures = defaultdict(float)

  for label in actual_counts.keys():
    precision = match_counts[label] / predicted_counts[label] if predicted_counts[label] > 0 else 0
    recall    = match_counts[label] / actual_counts[label] if actual_counts[label] > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Store these values in the metric dictionaries
    precisions[label] = 100 * precision
    recalls[label]    = 100 * recall
    f_measures[label] = 100 * f_measure

    # Print precision, recall, and f-measure for each class
    print("On class {}:".format(label))
    print("\tPrecision = {}%".format(100 * precision))
    print("\tRecall    = {}%".format(100 * recall))
    print("\tF-Measure = {}%".format(100 * f_measure))
  
  print("\nSubtask C measures:")

  # Ini ditambahi lagi untuk 5 label lain kalau dia bukan 5 label dibuat if
  sum_recall = 0
  # Dibuat if disini
  for label in recalls:
    if (label == "strong positive") or (label == "positive") or (label == "neutral") or (label == "negative") or (label == "strong negative"):
      sum_recall += recalls[label]
  avg_recall = sum_recall / 5

  # Dibuat if disini
  sum_f_measure = 0
  for label in f_measures:
    if (label == "strong positive") or (label == "positive") or (label == "negative") or (label == "strong negative"):
      sum_f_measure += f_measures[label]
  f_pos_neg = sum_f_measure / 4

  # ???
  test_size = sum(actual_counts.values())
  print("Test size: {}".format(test_size))
  print("AvgRecall = {}%".format(avg_recall))
  print("F-PosNeg  = {}%".format(f_pos_neg))
  print("Accuracy  = {}%".format(100 * sum(match_counts.values()) / test_size))


if __name__ == "__main__":

  args.model_type  = 'kim2014'
  args.embed_mode  = 'non-static'
  args.embeddings_source = 'none'
  args.epoch_num   = 5
  print("Model type       : {}".format(args.model_type))
  print("Embedding Mode   : {}".format(args.embed_mode))
  print("Number of Epochs : {}".format(args.epoch_num))

  in_data = prepare_data.prepare_data()
  in_data.read_dataset()
  embeddings = read_embed.Embeddings()

  embeddings.read_google()
  args.embeddings_dim = embeddings.embed_dim


  print("--- Execution Main.py ---")

  # vektor = in_data.emb_init_values
  
  # vocab_size  = in_data.embed_num
  # label_types = in_data.label_num - 1
  

  # # print('Nilai Label')
  # # Jumlah Label dalam data
  # # print(label_types)
  # print('Panjang Data')
  # print(len(vektor))
  # print('Nilai vektor niii')
  # print(type(vektor))
  # print(vektor[6])
  # print('Nilai vektor niii 2')
  # print(vektor[0])  
  # # abc = type(vektor[6][0])
  # # str_abc = str(abc)
  # # print(str_abc)
  # for x in range(0, len(vektor)):
  #   for y in range (0, len(vektor[x])):
  #     abc = type(vektor[x][y])
  #     # print(vektor[x][y])
  #     str_abc = str(abc)
  #     # print(str_abc)
  #     if str_abc != "<class 'numpy.float64'>":
  #       print('Index  : {}'.format(x))
  #       print("ADA YANG SALAH")
  #     # else:
  #     #   print("ok")


  # train_iter, dev_iter = data.Iterator.splits((in_data.train_set, in_data.dev_set),batch_sizes=(50, len(in_data.dev_set)),device=-1,repeat=False)



  # # print(len(train_iter))
  # print('--- Execution Model.py ---')



  # cnn = model.CNN(vocab_size, label_types, embed_mode, in_data.emb_init_values)
  
  # print('--- Execution Model.py Done ---')
  # # # print()

  # # # # kf = KFold(n_splits=2)
  # # # # num = kf.get_n_splits(X)

  # # # # for x in num:
  # # # #   print(x)

  # train(train_iter, dev_iter, cnn)
  cross_validate(10, in_data, embeddings, args)

  print('--- DONE MAN DONE ---')
  # # # print('Mau coba predict nih')
  # # # text = 'Gimana ya ? kok belum bisa dikirim sih sampe sekarang'
  # # # predict.predict(text, cnn, )
