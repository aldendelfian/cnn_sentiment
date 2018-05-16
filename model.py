import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Kim2014(nn.Module):

  def __init__(self, embed_num, label_num, embeddings_dim, embeddings_mode, initial_embeddings):
    
    # cannot assign module before Module.__init__() call
    super(CNN_Kim2014, self).__init__()

    self.embed_num  = embed_num
    self.label_num  = label_num
    # self.use_gpu    = use_gpu
    # self.embed_dim  = 300        # mengikuti dimensi embedding Mikolov 2013
    self.embed_dim    = embeddings_dim
    self.embed_mode   = embeddings_mode
    self.channel_in   = 1
    self.feature_num  = [100, 100]
    self.kernel_width = [7, 7]
    self.dropout_rate = 0.5
    self.norm_limit   = 3
    # self.act_layer    = activation_layer

    # Memastikan hasil True jika False Error
    assert (len(self.feature_num) == len(self.kernel_width))
    self.kernel_num = len(self.kernel_width)

    # Static gimana ???
    # Non-static gimana ???
    # Multichannel gimana ???
    self.embeddings = nn.Embedding(self.embed_num, self.embed_dim, padding_idx=1)
    if self.embed_mode == 'non-static' or self.embed_mode == 'static' or self.embed_mode == 'multichannel':
      # ???
      self.embeddings.weight.data.copy_(torch.from_numpy(initial_embeddings))
      # if self.embed_mode == 'static':
      #   print()
      # elif self.embed_mode == 'multichannel':
      #   print()
    self.convs = nn.ModuleList([nn.Conv1d(self.channel_in, self.feature_num[i],
                                          self.embed_dim*self.kernel_width[i], stride=self.embed_dim)
                                   for i in range(self.kernel_num)])

    self.linear = nn.Linear(sum(self.feature_num), self.label_num)
    
  def forward(self, input):
    batch_width = input.size()[1]
    x = self.embeddings(input).view(-1, 1, self.embed_dim*batch_width)
    
    conv_results = [F.max_pool1d(F.relu(self.convs[i](x)), batch_width - self.kernel_width[i] + 1).view(-1, self.feature_num[i])
            for i in range(len(self.feature_num))
        ]

    x = torch.cat(conv_results, 1)
    x = F.dropout(x, p=self.dropout_rate, training=self.training)
    x = self.linear(x)
    return x