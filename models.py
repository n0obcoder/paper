import sys 
def q(text = '====='):
    print(f'>{text}<')
    sys.exit()

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config as cfg

class CNN_Classifier(nn.Module):

    def __init__(self, in_channel):
        super(CNN_Classifier, self).__init__()

        # in_channel are the embedding size
        
        self.conv1 = nn.Conv1d(cfg.embed_dim, 256, 3, padding=1)

        self.conv2 = nn.Conv1d(256, 256, 3, padding=1)
        self.max_pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(256, 256, 3, padding=1)
        self.max_pool3 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(1280, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        print('> > > > > CNN Classifier INITIALIZED  < < < < <')

    def forward(self, embed_matrix):

        # print(embed_matrix.shape) # [bs, cfg.dim_emb, cfg.trg_len]

        features = F.relu(self.conv1(embed_matrix))
        # print(features.shape) # [bs, 256, cfg.trg_len]

        features = F.relu(self.conv2(features))
        # print(features.shape) # [bs, 256, cfg.trg_len]
        features = self.max_pool2(features)
        # print(features.shape) # [bs, 256, 10]

        features = F.relu(self.conv3(features))
        # print(features.shape) # [bs, 256, 10]
        features = self.max_pool3(features)
        # print(features.shape) # [bs, 256, 5]

        features = features.view(features.shape[0], -1)
        # print(features.shape) # [bs, 1280]

        features = F.relu(self.fc1(features))
        # print(features.shape) # [bs, 256]

        features = F.relu(self.fc2(features))
        # print(features.shape) # [bs, 128]

        probs = torch.sigmoid(self.fc3(features)).squeeze()
        # print(probs.shape) # [bs, 1]

        return probs

class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, style_embed_dim, hidden_dim, num_styles, n_layers, pretrained_weights = None, bidirectional = False):
        super(Encoder, self).__init__()

        # pretrained_weights: numpy matrix word embeddings of shape [vocab_size, hidden_dim]

        self.vocab_size    = vocab_size
        self.embed_dim     = embed_dim
        self.style_embed_dim = style_embed_dim
        self.hidden_dim    = hidden_dim
        self.num_styles    = num_styles
        self.n_layers      = n_layers

        self.text_embeddings  = nn.Embedding(vocab_size, embed_dim)  
        self.style_embeddings = nn.Embedding(num_styles, style_embed_dim)

        if pretrained_weights is not None:       
            self.text_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weights))
            print('\nPretrained word embeddings loaded in Encoder.')
            # self.text_embeddings.weight.requires_grad = False

        self.enc_gru          = nn.GRU(embed_dim + style_embed_dim, hidden_dim, n_layers, bidirectional = bidirectional, dropout = cfg.gru_dropout)

        print('\n> > > > > Encoder INITIALIZED  < < < < <')

    def forward(self, input_sequences, sequences_styles, sequences_lengths, hidden = None):

        # print(input_sequences.shape,  input_sequences.dtype)   # [max_length_in_this_batch, bs], torch.int64
        # print(sequences_styles.shape, sequences_styles.dtype)  # [max_length_in_this_batch, bs], torch.int64
        # print(type(sequences_lengths), len(sequences_lengths)) # list of len = 8

        # pdb.set_trace()

        text_embeddings  = self.text_embeddings(input_sequences)
        style_embeddings = self.style_embeddings(sequences_styles)

        # print(text_embeddings.shape , text_embeddings.dtype)  # [max_length_in_this_batch, bs, self.embed_dim], torch.float32
        # print(style_embeddings.shape, style_embeddings.dtype) # [max_length_in_this_batch, bs, self.hidden_dim], torch.float32
        
        combined_embedding = torch.cat([text_embeddings, style_embeddings], dim = 2)
        # print(combined_embedding.shape) # [max_length_in_this_batch, bs, self.embed_dim + self.style_embed_dim], torch.float32

        # Here we run rnns only on non-padded regions of the batch
        packed = pack_padded_sequence(combined_embedding, sequences_lengths)
        outputs, hidden = self.enc_gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs) # unpack (back to padded)

        # Here we run rnns only on all regions of the batch   !!!!!!!!!!!!!!!!!!!!!!!!!!!!! DEEP CHECK IS REQUIRED.
        # outputs, hidden = self.enc_gru(combined_embedding, hidden)
        # print('outputs.shape: ', outputs.shape)
        # print(output_lengths)
        # print(sequences_lengths)
        # pdb.set_trace()
        # print(outputs[12:, 2:, :5]) # this will be all zero in case we use pack_padded_sequences and pad_packed_sequences
        
        '''
        outputs, hidden = self.enc_lstm(combined_embedding, hidden)
        # We can also use the LSTM without packing the sequences.
        # outputs is the tensor containing the output features from the last layer of the LSTM for each t.
        # hidden[0] is the hidden state
        # hidden[1] is the cell state
        # hidden[0] is the tensor containing the hidden state for t=T
        # hidden[0] has the shape [num_layers*num_directions, bs, hidden_size]
        # outputs[-1, :, :] = hidden[0][1, :, :]
        # both are hidden state for last layer for t=T, with shape [bs, hidden_size]
        '''

        # print(output_lengths) # it is exactly the same as sequences_lengths 
        # print(outputs.shape)  # [max_length_in_this_batch, bs, self.hidden_dim*num_directions], torch.float32
        # print(hidden.shape)   # [num_layers*num_directions, bs, self.hidden_dim]
        # q()
        return outputs, hidden

class AttentionDecoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, style_emb_dim, hidden_dim, num_styles, n_layers, max_seq_length, bidirectional = False):
        super(AttentionDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.style_emb_dim = style_emb_dim
        self.hidden_dim = hidden_dim
        self.num_styles = num_styles
        self.n_layers   = n_layers
        self.max_seq_length = max_seq_length
        self.bidirectional = bidirectional

        self.text_embeddings  = nn.Embedding(vocab_size, embed_dim)     
        self.style_embeddings = nn.Embedding(num_styles, style_emb_dim)

        self.attn         = nn.Linear(self.embed_dim + self.style_emb_dim + self.hidden_dim*(1 + bidirectional), max_seq_length)
        self.attn_combine = nn.Linear(self.embed_dim + self.style_emb_dim + self.hidden_dim*(1 + bidirectional), self.hidden_dim)

        self.dec_gru      = nn.GRU(self.hidden_dim, self.hidden_dim, n_layers, bidirectional = bidirectional, dropout = cfg.gru_dropout)
        self.out          = nn.Linear(self.hidden_dim*(1 + bidirectional), self.vocab_size)

        print('> > > > > AttentionDecoder INITIALIZED  < < < < <')

    def forward(self, input_sequences, hidden, sequences_lengths, sequences_styles, encoder_outputs):
        # print('\ninside attention dec forward...')
        
        batch_size = input_sequences.shape[1]
        # print(input_sequences.shape, input_sequences.dtype)    # [1, bs]                                             , torch.int64
        # print(hidden.shape, hidden.dtype)                      # [cfg.n_layers*(1 + cfg.bidirectional), cfg.batch_size, cfg.hidden_dim], torch.float32
        # print(type(sequences_lengths), len(sequences_lengths)) # list of len = bs
        # print(sequences_styles.shape, sequences_styles.dtype)  # [1, bs]                                             , torch.int64
        # print(encoder_outputs.shape, encoder_outputs.dtype)    # [max_seq_length, bs, self.hidden_dim*num_directions], torch.float32
        
        last_layer_hidden = hidden.view(self.n_layers, 1 + self.bidirectional, batch_size, self.hidden_dim)
        last_layer_hidden = torch.cat((last_layer_hidden[-1, 0, :, :], last_layer_hidden[-1, 1, :, :]), dim = -1) if self.bidirectional else last_layer_hidden[-1, 0, :, :]
        # print(last_layer_hidden.shape) # [bs, (1+self.bidirectional)*self.hidden_dim]
    
        encoder_outputs = encoder_outputs.transpose(0,1)
        # print(encoder_outputs.shape, encoder_outputs.dtype)   # [bs, max_seq_length, self.hidden_dim*num_directions], torch.float32

        text_embeddings  = self.text_embeddings(input_sequences)
        style_embeddings = self.style_embeddings(sequences_styles)

        # print(text_embeddings.shape , text_embeddings.dtype)  # [1, bs, self.embed_dim], torch.float32
        # print(style_embeddings.shape, style_embeddings.dtype) # [1, bs, self.style_emb_dim], torch.float32
        
        combined_embedding = torch.cat([text_embeddings, style_embeddings], dim = 2)
        # print(combined_embedding.shape) # [1, bs, self.embed_dim + self.style_emb_dim], torch.float32

        attn_inp = torch.cat((combined_embedding.squeeze(), last_layer_hidden), dim = 1)
        # print(attn_inp.shape)           # [bs, self.embed_dim + self.style_emb_dim + self.hidden_dim*num_directions]
        attn_out = self.attn(attn_inp)
        # print(attn_out.shape)           # [bs, max_seq_length]

        # Masking of Attention Weights
        maxlen = attn_out.shape[1]
        idx = torch.arange(maxlen).unsqueeze(0).expand(attn_out.size())
        # print(idx, '\n----\n')
        len_expanded = torch.LongTensor(sequences_lengths).unsqueeze(1).expand(attn_out.size())
        # print(len_expanded, '\n----\n')
        attn_mask = idx < len_expanded
        # print(attn_mask, '\n----\n')

        attn_out[~attn_mask] = float('-inf')
        # print(attn_out, '\n----\n')

        attn_weights = F.softmax(attn_out, dim = 1).unsqueeze(dim = 1)
        # print(attn_weights.shape)       # [bs, 1, max_seq_length]

        # print(attn_weights, '\n----\n')

        # DONE

        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        # print(attn_applied.shape) # [bs, 1, self.hidden_dim*num_directions]

        output = torch.cat((combined_embedding.transpose(0,1), attn_applied), dim = 2)
        # print(output.shape) # [bs, 1, self.embed_dim + self.style_emb_dim + self.hidden_dim*num_directions]

        output = self.attn_combine(output)
        # print(output.shape) # [bs, 1, self.hidden_dim]

        output = F.relu(output)
        output, hidden = self.dec_gru(output.transpose(0,1), hidden)
        # print(output.shape) # [1, bs, self.hidden_dim*num_directions]

        output = F.log_softmax(self.out(output[0]), dim = 1)
        # print(output.shape) # [bs, self.vocab_size]
    
        return output, hidden

class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, style_embed_dim, hidden_dim, num_styles, n_layers, pretrained_weights, max_seq_length, bidirectional = False):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.style_embed_dim = style_embed_dim
        self.hidden_dim = hidden_dim
        self.num_styles = num_styles
        self.n_layers   = n_layers
        self.max_seq_length = max_seq_length
        self.bidirectional = bidirectional

        self.text_embeddings  = nn.Embedding(vocab_size, embed_dim)     
        self.style_embeddings = nn.Embedding(num_styles, style_embed_dim)

        if pretrained_weights is not None:       
            self.text_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weights))
            print('\nPretrained word embeddings loaded in Decoder.')
            # self.text_embeddings.weight.requires_grad = False

        # self.dec_gru      = nn.GRU(embed_dim + hidden_dim*(1 + bidirectional), hidden_dim, n_layers, bidirectional = bidirectional)
        self.dec_gru      = nn.GRU(embed_dim + style_embed_dim + hidden_dim*(1 + bidirectional), hidden_dim, n_layers, bidirectional = bidirectional)
        self.out          = nn.Linear(hidden_dim*(1 + bidirectional), vocab_size)

        print('> > > > > Decoder INITIALIZED  < < < < <')

    def forward(self, input_sequences, hidden, sequences_lengths, sequences_styles, encoder_outputs):
        # print('\ninside dec forward...')
        
        batch_size = input_sequences.shape[1]
        # print(input_sequences.shape, input_sequences.dtype)    # [1, bs]                                             , torch.int64
        # print(hidden.shape, hidden.dtype)                      # [cfg.n_layers*(1 + cfg.bidirectional), cfg.batch_size, cfg.hidden_dim], torch.float32
        # print(type(sequences_lengths), len(sequences_lengths)) # list of len = bs
        # print(sequences_styles.shape, sequences_styles.dtype)  # [1, bs]                                             , torch.int64
        # print(encoder_outputs.shape, encoder_outputs.dtype)    # [max_seq_length, bs, self.hidden_dim*num_directions], torch.float32
        
        last_layer_hidden = hidden.view(self.n_layers, 1 + self.bidirectional, batch_size, self.hidden_dim)
        last_layer_hidden = torch.cat((last_layer_hidden[-1, 0, :, :], last_layer_hidden[-1, 1, :, :]), dim = -1) if self.bidirectional else last_layer_hidden[-1, 0, :, :]
        # print(last_layer_hidden.shape) # [bs, (1+self.bidirectional)*self.hidden_dim]

        encoder_outputs = encoder_outputs.transpose(0,1)
        # print(encoder_outputs.shape, encoder_outputs.dtype)   # [bs, max_seq_length, self.hidden_dim*num_directions], torch.float32

        text_embeddings  = self.text_embeddings(input_sequences)
        style_embeddings = self.style_embeddings(sequences_styles)

        # print(text_embeddings.shape , text_embeddings.dtype)  # [1, bs, self.embed_dim], torch.float32
        # print(style_embeddings.shape, style_embeddings.dtype) # [1, bs, self.style_emb_dim], torch.float32
        
        # combined_embedding = text_embeddings
        combined_embedding = torch.cat([text_embeddings, style_embeddings], dim = 2)
        # print(combined_embedding.shape) # [1, bs, self.embed_dim + self.style_emb_dim], torch.float32

        # pdb.set_trace()
        dec_inp = torch.cat((combined_embedding, last_layer_hidden.unsqueeze(0)), dim = 2)
        # print(dec_inp.shape)           # [1, bs, self.embed_dim + self.style_emb_dim + self.hidden_dim*num_directions]
        
        output, hidden = self.dec_gru(dec_inp, hidden)
        # print(output.shape) # [1, bs, self.hidden_dim*num_directions]

        output = F.log_softmax(self.out(output[0]), dim = 1)
        # print(output.shape) # [bs, self.vocab_size]
    
        return output, hidden