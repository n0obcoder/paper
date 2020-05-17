import torch, os, pdb
import numpy as np

from torch.utils.data import Dataset
import pdb
import config as cfg 
from vocab import build_vocab, Vocabulary

def load_sent(path, max_size=-1):
    data = []
    with open(path) as f:
        for line in f:
            if len(data) == max_size:
                break
            data.append(line.split())
    return data
    
class YELP_Dataset(Dataset):
    
    def __init__(self, data_path, train = False, longest_sequence_length= None):

        data0 = load_sent(data_path + '.0')
        data1 = load_sent(data_path + '.1')
        print(f'\n------------------------ Building a Dataset ------------------------')
        print(f'#sents of {data_path}.0 file 0: {len(data0)}') # list of list of tokenized words
        print(f'#sents of {data_path}.1 file 1: {len(data1)}') # list of list of tokenized words
    
        self.data_all   = data0 + data1
        self.style_list = [0 for i in data0] + [1 for i in data1] # data0 is all neg, data1 is all pos
        
        # sorting all the data according to their seq lengths in descending order
        zip_item = zip(self.data_all, self.style_list)
        sorted_item = sorted(zip_item, key=lambda p: len(p[0]), reverse=True)
        tuple_item = zip(*sorted_item)
        self.data_all, self.style_list = [list(t) for t in tuple_item]

        print(f'len(self.data_all)  : {len(self.data_all)}')
        print(f'len(self.style_list): {len(self.style_list)}')

        if train:
            print('\ntrain: True')
            if not os.path.isfile(cfg.vocab):
                print(f'{cfg.vocab} does not exist')
                print('Building Vocab...')
                build_vocab(data0 + data1, cfg.vocab)
            else:
                print(f'{cfg.vocab} already exists')
        
        self.vocab = Vocabulary(cfg.vocab, cfg.embedding_file, cfg.embed_dim)
        print('\nvocabulary size:', self.vocab.size)
        print(f'vocabulary embedding matrix shape: {self.vocab.embedding.shape}')
        # print(type(self.vocab.embedding)) # np array

        self.longest_sequence_length = longest_sequence_length

        if longest_sequence_length is None:
            self.update_the_max_length()

        print(f'self.longest_sequence_length: {self.longest_sequence_length}')
        print(f'--------------------------------------------------------------------')


    def update_the_max_length(self):
        """
        Recomputes the longest sequence constant of the dataset.
        Goes through all the sentences finds the max length.
        """
        print('\nComputing the maximum sentence length...')
        sequences_lengths = list(map(lambda sent_data: len(sent_data),
                                self.data_all))
        
        max_length = max(sequences_lengths)
        self.longest_sequence_length = max_length
                
    def __len__(self):
        return len(self.data_all)
    
    def get_sent_array(self, sent_indices):

        sent_one_hot_array = np.zeros((len(sent_indices), self.vocab.size))
        # print(f'sent_one_hot_array.shape: {sent_one_hot_array.shape}') # [sent_len, vocab_size]

        for idx, sent_idx in enumerate(sent_indices):
            sent_one_hot_array[idx, sent_idx] = 1
        
        return sent_one_hot_array.transpose()

    def pad_sequence(self, input_sequence, max_length, pad_value):

        original_sent_length = len(input_sequence)
        
        padded_sent = np.zeros((max_length))

        padded_sent[:] = pad_value

        padded_sent[:original_sent_length] = input_sequence

        # np array of shape (max_length,)
        return padded_sent

    def __getitem__(self, index):
        
        sent  = self.data_all[index]
        style = self.style_list[index]

        # print(f'>{sent}< has style-{style}')
    
        # Converting the words into indices 
        # NOTE: due to very low occurance of some words, they might not have made it to self.vocab
        sent_indices = [self.vocab.word2id['<go>']] + [self.vocab.word2id[word] if word in self.vocab.word2id.keys() else self.vocab.word2id['<unk>'] for word in sent] + [self.vocab.word2id['<eos>']]
        # print(f'sent_indices: {sent_indices}') # list of len=len(sent)

        sequence_length = len(sent_indices) - 1
        # print(f'sequence_length   : {sequence_length}')

        # Shifted by one time step
        input_sequence        = sent_indices[:-1]
        ground_truth_sequence = sent_indices[1:]

        # pad sequence so that all of them have the same lenght
        # Otherwise the batching won't work
        input_sequence_padded = self.pad_sequence(input_sequence,
                                                  max_length=self.longest_sequence_length + 1, 
                                                  pad_value=0)
        
        ground_truth_sequence_padded = self.pad_sequence(ground_truth_sequence,
                                                      max_length=self.longest_sequence_length + 1,
                                                      pad_value=0)

        # print(torch.LongTensor(input_sequence_padded).shape, torch.LongTensor(input_sequence_padded).dtype)
        # print(torch.LongTensor(ground_truth_sequence_padded).shape, torch.LongTensor(ground_truth_sequence_padded).dtype)
        # print(torch.LongTensor([sequence_length]).shape, torch.LongTensor([sequence_length]).dtype)
        # print(torch.LongTensor([style]).shape, torch.LongTensor([style]).dtype)
        
        # int64 torch.Size([15]), torch.int64
        # int64 torch.Size([15]), torch.int64
        # int64 torch.Size([1]) , torch.int64
        # int64 torch.Size([1]) , torch.int64

        return (torch.LongTensor(input_sequence_padded), 
                torch.LongTensor(ground_truth_sequence_padded),
                torch.LongTensor([sequence_length]),
                torch.LongTensor([style])
                ) 

def post_process_sequence_batch(batch_list):

    input_sequences, output_sequences, lengths, styles = batch_list

    # batch_tuple is a list of len 4, all 4 items are torch tensors
    # print(input_sequences.shape, input_sequences.dtype)
    # print(output_sequences.shape, output_sequences.dtype)
    # print(lengths.shape, lengths.dtype)
    # print(styles.shape, styles.dtype)
    
    # shapes and dtypes being... 
    # [bs, self.longest_sequence_length + 1], torch.int64
    # [bs, self.longest_sequence_length + 1], torch.int64
    # [bs, 1]                               , torch.int64
    # [bs, 1]                               , torch.int64          

    splitted_input_sequence_batch = input_sequences.split(split_size=1) # default dim = 0
    # tuple of len = bs, and every element is a tensor of shape [split_size = 1, self.longest_sequence_length, self.vocab.size]

    splitted_output_sequence_batch = output_sequences.split(split_size=1)
    # tuple of len = bs, and every element is a tensor of shape [split_size = 1, self.longest_sequence_length, self.vocab.size]

    splitted_lengths_batch = lengths.split(split_size=1)
    # tuple of len = bs, and every element is a tensor of shape [split_size = 1, 1]

    splitted_styles_batch = styles.split(split_size=1)
    # tuple of len = bs, and every element is a tensor of shape [split_size = 1, 1]

    training_data_tuples = zip(splitted_input_sequence_batch,
                               splitted_output_sequence_batch,
                               splitted_lengths_batch, 
                               splitted_styles_batch) 

    # for j, jj, jjj in training_data_tuples: 
    #     print(jjj)
    # pdb.set_trace()
    # q()

    training_data_tuples_sorted = sorted(training_data_tuples,
                                         key=lambda p: int(p[2]),
                                         reverse=True)
    # training_data_tuples_sorted is a list of len = bs

    # for j, jj, jjj in training_data_tuples_sorted: 
    #     print(jjj, int(jjj))

    splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch, splitted_styles_batch = zip(*training_data_tuples_sorted)
    # splitted_input_sequence_batch  is a tuple of len = bs. Each element is a tensor of shape [split_size, max_length]
    # splitted_output_sequence_batch is a tuple of len = bs. Each element is a tensor of shape [split_size, max_length]
    # splitted_lengths_batch         is a tuple of len = bs. Each element is a tensor of shape [split_size, 1]
    # splitted_styles_batch          is a tuple of len = bs. Each element is a tensor of shape [split_size, 1]

    input_sequence_batch_sorted  = torch.cat(splitted_input_sequence_batch)                   # tensor shape [bs, max_length]
    output_sequence_batch_sorted = torch.cat(splitted_output_sequence_batch)                  # tensor shape [bs, max_length]
    lengths_batch_sorted         = torch.cat(splitted_lengths_batch)                          # tensor shape [bs, 1]
    styles_batch_sorted          = torch.cat(splitted_styles_batch)                           # tensor shape [bs, 1]
    styles_batch_sorted          = styles_batch_sorted.expand_as(input_sequence_batch_sorted) # tensor shape [bs, max_length]
    # NOTE- expand_as only works if the two tensors are broadcastable

    # '''
    ### WE ARE NOT GOING TO TRIM THE SEQUENCE USING THE SIZE OF LONGEST SEQUENCE IN THE BATCH
    # Here we trim overall data matrix using the size of the longest sequence
    input_sequence_batch_sorted  = input_sequence_batch_sorted[: , :lengths_batch_sorted[0, 0]]
    # tensor shape [bs, max_length_in_this_batch]
    output_sequence_batch_sorted = output_sequence_batch_sorted[:, :lengths_batch_sorted[0, 0]]
    # tensor shape [bs, max_length_in_this_batch]
    styles_batch_sorted = styles_batch_sorted[:, :lengths_batch_sorted[0, 0]]
    # tensor shape [bs, max_length_in_this_batch]
    # '''

    input_sequence_batch_transposed = input_sequence_batch_sorted.transpose(0, 1)
    # tensor shape [max_length_in_this_batch, bs]
    output_sequence_batch_sorted_transposed = output_sequence_batch_sorted.transpose(0, 1)
    # tensor shape [max_length_in_this_batch, bs]
    styles_batch_sorted_transposed = styles_batch_sorted.transpose(0, 1)
    # tensor shape [max_length_in_this_batch, bs]

    # we also need to invert the sequences_styles too
    styles_inverted_batch_sorted_transposed = torch.ones(styles_batch_sorted_transposed.shape, dtype = torch.int64) - styles_batch_sorted_transposed
    # tensor shape [max_length_in_this_batch, bs]

    # pytorch's api for rnns wants lenghts to be list of ints
    lengths_batch_sorted_list = list(lengths_batch_sorted) # converted from tensor to list
    lengths_batch_sorted_list = list(map(lambda x: int(x), lengths_batch_sorted_list)) # converted from list of tensor to list of int

    # print(styles_batch_sorted_transposed.shape)
    # print(styles_inverted_batch_sorted_transposed.shape)
    
    return input_sequence_batch_transposed, output_sequence_batch_sorted_transposed, lengths_batch_sorted_list, styles_batch_sorted_transposed, styles_inverted_batch_sorted_transposed