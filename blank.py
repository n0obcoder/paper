import os, sys, pdb, shutil

def q(text = '====='):
    print(f'>{text}<')
    sys.exit()

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from models import Encoder, AttentionDecoder, CNN_Classifier, Decoder
from dataset import YELP_Dataset, post_process_sequence_batch 
from helper_functions import show_reconstructions, get_text_reconstruction, decoder_step, count_parameters, cal_cnn_classifier_loss, sanity_check_train_dataset, create_necessary_directories, create_gif, save_model, write_to_board

if cfg.summary_writer:
    print(f'>>> cfg.summary_writer: {cfg.summary_writer}')
    if os.path.exists(cfg.summmary_dir):
        shutil.rmtree(cfg.summmary_dir)

    writer = SummaryWriter(cfg.summmary_dir)

device = cfg.device
print(f'\ndevice: {device}')

create_necessary_directories()

# Make Dataset
train_dataset = YELP_Dataset(cfg.train_data, train = True)
print(f'\nlen(train_dataset)                 : {len(train_dataset)}')
print(f'train_dataset.longest_sequence_length: {train_dataset.longest_sequence_length}')

'''
val_dataset   = YELP_Dataset(cfg.val_data  , train = not True)
print(f'\nlen(val_dataset)                  : {len(val_dataset)}')
print(f'val_dataset.longest_sequence_length : {val_dataset.longest_sequence_length}')
'''

# sanity_check_train_dataset(train_dataset) # SANITY CHECK TO ENSURE CORRECT MAPPING OF SEQUENCES AND STYLES 

# train_data_point = train_dataset[30052] # 30052 has lenght 0

# Make Dataloader
train_dataloader = DataLoader(train_dataset, batch_size = cfg.batch_size, 
                                shuffle= cfg.train_data_shuffle, num_workers=0, drop_last=True)
print(f'len(train_dataloader): {len(train_dataloader)} @bs={cfg.batch_size}')

'''
val_dataloader = DataLoader(val_dataset, batch_size = cfg.batch_size, 
                        shuffle= True, num_workers=0, drop_last=True)
print(f'len(val_dataloader)  : {len(val_dataloader)} @bs={cfg.batch_size}')
'''

enc = Encoder(vocab_size = train_dataset.vocab.size, embed_dim = cfg.embed_dim, style_embed_dim = cfg.style_embed_dim, hidden_dim = cfg.hidden_dim, num_styles = 2, 
                n_layers = cfg.n_layers, pretrained_weights = train_dataset.vocab.embedding, bidirectional = cfg.bidirectional).to(device)

dec = Decoder(vocab_size = train_dataset.vocab.size, embed_dim = cfg.embed_dim, style_embed_dim = cfg.style_embed_dim, hidden_dim = cfg.hidden_dim, num_styles = 2,
                n_layers = cfg.n_layers, pretrained_weights = train_dataset.vocab.embedding, 
                max_seq_length = train_dataset.longest_sequence_length, bidirectional = cfg.bidirectional).to(device)

# dec = AttentionDecoder(vocab_size = train_dataset.vocab.size, embed_dim = cfg.emb_dim, style_emb_dim = cfg.style_emb_dim, hidden_dim = cfg.hidden_dim,
#                         num_styles = 2, n_layers = cfg.n_layers, max_seq_length = train_dataset.longest_sequence_length,
#                         bidirectional = cfg.bidirectional).to(device)

cnn_classifier = CNN_Classifier(in_channel = cfg.hidden_dim).to(device)

enc_params = count_parameters(enc)
# print(f'\nEncoder has {enc_params} Million Paraeters')
dec_params = count_parameters(dec)
cls_params = count_parameters(cnn_classifier)
print(f'\nEncoder, AttentionDecoder and CNN_Classifier together have {enc_params + enc_params + cls_params} Million Paraeters')

encoder_optimizer    = optim.Adam(enc.parameters()           , lr=cfg.learning_rate)    
decoder_optimizer    = optim.Adam(dec.parameters()           , lr=cfg.learning_rate)
classifier_optimizer = optim.Adam(cnn_classifier.parameters(), lr=cfg.learning_rate)    

epochs_number = cfg.epochs_number
show_every    = cfg.show_every
    
rec_loss_func  = nn.NLLLoss(ignore_index = 0)
cls_loss_func  = nn.BCELoss()

min_loss = float('inf')
c = 0
print('\nTRAINING...')
for epoch_number in range(epochs_number):

    print(f'\nEPOCH {str(epoch_number+1).zfill(len(str(epochs_number)))}/{epochs_number}')
    for batch_idx, batch in tqdm(enumerate(train_dataloader)):
        
        post_processed_batch_tuple = post_process_sequence_batch(batch)
        input_sequences_batch, output_sequences_batch, sequences_lengths, sequences_styles, sequences_styles_inverted = post_processed_batch_tuple
        # print(input_sequences_batch.shape, input_sequences_batch.dtype)  
        # print(output_sequences_batch.shape, output_sequences_batch.dtype)
        # print(len(sequences_lengths), type(sequences_lengths))
        # print(sequences_styles.shape, sequences_styles.dtype)
        # print(sequences_styles_inverted.shape, sequences_styles_inverted.dtype)
        
        # [train_dataset.longest_sequence_length, bs], torch.int64 its actually [max_length_in_this_batch, bs]
        # [train_dataset.longest_sequence_length, bs], torch.int64 same
        # list of len = bs                                         same
        # [train_dataset.longest_sequence_length, bs], torch.int64 same
        # [train_dataset.longest_sequence_length, bs], torch.int64 same

        input_sequences_batch     = input_sequences_batch.to(device)
        output_sequences_batch    = output_sequences_batch.to(device) 
        sequences_styles          = sequences_styles.to(device)
        sequences_styles_inverted = sequences_styles_inverted.to(device) 

        input_text = get_text_reconstruction(input_sequences_batch, train_dataset.vocab.id2word, verbose = not True)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        
        encoder_outputs_i, encoder_hidden_i = enc(input_sequences_batch, sequences_styles, sequences_lengths, hidden = None)
        # print(encoder_outputs_i.shape) # [train_dataset.longest_sequence_length, bs, self.hidden_dim*num_directions], torch.float32 not "max_length_in_this_batch"
        # print(encoder_hidden_i.shape) # [cfg.n_layers*(1 + cfg.bidirectional), cfg.batch_size, cfg.hidden_dim]
                
        # Pad encoder_outputs_i 
        encoder_outputs_i_padded = torch.zeros((train_dataset.longest_sequence_length + 1, cfg.batch_size, cfg.hidden_dim*(1+cfg.bidirectional)))
        # print(encoder_outputs_i_padded.shape, encoder_outputs_i_padded.dtype, encoder_outputs_i_padded.requires_grad) # False
        for seq_idx in range(encoder_outputs_i.shape[0]):
            encoder_outputs_i_padded[seq_idx, :, :] = encoder_outputs_i[seq_idx, :, :]
        # print(encoder_outputs_i_padded.shape, encoder_outputs_i_padded.dtype, encoder_outputs_i_padded.requires_grad) # True
     
        # First decoder input will be the SOS token
        decoder_input = torch.tensor([train_dataset.vocab.word2id['<go>']],device=device).expand_as(torch.ones((1, cfg.batch_size)))
        # print(decoder_input.shape)    # [1, bs]

        # We need to store all the decoder outputs
        decoder_outputs_zero_i_i   = torch.zeros((cfg.trg_len, cfg.batch_size, train_dataset.vocab.size), device = device)
        decoder_outputs_zero_i_j   = torch.zeros((cfg.trg_len, cfg.batch_size, train_dataset.vocab.size), device = device)
        decoder_outputs_zero_i_j_i = torch.zeros((cfg.trg_len, cfg.batch_size, train_dataset.vocab.size), device = device)

        # DECODING FOR THE SAME STYLE
        decoder_outputs_i_i, rec_loss_decoder_i_i, _           = decoder_step(dec, decoder_input, encoder_hidden_i, sequences_lengths, sequences_styles,
                                                                        encoder_outputs_i_padded, decoder_outputs_zero_i_i, 
                                                                        output_sequences_batch, rec_loss_func, reconstruction = True)                                                                          

        reconstruction_i_i = get_text_reconstruction(decoder_outputs_i_i, train_dataset.vocab.id2word)
        
        # '''
        # DECODING FOR THE DIFFERENT STYLE
        decoder_outputs_i_j, _, decoder_hidden_i_j             = decoder_step(dec, decoder_input, encoder_hidden_i, sequences_lengths, sequences_styles_inverted,
                                                                        encoder_outputs_i_padded, decoder_outputs_zero_i_j, 
                                                                        reconstruction = False)  

        reconstruction_i_j = get_text_reconstruction(decoder_outputs_i_j, train_dataset.vocab.id2word)

        # print(decoder_outputs_i_j.shape,  decoder_outputs_i_j.dtype) # [cfg.trg_len, bs], torch.int64
        
        cls_loss_decoder_outputs_i_i, acc_i_i, recall_tuple_i_i = cal_cnn_classifier_loss(decoder_outputs_i_i, sequences_styles         , dec, cnn_classifier, cls_loss_func)    
        cls_loss_decoder_outputs_i_j, acc_i_j, recall_tuple_i_j = cal_cnn_classifier_loss(decoder_outputs_i_j, sequences_styles_inverted, dec, cnn_classifier, cls_loss_func)
        
        # '''
        # BACKWARD TRANSFER
        
        encoder_outputs_j, encoder_hidden_j = enc(decoder_outputs_i_j, sequences_styles_inverted[0].expand_as(decoder_outputs_i_j), sequences_lengths = [cfg.trg_len]*len(sequences_lengths), hidden = decoder_hidden_i_j)
        # print(encoder_outputs_j.shape) # [train_dataset.longest_sequence_length, bs, cfg.hidden_dim]

        # DECODING FOR THE SAME STYLE
        decoder_outputs_i_j_i, rec_loss_decoder_i_j_i, _           = decoder_step(dec, decoder_input, encoder_hidden_j, sequences_lengths, sequences_styles[0].expand_as(decoder_outputs_i_j),
                                                                    encoder_outputs_j, decoder_outputs_zero_i_j_i, 
                                                                    output_sequences_batch, rec_loss_func, reconstruction = True)                                                                                              
        
        reconstruction_i_j_i = get_text_reconstruction(decoder_outputs_i_j_i, train_dataset.vocab.id2word, verbose = False)

        cls_loss_decoder_outputs_i_j_i, acc_i_j_i, recall_tuple_i_j_i = cal_cnn_classifier_loss(decoder_outputs_i_j_i, sequences_styles       , dec, cnn_classifier, cls_loss_func)        
        # '''

        # rec_loss = rec_loss_decoder_i_i
        # cls_loss = 10*(cls_loss_decoder_outputs_i_i + cls_loss_decoder_outputs_i_j)
        rec_loss = rec_loss_decoder_i_i + rec_loss_decoder_i_j_i
        cls_loss = 10*(cls_loss_decoder_outputs_i_i + cls_loss_decoder_outputs_i_j + cls_loss_decoder_outputs_i_j_i)

        rec_loss.backward()
        cls_loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        classifier_optimizer.step()

        total_loss = rec_loss.item() + cls_loss.item()
        # print('-', total_loss)
        c = write_to_board(writer, total_loss, rec_loss, cls_loss, (acc_i_i, acc_i_j, acc_i_j_i), [recall_tuple_i_i, recall_tuple_i_j, recall_tuple_i_j_i], c)

        if batch_idx%show_every == 0:# and batch_idx != 0:
            show_reconstructions(input_text = input_text    , reconstruction_i_i = reconstruction_i_i    , reconstruction_i_j = reconstruction_i_j    , reconstruction_i_j_i = reconstruction_i_j_i    , first_few = 1)

        # Save the model
        if min_loss > total_loss:
            save_model(enc, dec, epoch_number)
            min_loss = total_loss

writer.close()

create_gif(cfg.losses_dir)