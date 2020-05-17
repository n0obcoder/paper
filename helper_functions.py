import torch
import matplotlib.pyplot as plt
import random, pdb, os
import imageio
from sklearn.metrics import accuracy_score#, confusion_matrix
from glob import glob 
import numpy as np

import config as cfg
from dataset import post_process_sequence_batch 

def write_to_board(writer, total_train_loss, train_rec_loss, train_cls_loss, train_cls_acc, recall_tuple_list, c):
    writer.add_scalar('train/total_train_loss', total_train_loss, c)
    writer.add_scalar('train/train_rec_loss', train_rec_loss.item(), c)
    writer.add_scalar('train/train_cls_loss', train_cls_loss.item(), c)

    # writer.add_scalar('train_cls_acc/i_i'  , train_cls_acc[0], c)
    # writer.add_scalar('train_cls_acc/i_j'  , train_cls_acc[1], c)
    # writer.add_scalar('train_cls_acc/i_j_i', train_cls_acc[2], c)

    writer.add_scalars('train_cls_acc', {'i_i'  : train_cls_acc[0],
                                         'i_j'  : train_cls_acc[1],
                                         'i_j_i': train_cls_acc[2]
                                         }, c)

    writer.add_scalar('train_recall/class_0_i_i', recall_tuple_list[0][0].item(), c)
    writer.add_scalar('train_recall/class_1_i_i', recall_tuple_list[0][1].item(), c)

    writer.add_scalar('train_recall/class_0_i_j', recall_tuple_list[1][0].item(), c)
    writer.add_scalar('train_recall/class_1_i_j', recall_tuple_list[1][1].item(), c)

    writer.add_scalar('train_recall/class_0_i_j_i', recall_tuple_list[2][0].item(), c)
    writer.add_scalar('train_recall/class_1_i_j_i', recall_tuple_list[2][1].item(), c)
    c += 1
    return c

def create_necessary_directories():
    losses_dir = cfg.losses_dir
    models_dir = cfg.models_dir

    dir_list = [losses_dir, models_dir]

    for path in dir_list:
        if not os.path.exists(path):
            os.mkdir(path)

def sanity_check_train_dataset(train_dataset):
    # style 1 is pos, 0 is neg
    idx = random.randint(0, len(train_dataset)-1)
    train_data = train_dataset[idx]

    inp_seq_padded = train_data[0].numpy()
    out_seq_padded = train_data[1].numpy()
    
    style          = train_data[3].numpy()
    inp_seq = ' '.join([train_dataset.vocab.id2word[i] for i in inp_seq_padded])
    out_seq = ' '.join([train_dataset.vocab.id2word[i] for i in out_seq_padded])
    
    # print(inp_seq_padded)
    print(f'{inp_seq} has style: {style}')
    print(f'{out_seq} has style: {style}\n')

    # pdb.set_trace()x
    # q()

def show_reconstructions(input_text, reconstruction_i_i = None, reconstruction_i_j = None, reconstruction_i_j_i = None, first_few = 1):
    # first_few: maximum number of reconstructions to be printed
    
    print('\n==================== VIZUALIZING RECONSTRUCTIONS ====================')
    
    for i in range(len(input_text)):
        print(f'ORIGINAL  -> {input_text[i]}')
        if reconstruction_i_i:
            print(f'REC_i_i   -> {reconstruction_i_i[i]}')
        if reconstruction_i_j:
            print(f'REC_i_j   -> {reconstruction_i_j[i]}')
        if reconstruction_i_j_i:
            print(f'REC_i_j_i -> {reconstruction_i_j_i[i]}')
        print('---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---')    

        if i == first_few-1:
            print('=====================================================================\n')
            break

def get_text_reconstruction(decoder_outputs, id2word_list, verbose = False):
    # decoder_outputs: [train_dataset.longest_sequence_length, bs], torch.int64
    # id2word_list   : id2word list

    decoder_outputs = decoder_outputs.cpu().detach().numpy()

    reconstructed_text = []
    for i in range(decoder_outputs.shape[1]): # iterates over the batch
        text = ' '.join([id2word_list[j] for  j in decoder_outputs[:, i]])
        reconstructed_text.append(text)    
        if verbose:
            print(text)

    return reconstructed_text

def decoder_step(dec, decoder_input, decoder_hidden, sequences_lengths, sequences_styles, encoder_outputs_longest_seq, decoder_outputs, output_sequences_batch = None, rec_loss_func = None, reconstruction = False):

    '''
    dec                        : decoder model
    decoder_input              : one time step long decoder input [1, bs]
    decoder_hidden             : starting hidden state of decoder   
    sequences_lengths          : list of len = bs
    sequences_styles           : sequences_styles                                                                              , [max_seq_len_in_batch, bs]
    encoder_outputs_longest_seq: all the encoder outputs for the max sequence length
    
    ***decoder_outputs         : decoder output matrix of shape (cfg.trg_len, cfg.batch_size, train_dataset.vocab.size) initialized to a zero matrix to be populated and returned
    
    reconstruction             : boolean whether reconstruction is to be performed or not (False by default)
    output_sequences_batch     : ground truth for the decoder (required only for reconstruction of the original input sequence)  
    rec_loss                   : starting reconstruction loss (required only for reconstruction of the original input sequence)
    rec_loss_func              : reconstruction loss function (required only for reconstruction of the original input sequence)
    '''

    if reconstruction:
        # randomly choosing whether or not to use teacher forcing
        use_teacher_forcing = True if random.random() < cfg.teacher_forcing_ratio else False
        rec_loss = 0
    else:
        # not going for teacher forcing because we dont have the output_sequences_batch (the ground truth for the decoder)
        use_teacher_forcing = 0
    
    if use_teacher_forcing:
        for i in range(cfg.trg_len):
        # for i in range(encoder_outputs_longest_seq.shape[0]): # loops for train_dataset.longest_sequence_length number of times
            # sequences_styles[0].unsqueeze(dim = 0).shape # [1, bs]
            decoder_output, decoder_hidden = dec(decoder_input, decoder_hidden, sequences_lengths, sequences_styles[0].unsqueeze(dim = 0), encoder_outputs_longest_seq)
            decoder_outputs[i, :, :] = decoder_output # storing the decoder output for all the time-steps
            
            if i < output_sequences_batch.shape[0]:
                rec_loss += rec_loss_func(decoder_output, output_sequences_batch[i]) 
                decoder_input = output_sequences_batch[i].unsqueeze(dim=0)
            else:
                topv, topi = decoder_output.topk(1, dim = 1)                
                decoder_input = topi.detach().transpose(0, 1) # [1, bs] # detach from history as input                

            # print(decoder_output.shape) # [bs, train_dataset.vocab.size]
            # print(decoder_input.shape)  # [1, bs]
    else:
        for i in range(cfg.trg_len):    
            # print(decoder_input.shape)
            decoder_output, decoder_hidden = dec(decoder_input, decoder_hidden, sequences_lengths, sequences_styles[0].unsqueeze(dim = 0), encoder_outputs_longest_seq)                    
            decoder_outputs[i, :, :] = decoder_output # storing the decoder output for all the time-steps

            if reconstruction and i < output_sequences_batch.shape[0]:
                rec_loss += rec_loss_func(decoder_output, output_sequences_batch[i])                    

            topv, topi = decoder_output.topk(1, dim = 1)                
            decoder_input = topi.detach().transpose(0, 1) # [1, bs] # detach from history as input

    _, dec_out_topi = decoder_outputs.topk(1, dim = 2)
    dec_out_topi = dec_out_topi.squeeze()

    # print('dec_out_topi.shape: ', dec_out_topi.shape, dec_out_topi.device, dec_out_topi.requires_grad) 
    # [train_dataset.longest_sequence_length, bs]

    if reconstruction:
        return dec_out_topi, rec_loss, None
    else:
        return dec_out_topi, None    , decoder_hidden

# define a function to count the total number of trainable parameters
def count_parameters(model): 
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

def cal_cnn_classifier_loss(dec_out_topi, style_gt, dec, cnn_classifier, cls_loss_func):
    
    style_gt = style_gt[0].to(torch.float32) # [bs]

    dec_output_dec_emb_matrix = dec.text_embeddings(dec_out_topi).transpose(0, 1).transpose(1, 2)
    # print(dec_output_dec_emb_matrix.shape, dec_output_dec_emb_matrix.requires_grad) # [bs, cfg.dim_emb, train_dataset.longest_sequence_length], True

    # print(dec_output_dec_emb_matrix.shape)
    probs = cnn_classifier(dec_output_dec_emb_matrix)

    cls_loss = cls_loss_func(probs, style_gt)
    
    acc =  accuracy_score(style_gt.cpu(), probs.detach().cpu().round())

    gt = style_gt.cpu()#np.array([1,1,1,0,0,0])
    y  = probs.detach().cpu().round()#np.array([1,0,1,0,1,1])

    idx_class_1 = np.where(gt == 1)
    idx_class_0 = np.where(gt == 0)

    recall_class_1 = sum(y[idx_class_1])/sum(gt)
    recall_class_0 = (len(y[idx_class_0]) - sum(y[idx_class_0]))/len(y[idx_class_0])

    '''
    tn, fp, fn, tp = confusion_matrix(style_gt.cpu(), probs.detach().cpu().round()).ravel()
    accuracy_class_0 = tn/(tn + fp)
    accuracy_class_1 = tp/(tp + fn) 
    '''

    return cls_loss, acc, (recall_class_0, recall_class_1)#, (accuracy_class_0, accuracy_class_1)

def create_gif(dir_path):
    images = []
    for filename in glob(f'{cfg.losses_dir}/*'):
        images.append(imageio.imread(filename))
    
    imageio.mimsave('losses_gif.gif', images)

def save_model(enc, dec, epoch_number):

    torch.save({'enc_state_dict': enc.state_dict(),
                'dec_state_dict': dec.state_dict(), 
                # 'losses': {'total_loss_list': total_loss_list, 
                #            'rec_loss_list': rec_loss_list, 
                #            'cls_loss_list': cls_loss_list, 
                #            },
                'epochs_till_now': epoch_number + 1}, 
                os.path.join(cfg.models_dir, 'model{}.pth'.format(str(epoch_number + 1).zfill(2))))