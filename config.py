import os, sys, pdb, torch

def device_override():
    print('\nDEVICE OVERRIDE TO CPU')
    return torch.device("cpu")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = device_override()

train = True
dev   = True
test  = False

summary_writer = not False

data_dir = os.path.join('data', 'yelp')
train_data  = os.path.join(data_dir, 'sentiment.train')
val_data    = os.path.join(data_dir, 'sentiment.dev')
test_data   = os.path.join(data_dir, 'sentiment.test')

losses_dir = 'losses'
models_dir = 'models'
summmary_dir   = 'summary'

max_train_size = -1 # max number of sentences to be kept in the dataset 

train_data_shuffle = True

vocab = os.path.join('tmp', 'yelp.vocab')

embedding_file = os.path.join('tmp', 'yelp.d100.emb.txt')
embed_dim   = 100
style_embed_dim = 64
trg_len = 20

batch_size = int(128) # 256  
show_every = 200

learning_rate = 5e-4
epochs_number = 4
teacher_forcing_ratio = 0.5

bidirectional =   False
n_layers = 2 # it must be > 1 if gru_dropout > 0
gru_dropout = 0.15
hidden_dim = 512