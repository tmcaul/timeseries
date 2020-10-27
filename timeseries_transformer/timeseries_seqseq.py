# Timeseries transformer encoder / LSTM decoder - TPM 23/10/20
#%% Import modules
#data
import os

os.chdir(r'/Users/tom/Documents/GitHub/timeseries/data')
from load_co2 import load

#model
import torch
import torch.nn.functional as f
from transformer import Transformer_seq2seq, Transformer_LSTMdec, Transformer

#for plotting
import matplotlib.pyplot as plt
from plotting import plot_profiles, live_plot, torch_imshow
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

%matplotlib inline

# %% Load the data
seq_len=60 #the length of sequence subset to take
output_seq_len=60
batch_size = 100

#X is a subset of the sequence; Y is the next value in the sequence; D is whether the sequence goes up or down
train_data, validation_data = load(r"/Users/tom/Documents/GitHub/timeseries/data/data/AMZN.csv", n_features = 1, n_steps_in=seq_len, n_steps_out=seq_len)

#shuffle these data
ind=torch.randperm(train_data[0].shape[0])
train_data[0][:,:]=train_data[0][ind,:]
train_data[1][:,:]=train_data[1][ind,:]

#%% instantiate the transformer classifier model and send it to the correct device
# model = Transformer_seq2seq(num_layers=3, num_rnn_layers=3, d_model=20, num_heads=2, conv_hidden_dim=64, num_answers=80)
# model = Transformer_LSTMdec(num_layers=2, num_rnn_layers=2, d_model=20, num_heads=2, conv_hidden_dim=64, num_answers=output_seq_len)

#TO DO - implement masking
model = Transformer(num_layers=2, d_model=40, num_heads=2, conv_hidden_dim=64, num_answers=output_seq_len)

model.to(device)

#%% testing
x = torch.tensor(train_data[0][0:124], requires_grad=False, dtype=torch.float).float()
y = torch.tensor(train_data[1][0:124], requires_grad=False, dtype=torch.float).float()

# x_embed=model.encoder.embedding(x)
# out, attn = model.encoder.enc_layers[0](x_embed)
out, attn = model(x)
plot_profiles(x,y,out,123)

#attn dimensions: [batch_size, head, input, input, layer]
#plt.imshow(attn[0,1,:,:,0].detach().numpy()) #this shows how every data point in a sample attends to every other data point in a sample
#plt.plot(attn[0,0,0,:,0].detach().numpy())

#%% Set up training and evaluation loops

#set up importance factor for loss
loss_importance_factor=torch.tensor([output_seq_len-n for n in range(0,output_seq_len)],requires_grad=False, dtype=torch.float)
# loss_importance_factor=torch.exp(loss_importance_factor)**0.5
loss_importance_factor=loss_importance_factor/sum(loss_importance_factor)

def train(train_data, validation_data, epochs, plot=True):
    
    if plot==True:
        plotdata = collections.defaultdict(list)

    for epoch in range(epochs):
        nb_batches_train = len(train_data[0])//batch_size
        train_err = 0
        model.train()
        losses = 0.0

        for b in range(0,nb_batches_train):

            i1=b*(nb_batches_train)
            i2=i1+batch_size
            if i2>len(train_data[0]): i2=-1

            x = torch.tensor(train_data[0][i1:i2], requires_grad=False, dtype=torch.double).float()
            y = torch.tensor(train_data[1][i1:i2,0:output_seq_len], requires_grad=False, dtype=torch.double).float()
            
            out, attn = model(x)  # ①
            loss = f.mse_loss(out*loss_importance_factor, y*loss_importance_factor, reduce=True, reduction='sum')  # ②
            model.zero_grad()  # ③
            loss.backward()  # ④
            losses += loss.item()
            optimizer.step()  # ⑤
                        
            # d_pred=torch.tensor(out.squeeze()>x[:,-1].squeeze(), requires_grad=False, dtype=torch.int32)
            # d_true=torch.tensor(y.squeeze()>x[:,-1].squeeze(), requires_grad=False, dtype=torch.int32)
            # train_err += sum((d_pred-d_true)**2)
        
        #take average over batches for losses
        losses=losses/nb_batches_train
        
        # print(f"Training loss at epoch {epoch} is {losses}")
        # print('Evaluating on validation:')
        val_loss=evaluate(validation_data)

        if plot==True:
            #update the training plot
            plotdata['Training'].append(losses)
            plotdata['Validation'].append(val_loss)
            live_plot(plotdata)
        
        else:
            print(f"Training loss at epoch {epoch} is {losses}")
            print(f"Validation error at epoch {epoch} is {val_loss}")

        

def evaluate(validation_data):
    nb_batches = len(validation_data[0])//batch_size
    model.eval()
    torch.no_grad()
    losses=0.0

    for b in range(0,nb_batches):

        i1=b*(nb_batches)
        i2=i1+batch_size
        if i2>len(validation_data[0]): i2=-1

        x = torch.tensor(validation_data[0][i1:i2], requires_grad=False, dtype=torch.double).float()
        y = torch.tensor(validation_data[1][i1:i2,0:output_seq_len], requires_grad=False, dtype=torch.double).float()
                
        out, attn = model(x)
        # d_pred=torch.tensor(out.squeeze()>x[:,-1].squeeze(), requires_grad=False, dtype=torch.int32)
        # d_true=torch.tensor(y.squeeze()>x[:,-1].squeeze(), requires_grad=False, dtype=torch.int32)
        # err += sum((d_pred-d_true)**2)

        loss=f.mse_loss(out,y, reduce=True, reduction='mean')
        losses+=loss.item()
    
    losses=losses/nb_batches

    return losses

#%% Train
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
epochs = 20
train(train_data, validation_data, epochs, plot=True)

# %% Visualise results
x = torch.tensor(train_data[0][0:10], requires_grad=False, dtype=torch.float).float()
out, attn = model(x)
plot_profiles(x,y,out,4)

# %% Histograms of activations
# w,_=model.encoder(x)

# w,_=model.encoder(x)
# plt.hist(w.detach().numpy().flatten())

# %%
