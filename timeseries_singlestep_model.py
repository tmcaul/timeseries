#%%
import torch
import torch.nn.functional as f

from load_co2 import load
from transformer import TransformerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load the data
max_len=80 #the length of sequence subset to take
batch_size = 164

#X is a subset of the sequence; Y is the next value in the sequence; D is whether the sequence goes up or down
train_data, validation_data = load(r"/Users/tom/Documents/GitHub/timeseries/data/data/AMZN.csv", n_features = 1, n_steps=max_len)

#%% instantiate the transformer classifier model and send it to the correct device
model = TransformerClassifier(num_layers=3, d_model=20, num_heads=2, conv_hidden_dim=64, num_answers=1)
model.to(device)

# %% Set up training and evaluation loops
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
epochs = 10

#%% testing
import matplotlib.pyplot as plt 
epoch=0
b=0

x = torch.tensor(train_data[0][0:124], requires_grad=False, dtype=torch.float).float()
x_embed=model.encoder.embedding(x)

# out, attn = model.encoder.enc_layers[0](x_embed)
out, attn = model(x)
#attn dimensions: [batch_size, head, input, input, layer]
plt.imshow(attn[0,1,:,:,0].detach().numpy()) #this shows how every data point in a sample attends to every other data point in a sample
# plt.plot(attn[0,0,0,:,0].detach().numpy())
#/testing

#%% Set up training and evaluation loops
def train(train_data, validation_data):
    
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
            d = torch.tensor(train_data[2][i1:i2], requires_grad=False, dtype=torch.double).float()
            y = torch.tensor(train_data[1][i1:i2], requires_grad=False, dtype=torch.double).float()
            
            out, attn = model(x)  # ①
            loss = f.mse_loss(out, y)  # ②
            model.zero_grad()  # ③
            loss.backward()  # ④
            losses += loss.item()
            optimizer.step()  # ⑤
                        
            d_pred=torch.tensor(out.squeeze()>x[:,-1].squeeze(), requires_grad=False, dtype=torch.int32)
            d_true=torch.tensor(y.squeeze()>x[:,-1].squeeze(), requires_grad=False, dtype=torch.int32)
            train_err += sum((d_pred-d_true)**2)
        
        print(f"Training loss at epoch {epoch} is {losses / nb_batches_train}")
        print(f"Training error: {train_err // nb_batches_train}")
        print('Evaluating on validation:')
        evaluate(validation_data)


def evaluate(valid_data):
    nb_batches = len(validation_data[0])//batch_size
    model.eval()
    err = 0 

    for b in range(0,nb_batches):

        i1=b*(nb_batches)
        i2=i1+batch_size
        if i2>len(valid_data[0]): i2=-1

        x = torch.tensor(validation_data[0][i1:i2], requires_grad=False, dtype=torch.double).float()
        d = torch.tensor(validation_data[2][i1:i2], requires_grad=False, dtype=torch.double).float()
        y = torch.tensor(validation_data[1][i1:i2], requires_grad=False, dtype=torch.double).float()
                
        out, attn = model(x)
        d_pred=torch.tensor(out.squeeze()>x[:,-1].squeeze(), requires_grad=False, dtype=torch.int32)
        d_true=torch.tensor(y.squeeze()>x[:,-1].squeeze(), requires_grad=False, dtype=torch.int32)
        err += sum((d_pred-d_true)**2)

    print(f"Eval err: {err // nb_batches}")

#%% Train
train(train_data, validation_data)

#%% Evaluate
evaluate(validation_data)
# %%
