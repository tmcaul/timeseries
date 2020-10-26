#%% This is a modified implementation of a multiheaded attention mechanism in PyTorch from Alfredo Canziani (NYU) at https://atcold.github.io/pytorch-Deep-Learning/en/week12/12-3/

import torch 
from torch import nn
import torch.nn.functional as f
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_Softargmax = nn.Softmax  # fix wrong name


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, p, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input
        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0
        self.d_k = d_model // self.num_heads
        # These are still of dimension d_model. They will be split into number of heads 
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)
        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0) 
        k_length = K.size(-2) 
        # Scaling by d_k so that the soft(arg)max doesnt saturate
        Q = Q / np.sqrt(self.d_k)                         # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2,3))          # (bs, n_heads, q_length, k_length)
        A = nn_Softargmax(dim=-1)(scores)   # (bs, n_heads, q_length, k_length)
        # Get the weighted average of the values
        H = torch.matmul(A, V)     # (bs, n_heads, q_length, dim_per_head)
        return H, A 
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (heads X depth)
        Return after transpose to put in shape (batch_size X num_heads X seq_length X d_k)
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        """
        Combine the heads again to get (batch_size X seq_length X (num_heads times d_k))
        """
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
    
    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size()
        # After transforming, split into num_heads 
        Q = self.split_heads(self.W_q(X_q), batch_size)  # (bs, n_heads, q_length, dim_per_head)
        K = self.split_heads(self.W_k(X_k), batch_size)  # (bs, n_heads, k_length, dim_per_head)
        V = self.split_heads(self.W_v(X_v), batch_size)  # (bs, n_heads, v_length, dim_per_head)
        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)
        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size)    # (bs, q_length, dim)
        # Final linear layer  
        H = self.W_h(H_cat)          # (bs, q_length, dim)
        return H, A


class CNN(nn.Module):
    def __init__(self, d_model, hidden_dim, p):
        super().__init__()
        self.k1convL1 = nn.Linear(d_model,    hidden_dim)
        self.k1convL2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.k1convL2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, conv_hidden_dim, p=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, p)

        #this is a convolution because every x_i member of the input set passes its hidden representation h_i into the CNN separately
        self.cnn = CNN(d_model, conv_hidden_dim, p)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, x):
        attn_output, attn = self.mha(x, x, x)
        
        out1 = self.layernorm1(x + attn_output)
        cnn_output = self.cnn(out1)
        out2 = self.layernorm2(out1 + cnn_output)
        return out2, attn


def create_sinusoidal_embeddings(nb_p, dim, E):
    theta = np.array([
        [p / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for p in range(nb_p)
    ])
    E[:, 0::2] = torch.FloatTensor(np.sin(theta[:, 0::2]))
    E[:, 1::2] = torch.FloatTensor(np.cos(theta[:, 1::2]))
    E.detach_()
    E.requires_grad = False
    E = E.to(device)


#create sinusoidal embeddings - need additional positional encodings to make this work
class Embeddings(nn.Module):
    def __init__(self, d_model, max_position_embeddings, p):
        super().__init__()
        #self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        create_sinusoidal_embeddings(nb_p=max_position_embeddings, dim=d_model, E=self.position_embeddings.weight)

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)                      # (bs, max_seq_length)
        
        # Word embeddings not needed for timeseries data
        # Get word embeddings for each input id
        # word_embeddings = self.word_embeddings(input_ids)                   # (bs, max_seq_length, dim)
        
        # Get position embeddings for each position id 
        position_embeddings = self.position_embeddings(position_ids)        # (bs, max_seq_length, dim)
        
        # For timeseries, embeddings are just position embeddings
        embeddings = position_embeddings  # (bs, max_seq_length, dim)
        
        # Layer norm 
        embeddings = self.LayerNorm(embeddings)             # (bs, max_seq_length, dim)
        return embeddings


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_hidden_dim, maximum_position_encoding, p=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embeddings(d_model, maximum_position_encoding, p)
        self.enc_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(d_model, num_heads, ff_hidden_dim, p))
        
    def forward(self, x):
        x = torch.unsqueeze(x,-1).repeat(1,1,self.d_model) + self.embedding(x) #Add the val to every row of the positional embedding ; # Transform to (batch_size, input_seq_length, d_model)

        for i in range(self.num_layers):
            #feed the input through the network
            x, a = self.enc_layers[i](x)
            
            #keep track of the attention layers
            if i==0:
                attention=a
                attention=torch.unsqueeze(attention,-1)
            else:
                attention=torch.cat((attention,torch.unsqueeze(a,-1)),-1)

        return x, attention  # (batch_size, input_seq_len, d_model)


class TransformerClassifier(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, num_answers):
        super().__init__()
        
        #define the multiheaded attention encoder
        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim, maximum_position_encoding=10000)
        #and a dense classification layer on top
        self.dense = nn.Linear(d_model, num_answers)

    def forward(self, x):
        x, attention = self.encoder(x)
        x, _ = torch.max(x, dim=1)
        x = self.dense(x)
        return x, attention


class LinearDecoderLayer(nn.Module):
    def __init__(self, input_dims, output_dims, act, norm):
        super().__init__()
        self.layer=nn.ModuleList()
        self.layer.append(nn.Linear(input_dims,output_dims))
        self.layer.append(act)
        self.layer.append(norm)

    def forward(self, x):
        for i in range(len(self.layer)):
            #feed the input through the network
            x= self.layer[i](x)
        return x



class LinearDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_answers):
        super().__init__()
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_answers=num_answers
        self.act=nn.ReLU()
        self.norm=nn.BatchNorm1d
        # self.norm=nn.Identity()

        self.dec_layers = nn.ModuleList()

        dim_increase_per_layer=int((num_answers-d_model)//(num_layers+0.5))
        input_dims = [d_model + dim_increase_per_layer*n for n in range(0,num_layers)]
        output_dims = [input_dims[n]+dim_increase_per_layer for n in range(0, num_layers-1)]
        output_dims.append(num_answers)

        for n in range(num_layers-1):
            # self.dec_layers.append(LinearDecoderLayer(input_dims[n], output_dims[n], self.act, self.norm))
            self.dec_layers.append(LinearDecoderLayer(input_dims[n], output_dims[n], self.act, self.norm(output_dims[n])))
        self.dec_layers.append(LinearDecoderLayer(input_dims[n+1], output_dims[n+1], self.act, nn.Identity()))

    def forward(self, x):
        for i in range(self.num_layers):
            #feed the input through the network
            x= self.dec_layers[i](x)
        return x


class Transformer_seq2seq(nn.Module):
    def __init__(self, num_layers, num_rnn_layers, d_model, num_heads, conv_hidden_dim, num_answers):
        super().__init__()
        
        #define the multiheaded attention encoder
        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim, maximum_position_encoding=10000)
        #and a LSTM RNN on top
        # self.LSTM = torch.nn.LSTM(d_model, hidden_size=d_model, num_layers=num_rnn_layers, bidirectional=False, bias=False)
        self.layernorm=torch.nn.LayerNorm(d_model, eps=1e-12)
        # self.decoder = torch.nn.Linear(d_model,num_answers, bias=True)
        self.decoder = LinearDecoder(num_layers, d_model, num_answers)

    def forward(self, x, n):
        #encode the input
        x, attention = self.encoder(x)

        #max pooling
        x,_ = torch.max(x, dim=1)

        x = self.decoder(x)

        # #add an extra dim to beginning of sequence (for LSTM)
        # # x=torch.unsqueeze(x,0)

        # #preallocate the output
        # y=torch.zeros(x.shape[1],n)

        # #run the RNN for the number of requested times
        # for i in range(0,n):
        #     x, _ = self.LSTM(x)        
        #     #layernorm
        #     x = self.layernorm(x)
        #     #and decode
        #     out = self.decoder(x)

        #     out=out.squeeze()
        #     #and append
        #     y[:,i]=out

        return x, attention


class Transformer_LSTMdec(nn.Module):
    def __init__(self, num_layers, num_rnn_layers, d_model, num_heads, conv_hidden_dim):
        super().__init__()
        
        #define the multiheaded attention encoder
        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim, maximum_position_encoding=10000)
        #and a LSTM RNN on top
        self.LSTM = torch.nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_rnn_layers, bidirectional=False, bias=True, batch_first=True)
        self.layernorm=torch.nn.LayerNorm(d_model, eps=1e-12)
        self.decoder = torch.nn.Linear(d_model,1, bias=True)
        # self.decoder = LinearDecoder(num_layers, d_model, num_answers)

    def forward(self, x, n):
        #encode the input
        x, attention = self.encoder(x)

        #preallocate the output
        y=torch.zeros(x.shape[0],n)

        #run the RNN for the number of requested times
        for i in range(0,n):
            x, _ = self.LSTM(x)        
            #layernorm
            # x = self.layernorm(x)
            #and decode
            out,_ = torch.max(x, dim=1)
            out = self.decoder(out)

            out=out.squeeze()
            #and append
            y[:,i]=out

        return y, attention
