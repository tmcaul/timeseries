#%%
import torch
import torch.nn.functional as f
import torchtext.data as data
import torchtext.datasets as datasets

from transformer_nlp import TransformerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Construct the model
max_len = 200

text = data.Field(sequential=True, fix_length=max_len, batch_first=True, lower=True, dtype=torch.long)
label = data.LabelField(sequential=False, dtype=torch.long)

datasets.IMDB.download('data/') #download to the data/ directory if haven't already
ds_train, ds_test = datasets.IMDB.splits(text, label, path='data/imdb/aclImdb/') #define location of data directory
print('train : ', len(ds_train))
print('test : ', len(ds_test))
print('train.fields :', ds_train.fields)

ds_train, ds_valid = ds_train.split(0.9)
print('train : ', len(ds_train))
print('valid : ', len(ds_valid))
print('test : ', len(ds_test))

num_words = 50_000
text.build_vocab(ds_train, max_size=num_words)
label.build_vocab(ds_train)
vocab = text.vocab

batch_size = 164
train_loader, valid_loader, test_loader = data.BucketIterator.splits((ds_train, ds_valid, ds_test), batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False)

#instantiate the transformer classifier model and send it to the correct device
model = TransformerClassifier(num_layers=1, d_model=16, num_heads=1, conv_hidden_dim=64, input_vocab_size=50002, num_answers=2)
model.to(device)

# %% Set up training and evaluation loops
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
epochs = 10
t_total = len(train_loader) * epochs

def train(train_loader, valid_loader):
    
    for epoch in range(epochs):
        train_iterator, valid_iterator = iter(train_loader), iter(valid_loader)
        nb_batches_train = len(train_loader)
        train_acc = 0
        model.train()
        losses = 0.0

        for batch in train_iterator:
            x = batch.text.to(device) #has dimenstions [164, 200]
            y = batch.label.to(device) #has dimenstions [164]
            
            out = model(x)  # ①
            loss = f.cross_entropy(out, y)  # ②
            model.zero_grad()  # ③
            loss.backward()  # ④
            losses += loss.item()
            optimizer.step()  # ⑤
                        
            train_acc += (out.argmax(1) == y).cpu().numpy().mean()
        
        print(f"Training loss at epoch {epoch} is {losses / nb_batches_train}")
        print(f"Training accuracy: {train_acc / nb_batches_train}")
        print('Evaluating on validation:')
        evaluate(valid_loader)


def evaluate(data_loader):
    data_iterator = iter(data_loader)
    nb_batches = len(data_loader)
    model.eval()
    acc = 0 
    for batch in data_iterator:
        x = batch.text.to(device)
        y = batch.label.to(device)
                
        out = model(x)
        acc += (out.argmax(1) == y).cpu().numpy().mean()

    print(f"Eval accuracy: {acc / nb_batches}")

#%% Train
train(train_loader, valid_loader)

#%% Evaluate
evaluate(test_loader)