import pandas as pd
import numpy as np

def load(path, n_features = 1, n_steps_in=80, n_steps_out=80):

    # split a univariate sequence into samples
    def split_sequence(sequence, n_steps_in, n_steps_out):
        X, y, direction = list(), list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            # check if we are beyond the sequence
            if end_ix+n_steps_out >= len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+n_steps_out]
            X.append(seq_x)
            y.append(seq_y)

            # trend=[0,0]
            # #get direction of the sequence
            # if y[-1]>X[-1][-1]:
            #     trend[0]=1.
            # else:
            #     trend[1]=1.
            # direction.append(trend)

        return np.array(X), np.array(y)#, np.array(direction)

    df=pd.read_csv(path,parse_dates=["Date"],index_col="Date")
    date=df.index
    d_date=date-min(date)

    #get the date and response into a nice format
    t=[d.days for _,d in enumerate(d_date)]
    y=df["Adj Close"].to_numpy()

    #split into training and validation sets
    train_lim=int(0.7*len(y))
    t_train=t[0:train_lim]
    y_train=y[0:train_lim]
    t_valid=t[train_lim:]
    y_valid=y[train_lim:]

    #D is 1 in first col if goes up, 1 in second col if goes down.
    X,Y=split_sequence(y_train,n_steps_in,n_steps_out)
    # X = X.reshape((X.shape[0], X.shape[1], n_features))
    # D = D.reshape((D.shape[0], D.shape[1], n_features))

    Xv,Yv=split_sequence(y_valid,n_steps_in,n_steps_out)
    # Xv = Xv.reshape(n_features, Xv.shape[0], Xv.shape[1])
    # Dv = Dv.reshape(n_features, Dv.shape[0], Dv.shape[1])

    return (X,Y), (Xv,Yv)


