from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

def plot_profiles(x,y,out,batchno):
    #plot profiles
    X=x[batchno,:].detach().numpy()
    Ytrue=y[batchno,:].detach().numpy()
    Y=out[batchno,:].detach().numpy()

    #add the final element of X to Y for plotting purposes
    Y2true=np.insert(Ytrue,0,X[-1])
    Y2model=np.insert(Y,0,X[-1])

    fig,ax=plt.subplots()
    inputdata, = ax.plot(range(0,len(X)),X,'r') #input data
    modeloutput, = ax.plot(range(len(X)-1,len(X)+len(Y2model)-1),Y2model,'b') #model output
    outputdata, = ax.plot(range(len(X)-1,len(X)+len(Y2true)-1),Y2true,'g') #real output
    ax.legend((inputdata,modeloutput,outputdata),('Input', 'Model', 'True'))


def live_plot(data_dict, title=''):
    clear_output(wait=True)
    plt.figure()
    for label,data in data_dict.items():
        plt.plot(data, label=label, marker='o')
    plt.title(title)
    plt.grid(False)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='center left') # the plot evolves to the right

    plt.show();


def torch_imshow(tensor):
    plt.imshow(tensor.detach().numpy()) #this shows how every data point in a sample attends to every other data point in a sample