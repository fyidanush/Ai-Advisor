import pandas as pd
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

data = pd.read_csv('portfolio.csv')
# print(data)
data = torch.tensor([data["0"],data["1"],data["2"],data["3"],data["4"],data["5"],data["6"],data["7"],data["8"],data["9"],data["10"],data["11"]])

X = data[0:3]
Y = data[3:12]

Xt=[]
Yt=[]
for i in range(len(X[0])):
    tempx = ([X[0][i],X[1][i],X[2][i]])
    tempy = ([Y[0][i],Y[1][i],Y[2][i],Y[3][i],Y[4][i],Y[5][i],Y[6][i],Y[7][i],Y[8][i]])
    Xt.append(tempx)
    Yt.append(tempy)
Xt=torch.tensor(Xt)
Yt=torch.tensor(Yt)

Xt = Xt.float()
Yt=Yt.float()

xtrain = Xt[:int(0.9*len(Xt))]
ytrain = Yt[:int(0.9*len(Yt))]
xtest = Xt[int(0.9*len(Xt)):]
ytest = Yt[int(0.9*len(Yt)):]

ex_len = 3
n_ol = 9

bat_sz = 4
n_fl = 64
n_ml = 64
epoch = 10

head_sz = 16

class Reshaper(nn.Module):
    def __init__(self,n):
        self.n = n
        super(Reshaper, self).__init__()

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    

modely = nn.Sequential(
    nn.Embedding(ex_len,n_fl),     #  Gets the input ([10000,8,16])
    Reshaper(2),  # becomes [10000,4,32]  # [inp_sz,blk_sz//2,emb_sz*2]
    nn.Linear(n_fl*2,n_fl),       # takes [32,200]  becomes [10000,4,200]
    nn.Tanh(),
    Reshaper(2),  # becomes [10000,2,400]   # [inp_sz,blk_sz//4,n_fl*2]
    nn.Linear(n_fl*2,n_fl),    # becomes  [10000,2,200]
    nn.Tanh(),
    Reshaper(2),  # becomes [10000,400]  # [inp_sz,n_fl*2]
    nn.Linear(n_fl*2,n_ol),    # become [400,voc_sz]
    nn.Softmax(),
)

optimizer = torch.optim.SGD(modely.parameters(), lr=0.001)


# attention
@torch.no_grad()
def self_attention(x):  

    B,T,C = x.shape
    # B = 4  # block size
    # T = 8  # token (number of tokens in one example)
    # C = 32 # embedding space vector of the each one token
    # x = torch.randn(B,T,C) # example sequence
    # x -> this is how our one batch should look like here 


    # head_sz = 32# this is the splits we do of each token seqence to parellelize the sequence gen

    key = nn.Linear(C,head_sz,bias=False) # this is to make to complexities truly match  
    query =  nn.Linear(C,head_sz,bias=False) 
    value =  nn.Linear(C,head_sz,bias=False) 
    k = key(x)  # [B,T,head_sz]
    q = query(x)  # [B,T,head_sz]
    # this is -- x @ key --  then 
    # upcoming wei is the main token to token connection matrix
    wei = q @ k.transpose(-2,-1)
    # wei[0]  # TxT

    tril = torch.tril(torch.ones(T, T))  # makes a lower triangular matrix of TxT
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

    v = value(x)
    out = wei @ v
    # out[0].shape
    return out

# basically this is what makes up the new and improved input to the neural net



def embed(dim,sqz):
    a="sdjnf"

modely = nn.Sequential(

    nn.Linear(ex_len,n_fl),
    nn.ReLU(),
    nn.Linear(n_fl,n_ml), 
    # nn.Tanh(),
    nn.Linear(n_ml,n_ol),
    # nn.Softmax(),
)

ix = (torch.randperm(len(xtrain)-bat_sz))
optimizer = torch.optim.SGD(modely.parameters(), lr=0.001)

X_Train = torch.zeros([64,16,16])


def accuracy():
    k,d=len(xtest),len(xtest)
    for n in range(len(xtest)):
        omt = ((modely(xtest[n]))).tolist()
        # print(omt.index(max(omt)),ytest[n].item())

        k -= (omt.index(max(omt))==ytest[n].item())
    accu = (((d-k)/d) *100)
    return("Accuracy = " + str(accu))


for e in range(epoch):
    for i in ix:
        xt = xtrain[i:i+bat_sz]
        yt = ytrain[i:i+bat_sz]
        X_train = xt.view([bat_sz,ex_len]) #[4,784]
        Y_train = yt
        # print(X_train,Y_train)
        # break
        
        X_Train = X_Train +self_attention(X_Train)
        
        
        t = modely(X_train)
        loss = F.cross_entropy(t,Y_train) 
        optimizer.zero_grad()

        loss.backward()  # ------------
        optimizer.step()
        
        # print(loss.item())
    # print(accuracy())
    if(e == int(0.8*epoch)):
        optimize = torch.optim.SGD(modely.parameters(), lr=0.0001)



import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import random


app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def index():
    return render_template('index.html', items = "Hello from backend")
    # return "<h1>Hello, World</h1>"

output_value=[0,0,0]

@app.route('/do', methods=['POST'])
def do():
    data = json.loads(request.data)
    print(data)
    if(data['calculate'] == []):
        return jsonify({'output': "Failed"})
    else:
        
        qns = (data['calculate'])
        rmax = 4.4
        smax = 5.25
        dmax = 4

        rout = (qns[0]/rmax)*10
        sout = (qns[1]/smax)*10
        dout = (qns[2]/dmax)*10

        h=torch.tensor([int(rout),int(sout),int(dout)])
        g=modely(h.float())
        j = F.softmax(g).tolist()
        print(j)
        return jsonify({'output': j})
            
    #except:
    adkfj=1237 
        # return jsonify({'output': 0})
    

if __name__ == '__main__':
    app.run(host = '192.168.197.222',port='5000',debug=True)