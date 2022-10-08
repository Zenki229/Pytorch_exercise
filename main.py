import torch.utils.data

from lib import *
train_dataset =torch.load('train.pt')
test_dataset=torch.load('test.pt')
BATCH_SIZE=20
N=1000
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
class DNN(nn.Module):
    def __init__(self,N1,N2):
        super(DNN,self).__init__()
        self.one_layer=nn.Linear(N1,N2)
    def forward(self,x):
        out=self.act(self.one_layer(x))
        return out
    def act(self,x):
        return x

Model=DNN(N,N)
learning_rate=0.001
optimizer=torch.optim.Adam(Model.parameters(),lr=learning_rate)
lossF=nn.MSELoss()
Epoch=7000
for epoch in np.arange(Epoch):
    XY=next(iter(train_loader))
    X=XY[:,:N]
    Y=XY[:,N:]
    out=Model(X)
    loss=lossF(out,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/{Epoch}], Loss: {loss.item():.8f}')
joblib.dump(Model, 'my_trained_model.pkl', compress=0)
Epoch = 10
for epoch in np.arange(Epoch):
    XY = next(iter(train_loader))
    X = XY[:, :N]
    Y = XY[:, N:]
    out = Model(X)
    loss = lossF(out, Y)
    print (f'Epoch [{epoch+1}/{Epoch}], Loss: {loss.item():.4f}')