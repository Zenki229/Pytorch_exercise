from lib import *
class DNN(nn.Module):
    def __init__(self,N1,N2):
        super(DNN,self).__init__()
        self.one_layer=nn.Linear(N1,N2)
    def forward(self,x):
        out=self.act(self.one_layer(x))
        return out
    def act(self,x):
        return x
Model= joblib.load('my_trained_model.pkl')
test_dataset=torch.load('test.pt')
BATCH_SIZE=20
N=1000
# test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
# lossF=nn.MSELoss()
# Epoch = 10
# for epoch in np.arange(Epoch):
#     XY = next(iter(test_loader))
#     X = XY[:, :N]
#     Y = XY[:, N:]
#     out = Model(X)
#     loss = lossF(out, Y)
#     print (f'Epoch [{epoch+1}/{Epoch}], Loss: {loss.item():.4f}')
for p in Model.parameters():
    print(p)