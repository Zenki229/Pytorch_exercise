from lib import *
N=1000
# A=torch.eye(N)*2
# for i in np.arange(N-1):
#     A[i,i+1]=-1
#     A[i+1,i]=-1
# torch.save(A,'A.pt')
A=torch.load('A.pt')
# Count=1
# TrainData=torch.zeros((Count,2*N))
# # TrainData=torch.load('train.pt')
# for i in np.arange(Count):
#     x=torch.rand(N)
#     y=torch.matmul(A,x)
#     aux=torch.cat((x,y)).reshape(1,-1)
#     TrainData[i,:]=aux
#     #TrainData=torch.cat((TrainData,aux),dim=0)
# torch.save(TrainData,'train.pt')
# TestData=torch.zeros((Count,2*N))
# #TestData=torch.load('test.pt')
# for i in np.arange(Count):
#     x=torch.rand(N)
#     y=torch.matmul(A,x)
#     aux=torch.cat((x,y)).reshape(1,-1)
#     TestData[i,:]=aux
#     # TestData=torch.cat((TestData,aux),dim=0)
# torch.save(TestData,'test.pt')
Count=10000
TestData=torch.load('test.pt')
TrainData=torch.load('train.pt')
for i in np.arange(Count):
    x=torch.rand(N)
    y=torch.matmul(A,x)
    aux=torch.cat((x,y)).reshape(1,-1)
    TrainData=torch.cat((TrainData,aux),dim=0)
    x = torch.rand(N)
    y = torch.matmul(A, x)
    aux = torch.cat((x, y)).reshape(1, -1)
    TestData = torch.cat((TestData, aux), dim=0)
torch.save(TrainData,'train.pt')
torch.save(TestData,'test.pt')

