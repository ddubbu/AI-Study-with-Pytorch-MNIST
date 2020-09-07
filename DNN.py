''' import libraries '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms  # 1 batch = (1, 784)
from torchvision.datasets import MNIST
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
import numpy as np

''' Build Network, 6 layer DNN '''
class DNN_Net(nn.Module):
    def __init__(self):
        super(DNN_Net, self).__init__()
        # weight, bias 따로 설정 안해도 됨?
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)  # output = 10

    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x.view(-1, 784)))  # push reshape input
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        output = F.log_softmax(self.fc6(h5), dim=1)  # softmax

        return output


''' data load '''
batch_size = 50
download_root = 'data'
# Normalize data with mean=0.5, std=1.0
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

# 60000 if train=True, 60000. else 10000.
train_data = MNIST(download_root, transform=mnist_transform, train=True, download=True)
test_data = MNIST(download_root, transform=mnist_transform, train=False, download=True)

# 위 데이터를 batch size로 나눴구나
batch_size = 64
# 938 = int(60000/batch_size)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# if you wanna pick one sample
# example_mini_batch_img, example_mini_batch_label  = next(iter(train_loader))
# print(example_mini_batch_img.shape)  # torch.Size([batch_size, 1, 28, 28])


''' hyper parameters '''
# total_batch_num = int(len(train_data) / batch_size)
epochs = 10
lr = 0.01
momentum = 0.5
print_interval = 100
model = DNN_Net()
optimizer = optim.Adam(model.parameters(), lr=lr)


''' Train '''
model.train()
train_loss = []
train_acc = []

for epoch in range(epochs):
    for batch_idx, (x, target) in enumerate(train_loader):
        if batch_idx == 0:
            print('x.shape', x.shape, 'target.shape', target.shape)  # torch.Size([64, 1, 28, 28]) torch.Size([64])
            print(len(train_loader.dataset))  # 60000

        x, target = Variable(x), Variable(target)
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, target)
        loss.backward()    # calc gradients
        train_loss.append(loss.item())
        # from tensor -> get value loss.item() or loss.data
        optimizer.step()   # update gradients
        prediction = output.argmax(dim=1, keepdims=True)
        accuracy = prediction.eq(target.view_as(prediction)).sum().data/batch_size*100
        train_acc.append(accuracy)
        if batch_idx % print_interval == 0:
            print('epoch: {}\tbatch Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(epoch, batch_idx, loss.item(), accuracy))


    np.save('train_loss.npy', train_loss)
    np.save('train_acc.npy', train_acc)

    plt.plot(train_loss)
    plt.plot(train_acc)

torch.save(model, './model.pt')