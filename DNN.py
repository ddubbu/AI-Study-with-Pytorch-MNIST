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
batch_size = 50
# 938 = int(60000/batch_size)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# if you wanna pick one sample
# example_mini_batch_img, example_mini_batch_label  = next(iter(train_loader))
# print(example_mini_batch_img.shape)  # torch.Size([batch_size, 1, 28, 28])


''' hyper parameters '''
# total_batch_num = int(len(train_data) / batch_size)
epochs = 10
lr = 0.001
momentum = 0.5
print_interval = 100
model = DNN_Net()
optimizer = optim.Adam(model.parameters(), lr=lr)


''' Train '''
# 통일성 있게 코드 짜자.
train_epoch_loss = []
train_epoch_acc = []
test_epoch_loss = []
test_epoch_acc = []

for epoch in range(epochs):
    model.train()
    train_batch_loss = []
    train_batch_acc = []
    train_batch_num = len(train_loader)
    print("train_batch_num: ", train_batch_num)
    for batch_idx, (x, target) in enumerate(train_loader):
        if batch_idx == 0:
            print('x.shape', x.shape, 'target.shape', target.shape)  # torch.Size([64, 1, 28, 28]) torch.Size([64])
            print(len(train_loader.dataset))  # 60000

        # if batch_idx == 200:
        #     break

        x, target = Variable(x), Variable(target)
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, target)
        loss.backward()    # calc gradients
        train_batch_loss.append(loss.item()/batch_size*100) # from tensor -> get value loss.item() or loss.data
        optimizer.step()   # update gradients
        prediction = output.argmax(dim=1, keepdims=True)
        accuracy = prediction.eq(target.view_as(prediction)).sum().data/batch_size*100
        train_batch_acc.append(accuracy)
        if batch_idx % print_interval == 0:
            print('epoch: {}\tbatch Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(
                    epoch, batch_idx, train_batch_loss[batch_idx], train_batch_acc[batch_idx]))

    train_epoch_loss.append(np.sum(train_batch_loss)/train_batch_num)
    train_epoch_acc.append(np.sum(train_batch_acc)/train_batch_num)



    ''' Test '''
    model.eval()
    test_batch_loss = []
    test_batch_acc = []
    test_batch_num = len(test_loader)

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_loader):

            # if batch_idx == 10:
            #     break

            x, target = Variable(x), Variable(target)
            output = model(x)
            test_batch_loss.append(loss.item()/batch_size*100)
            prediction = output.argmax(dim=1, keepdims=True)
            accuracy = prediction.eq(target.view_as(prediction)).sum().data/batch_size*100
            test_batch_acc.append(accuracy)


    test_epoch_loss.append(np.sum(test_batch_loss)/test_batch_num)
    test_epoch_acc.append(np.sum(test_batch_acc)/test_batch_num)

    ''' save results to numpy '''
    train_test_result = (train_epoch_loss, test_epoch_loss, train_epoch_acc, test_epoch_acc)
    np.save("result.npy", train_test_result)

    # print("train_epoch_loss:", train_epoch_loss)
    # print("test_epoch_loss:", test_epoch_loss)
    x = np.arange(start=1, stop=len(train_epoch_loss)+1, step=1)

    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x, train_epoch_loss, label='train')
    plt.plot(x, test_epoch_loss, label='test')
    ax1.legend()
    ax1.set(ylabel="Loss", xlabel='epoch')

    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x, train_epoch_acc, label='train')
    plt.plot(x, test_epoch_acc, label='test')
    ax2.legend()
    ax2.set(ylabel="Accuracy", xlabel='epoch')

    plt.show()

torch.save(model, './model.pt')

