import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import time
import matplotlib.pyplot as plt
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': '8',
        }   #引用此才能在图中打印中文

plt.rc("font", **font)
start =time.time()
class MyCNN(nn.Module):
    def __init__(self, image_size, num_classes):
        super(MyCNN, self).__init__()
        # conv1: Conv2d -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # conv2: Conv2d -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # fully connected layer
        self.fc = nn.Linear(32768, num_classes)

    def forward(self, x):
        """
        input: N * 3 * image_size * image_size
        output: N * num_classes
        """
        x = self.conv1(x)
        x = self.conv2(x)
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

accs1=[]
def train(model, train_loader, loss_func, optimizer, device):
    """
    train model using loss_fn and optimizer in an epoch.
    model: CNN networks
    train_loader: a Dataloader object with training data
    loss_func: loss function
    device: train on cpu or gpu device
    """
    correct = 0
    total = 0
    total_loss=0
    # train the model using minibatch

    for i, (images, targets) in enumerate(train_loader):

        images = images.to(device)
        targets = targets.to(device)

        # forward
        outputs = model(images)
        _, predicted = torch.max(outputs.data, dim=1)
        loss = loss_func(outputs, targets)
        # backward and optimize
        optimizer.zero_grad()   #梯度清零
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # every 100 iteration, print loss
        if (i + 1) % 10 == 0:
            print("Step [{}/{}] Train Loss: {:.4f}"
                  .format(i + 1, len(train_loader), loss.item()))
        correct += (predicted == targets).sum().item()

        total += targets.size(0)

    accuracy = correct / total
    accs1.append(accuracy)
    print('Accuracy on Train Set: {:.4f} %'.format(100 * accuracy))
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """
    model: CNN networks
    val_loader: a Dataloader object with validation data
    device: evaluate on cpu or gpu device
    return classification accuracy of the model on val dataset
    """
    # evaluate the model
    model.eval()#开启测试模式，主要是关闭BN和dropout
    # context-manager that disabled gradient computation
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, targets) in enumerate(val_loader):

            # device: cpu or gpu
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            # return the maximum value of each row of the input tensor in the
            # given dimension dim, the second return vale is the index location
            # of each maxium value found(argmax)
            _, predicted = torch.max(outputs.data, dim=1)

            correct += (predicted == targets).sum().item()

            total += targets.size(0)
        model.train()#开启训练模式，主要是开启BN和dropout
        accuracy = correct / total
        print('Accuracy on Test Set: {:.4f} %'.format(100 * accuracy))
        return accuracy
#def save_model(model, save_path):
    # save model
    #torch.save(model.state_dict(), save_path)
import matplotlib.pyplot as plt
'''
def show_curve(ys, title):
    """
    plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: loss or accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='#FFA07A')
    plt.axvspan(-1, 21, facecolor='#ADD8E6', alpha=0.5)
    plt.axis([0,21,-0.01,1.5])

    #plt.title('{} curve'.format(title))
    plt.xlabel('迭代次数')
    plt.ylabel('{}'.format(title))
    plt.grid()  # 生成网格
    plt.show()
'''
def show1_curve(ys,yb,yl):
    """
    plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: loss or accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    y1 = np.array(yb)
    y2=np.array(yl)
    #ax.set_xlim(0, 1.00)
    plt.plot(x, y, c='#8FBC8F',label = '测试集',linewidth=2,marker='o')
    plt.plot(x, y1, c='#8FBC8F', linestyle='--', label='训练集',linewidth=2)
    plt.legend(loc=2, bbox_to_anchor=(0,1.1),borderaxespad = 0.)  # 标示不同图形的文本标签图例
    plt.yticks(color='#8FBC8F')
    plt.xlabel('迭代次数')
    plt.ylabel("准确率",color='#8FBC8F')
    #plt.axvspan(-1, 21, facecolor='#ADD8E6', alpha=0.5)
    #plt.axis([-0.05,21,0,1.05])
    #plt.yticks((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.966, 1))
    ax2 = plt.twinx()
    ax2.set_ylabel("损失函数值",color='#FA8072')
    plt.plot(x, y2, c='#FA8072', label='损失曲线',linewidth=2,marker='o')
    plt.legend(loc=1, bbox_to_anchor=(1.0,1.06),borderaxespad = 0.)    #标示不同图形的文本标签图例
    plt.yticks(color='#FA8072')
    #plt.title('{} curve'.format(title))
    #plt.grid()  # 生成网格
    plt.show()

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# mean and std of cifar10 in 3 channels
#cifar10_mean = (0.49, 0.48, 0.45)
#cifar10_std = (0.25, 0.24, 0.26)

# define transform operations of train dataset
train_transform = transforms.Compose([
    # data augmentation
    #transforms.Pad(4),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(32),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    #transforms.Normalize(cifar10_mean, cifar10_std)
    ])

test_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),

   # transforms.Normalize(cifar10_mean, cifar10_std)
    ])

# mini train Cifar10 datasets: 1000 images each class
train_dataset = torchvision.datasets.ImageFolder('C:\\Users\Administrator\Desktop\超声训练集', transform=train_transform)
# mini test Cifar10 datasets: 500 images each class
test_dataset = torchvision.datasets.ImageFolder('C:\\Users\Administrator\Desktop\超声测试集', transform=test_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=10,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=10,
                                          shuffle=True)


def fit(model, num_epochs, optimizer, device):
    """
    train and evaluate an classifier num_epochs times.
    We use optimizer and cross entropy loss to train the model.
    Args:
        model: CNN network
        num_epochs: the number of training epochs
        optimizer: optimize the loss function
    """

    # loss and optimizer
    loss_func = nn.CrossEntropyLoss()

    model.to(device)
    loss_func.to(device)

    # log train loss and test accuracy
    losses = []
    accs = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch + 1, num_epochs))
        # train step
        loss = train(model, train_loader, loss_func, optimizer, device)
        losses.append(loss)

        # evaluate step
        accuracy = evaluate(model, test_loader, device)
        accs.append(accuracy)

    # show curve
    #show_curve(losses, "损失函数值")
    show1_curve(accs,accs1, losses)
# hyper parameters
num_epochs = 20
lr = 0.001
image_size = 128
num_classes = 3
# declare and define an objet of MyCNN
mycnn = MyCNN(image_size, num_classes)
print(mycnn)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(mycnn.parameters(), lr=lr)

# start training on cifar10 dataset
fit(mycnn, num_epochs, optimizer, device)
end = time.time()

print('Running time: %s 秒'%(end-start))