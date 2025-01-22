import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger

import matplotlib.pyplot as plt

data = np.load('digitsMnist.npy', allow_pickle = True)
print(data[1])

fig = plt.subplots(5,5, figsize = (10,10))

for i in np.arange(0, 25):
    index = np.random.randint(0, 1000)
    plt.subplot(5, 5, i+1)
    img = data[0][index].reshape(28,28)
    plt.imshow(img, cmap = 'gray')
    plt.title(f'data[0][{index}]')
    plt.axis('off')
plt.show()

indexes = []

for i in range(len(data[1])):
    if (data[1][i] == 6) or (data[1][i] == 8):
        indexes.append(i)

X_forTrain = np.array([data[0][i] for i in indexes])
Y_forTrain = np.array([data[1][i] for i in indexes])

fig = plt.subplots(1,5, figsize = (10,10))

for i in np.arange(0, 5):
    ind = random.randint(0, len(indexes))
    plt.subplot(1, 5, i+1)
    img = X_forTrain[ind]
    res = Y_forTrain[ind]
    plt.imshow(img, cmap = 'gray')
    plt.title(f'X_forTrain[{ind}] | Y_forTrain: {res}', fontsize = 7)
    plt.axis('off')
plt.show()

indexes = []

for i in range(len(data[3])):
    if (data[3][i] == 6) or (data[3][i] == 8):
        indexes.append(i)

X_forVal = np.array([data[2][i] for i in indexes])
Y_forVal = np.array([data[3][i] for i in indexes])

fig = plt.subplots(1,5, figsize = (10,10))

for i in np.arange(0, 5):
    ind = random.randint(0, len(indexes))
    plt.subplot(1, 5, i+1)
    img = X_forVal[ind]
    res = Y_forVal[ind]
    plt.imshow(img, cmap = 'gray')
    plt.title(f'X_forVal[{ind}] | Y_forVal: {res}', fontsize = 7)
    plt.axis('off')
plt.show()

Y_forTrain = np.where(Y_forTrain == 6, 0, 1)
Y_forTrain

Y_forVal = np.where(Y_forVal == 6, 0, 1)
Y_forVal

X_forTrain = X_forTrain/255
X_forVal = X_forVal/255

X_train, X_test, Y_train, Y_test = train_test_split(X_forTrain,
                                                    Y_forTrain,
                                                    test_size = 0.22, random_state = 55)

X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype = torch.long)
Y_test_tensor = torch.tensor(Y_test, dtype = torch.long)

X_val_tensors = torch.tensor(X_forVal, dtype = torch.float32)
Y_val_tensors = torch.tensor(Y_forVal, dtype = torch.long)

trainDataset = TensorDataset(X_train_tensor, Y_train_tensor)
testDataset = TensorDataset(X_test_tensor, Y_test_tensor)
valDataset = TensorDataset(X_val_tensors, Y_val_tensors)

train_loader = DataLoader(trainDataset, batch_size = 5, shuffle = True)
test_loader = DataLoader(testDataset, batch_size = 5, shuffle = False)
val_loader = DataLoader(valDataset, batch_size = 5, shuffle = False)


class SimpleNet(pl.LightningModule):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(784, 472)
        #self.fc2 = nn.Linear(628, 472)
        #self.fc3 = nn.Linear(472, 316)
        #self.fc4 = nn.Linear(472, 160)
        self.fc5 = nn.Linear(472, 2)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        #x = self.relu(self.fc3(x))
        #x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, targets)

        loss = self.loss_fn(outputs, targets)
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)

        return {'test_loss': loss, 'test_acc': acc}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, targets)

        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

        # return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.1)


model = SimpleNet()
logger = TensorBoardLogger('logs/', name = 'my_model')

trainer = pl.Trainer(max_epochs = 25, accelerator = 'cpu', logger = logger, log_every_n_steps = 10)

trainer.fit(model, train_loader, test_loader)
trainer.test(model, val_loader)

