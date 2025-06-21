import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from glob import glob

from train_model import *
from data import *


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(32*8*8, 1)
        


    def forward(self, x):

        x = nn.GELU()(self.conv1(x))
        x = self.pool(x)
        x = nn.GELU()(self.conv2(x))
        x = self.pool(x)
#        x = torch.flatten(x, -1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return torch.sigmoid(x).squeeze()



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(fname, device):

    NN = Model()
    NN.to(device)

    NN.load_state_dict(torch.load(fname, weights_only=True, map_location=torch.device(device)))
    NN.eval()

    return NN



def get_model():
    savemodel = True 
    loadmodel = False

    if savemodel and loadmodel:
        raise Exception("Can't load model and save it again...")
    

    device = get_device()
    print("Found best device: ", device)


    NN = Model()
    NN.to(device)
    testdataloader = None

    if loadmodel:
        NN.load_state_dict(torch.load("../models/model.pth", weights_only=True, map_location=torch.device(device)))
        NN.eval()
    else:
        # create training data
        BATCH_SIZE=16
        dataset = myDataset(device)

        train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
        dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # create some more data for testing
        testdataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        #loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.BCELoss()
        loss_fn.to(device)
        optimizer = torch.optim.Adam(NN.parameters(), lr=1e-4)
        train(NN, dataloader, testdataloader, 50, loss_fn, optimizer, device=device)

        # save model
        if savemodel:
            torch.save(NN.state_dict(), "../models/model.pth")
    
    if testdataloader is None:
        BATCH_SIZE=16
        dataset = myDataset(device)


        train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
        dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # create some more data for testing
        testdataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


    
    return NN, device, testdataloader

if __name__ == "__main__":

    device = get_device()

    dataset = myDataset(device)

    NN, device, test_dl = get_model()


    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    train_features, train_labels = next(iter(dataloader))


    print(NN(train_features))
    print(train_labels)

    fnames = glob("./data/yes/*png")    
#
    nx = int(np.sqrt(len(fnames)))
#
#
#
#    train_features, train_labels = next(iter(test_dl))
#
#    fig, axs = plt.subplots(nx, int(len(fnames)/nx)+1)
    fig, axs = plt.subplots(4, 8, figsize=(12, 5))
    for ii, (feat, lab) in enumerate(zip(train_features, train_labels)):
        print(np.array(feat.cpu()))
        print(float(lab))
        ax = axs.flatten()[ii]
        ax.imshow(np.array(feat.cpu()).squeeze())
        ax.set_xticks([])
        ax.set_yticks([])

        res = NN(feat[None,:,:])
        print(float(res))
        ax.text(0.02, 0.02, f"{res:.2f}", transform=ax.transAxes, color='white')


#        ax.text(0.02, 0.02, f"{float(lab):.2f}", transform=ax.transAxes, color='white')


    train_features, train_labels = next(iter(test_dl))

    for ii, (feat, lab) in enumerate(zip(train_features, train_labels)):
        print(np.array(feat.cpu()))
        print(float(lab))
        ax = axs.flatten()[ii+16]
        ax.imshow(np.array(feat.cpu()).squeeze())
        ax.set_xticks([])
        ax.set_yticks([])
        res = NN(feat[None,:,:])
        print(float(res))
        ax.text(0.02, 0.02, f"{res:.2f}", transform=ax.transAxes, color='white')


#        ax.text(0.02, 0.02, f"{float(lab):.2f}", transform=ax.transAxes, color='white')




    plt.show()
    exit()

#    for feat, lab in test_dl:
#        print(feat, lab)

    for ii, fname in enumerate(fnames):
        img = cv.imread(fname, cv.IMREAD_GRAYSCALE)

        ax = axs.flatten()[ii]
        ax.imshow(img)

        ax.set_xticks([])
        ax.set_yticks([])



        img = torch.tensor(img, dtype=torch.float32)[None,None,:,:]
        print(img.shape)
        img = img.to(device)

        res = NN(img)
        print(float(res))
        ax.text(0.02, 0.02, f"{res:.2f}", transform=ax.transAxes)


    plt.show()

    
