# -*- coding: utf-8 -*-
# %pip install -q torch_snippets
import torch
import modelo
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch_snippets import *
from torch.utils.data import random_split

################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
################################################################
def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss
################################################################
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(1),
    transforms.Resize((28,28)),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.to(device))
])
# Usar el dataset LPR
# Instanciar la clase Dataset que cargue los datos
dataset = ImageFolder(root='svmsmo_1\\svm_smo\\SVMCode\\Datasets\\BaseOCR_MultiStyle', transform=img_transform)
trn_ds, val_ds = random_split(
    dataset, [0.8, 0.2],
    generator=torch.Generator().manual_seed(42)
)
# Instanciar la clase Dataloader que sea usada luego para el entrenamiento
trn_dl = DataLoader(trn_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)

# y aplique a cada imagen la transformacion img_transform
model = modelo.AutoEncoder(10).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 5

train_loss = []
val_loss = []
for epoch in range(num_epochs):
    N = len(trn_dl)
    tloss = 0.
    for ix, (data, _) in enumerate(trn_dl):
        loss = train_batch(data, model, criterion, optimizer)
        tloss += loss.item()
    train_loss.append(tloss / N)
    N = len(val_dl)
    vloss = 0.
    for ix, (data, _) in enumerate(val_dl):
        loss = validate_batch(data, model, criterion)
        vloss += loss.item()
    val_loss.append(vloss / N)    

####################################################################
# graficar las losses de entrenamiento y de validacion

####################################################################
for _ in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1,2,figsize=(3,3)) 
    show(im[0], ax=ax[0], title='input')
    show(_im[0], ax=ax[1], title='prediction')
    plt.tight_layout()
    plt.show()