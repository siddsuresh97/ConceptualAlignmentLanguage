# %%
# Load packages
###############################################################################
import wandb
import torch
import torchvision
from torch.utils.data import TensorDataset

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

base_dir = os.path.abspath('./')
save_dir = os.path.join(base_dir,'results')
data_dir = os.path.join(base_dir,'data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from module import *
from image_augmentation import *

# %%
# Load Data
###############################################################################
X = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_normal)
Y = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_normal)

X_pos = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmentation)
Y_pos = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_augmentation)

X_images, X_labels = zip(*X)
X_images, X_labels = torch.stack(X_images), torch.tensor(X_labels)
Y_images, Y_labels = zip(*Y)
Y_images, Y_labels = torch.stack(Y_images), torch.tensor(Y_labels)

X_pos, _ = zip(*X_pos)
Y_pos, _ = zip(*Y_pos)
X_pos, Y_pos = torch.stack(X_pos), torch.stack(Y_pos)

X_perm, Y_perm = torch.randperm(len(X)), torch.randperm(len(Y))
X_neg, Y_neg = X_images[X_perm], Y_images[Y_perm]

X_cifar10_triplet_data = TensorDataset(X_images, X_pos, X_neg, X_labels)
Y_cifar10_triplet_data = TensorDataset(Y_images, Y_pos, Y_neg, Y_labels)

# %%
# Train Models
###############################################################################
data_set, loss_name, mode = "CIFAR10", "NCE_i", "dot"

def wandb_init(epochs, lr, batch_size, model_number, data_set, loss):
    wandb.init(
        project="nces",
    )
    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size, 
        # "label_ratio":label_ratio, 
        "model_number": model_number,
        "dataset": data_set,
    }
    wandb.run.name = f'{data_set}_{loss}_{lr}_{epochs}'
    wandb.run.save()

def train_cifar10_nce(save_dir, num_models, epochs, num_classes, batch_size, lr, latent_dims):    
    if os.path.isdir(save_dir):
        pass
    else:
        os.mkdir(save_dir)

    np.random.seed(56)

    wandb_init(epochs, lr, batch_size, num_models, data_set, loss_name)

    train_data = torch.utils.data.DataLoader(X_cifar10_triplet_data, batch_size=batch_size, shuffle=True)
    val_data = torch.utils.data.DataLoader(Y_cifar10_triplet_data, batch_size=batch_size, shuffle=True)
        
    train_obj = TrainModelsNCE(latent_dims, num_classes, mode=mode).to(device) # GPU
    optimizer = torch.optim.Adam(train_obj.parameters(), lr=lr, weight_decay=1e-05)
    train_losses, val_losses = train_obj.training_loop(train_data = train_data,
                                                        test_data = val_data,
                                                        epochs = epochs,
                                                        optimizer = optimizer,
                                                        thres_start = 0.02,
                                                        thres_2 = 0.005)

    torch.save(train_obj.triplet_lab_model.state_dict(), os.path.join(save_dir,f'{data_set}_{loss_name}_{lr}_{epochs}.pth'))


# %%  
wandb.finish()
train_cifar10_nce(save_dir=save_dir,
              num_models=1,
              epochs=50,
              num_classes=10,
              batch_size=256,
              lr=1e-5,
              latent_dims=64)
wandb.finish()

# %%
