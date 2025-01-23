import os
import torch
import pandas as pd
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Resize
import torchvision
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

import os
base_dir = os.path.abspath('../..')
save_dir = os.path.join(base_dir,'results')
data_dir = os.path.join(base_dir,'data')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TripletLabelModel(nn.Module):
    def __init__(self, encoded_space_dim=64, num_classes=10):
        super().__init__()
        self.encoded_space_dim = encoded_space_dim
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
            ,
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(32*2*2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        self.decoder_triplet_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 32),
            nn.ReLU(True)
        )
        self.decoder_labels_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y=None):
        batch_s = x.size(0)
        img_features = self.encoder_cnn(x)
        img_features = self.flatten(img_features)
        enc_latent = self.encoder_lin(img_features)
        triplet_latent = self.decoder_triplet_lin(enc_latent)
        label = self.decoder_labels_lin(enc_latent)
        return enc_latent, label

class CustomLoss(nn.Module):
    def __init__(self, margin=10):
        super(CustomLoss, self).__init__()
        self.margin = margin
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, anchor, positive, negative, label, pred_label):
        cosine_sim = torch.nn.CosineSimilarity(1)
        triplet_loss = (nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)))
        triplet_loss = triplet_loss(anchor, positive, negative)
        label_loss = F.binary_cross_entropy_with_logits(pred_label.float(), label.float())
        total_loss = triplet_loss + label_loss
        return triplet_loss, label_loss, total_loss

t = TripletLabelModel()
cifar_model_path = '../../data/CIFAR10_NCE_i_1e-05_50.pth'
t.load_state_dict(torch.load(cifar_model_path))

class TrainModels(nn.Module):
    def __init__(self, latent_dims, num_classes, weights_path=None):
        super(TrainModels, self).__init__()
        self.triplet_lab_model = TripletLabelModel(latent_dims, 10)
        if weights_path != None:
            cifar_model_path = '../../data/CIFAR10_NCE_i_1e-05_50.pth'
            self.triplet_lab_model.load_state_dict(torch.load(cifar_model_path))
            self.triplet_lab_model.decoder_labels_lin[4] = nn.Linear(16, num_classes)
            # self.triplet_lab_model.decoder_labels_lin =nn.Linear(self.triplet_lab_model.encoded_space_dim, num_classes)
        self.custom_loss = CustomLoss()
        self.num_classes = num_classes

    def forward(self, anchor_im, positive_im, negative_im):
        anchor_latent, anchor_label = self.triplet_lab_model(anchor_im)
        positive_latent, _ = self.triplet_lab_model(positive_im)
        negative_latent, _ = self.triplet_lab_model(negative_im)
        return anchor_latent, positive_latent, negative_latent, anchor_label

    def test_epoch(self, test_data):
        self.eval()
        with torch.no_grad():
            test_triplet_loss = []
            test_label_loss = []
            test_total_loss = []
            total = 0
            correct = 0
            
            for anchor_ims, option_1_ims, option_2_ims, labels, correct_option in test_data:
                anchor_ims = anchor_ims.to(device)
                option_1_ims = option_1_ims.to(device)
                option_2_ims = option_2_ims.to(device)
                labels = F.one_hot(labels, num_classes=self.num_classes)
                labels = labels.to(device)
                # Create masks for correct options
                mask_option_1 = correct_option == 0
                mask_option_2 = correct_option == 1
                
                correct_labels_1 = labels[mask_option_1]
                correct_labels_2 = labels[mask_option_2]
                # Forward pass with masked options
                anchor_latent_1, positive_latent_1, negative_latent_1, pred_label_1 = self.forward(anchor_ims[mask_option_1], option_1_ims[mask_option_1], option_2_ims[mask_option_1])
                anchor_latent_2, positive_latent_2, negative_latent_2, pred_label_2 = self.forward(anchor_ims[mask_option_2], option_2_ims[mask_option_2], option_1_ims[mask_option_2])
                
                # Combine results
                anchor_latent = torch.cat((anchor_latent_1, anchor_latent_2), dim=0)
                positive_latent = torch.cat((positive_latent_1, positive_latent_2), dim=0)
                negative_latent = torch.cat((negative_latent_1, negative_latent_2), dim=0)
                pred_label = torch.cat((pred_label_1, pred_label_2), dim=0)
                labels = torch.cat((correct_labels_1, correct_labels_2), dim=0)
                triplet_loss, label_loss, total_loss = self.custom_loss(anchor_latent, positive_latent, negative_latent, labels, pred_label)
                total += labels.size(0)
                correct += (torch.argmax(pred_label, dim=1) == torch.argmax(labels, dim=1)).sum().item()
                test_triplet_loss.append(triplet_loss.item())
                test_label_loss.append(label_loss.item())
                test_total_loss.append(total_loss.item())
        test_triplet_loss = sum(test_triplet_loss) / len(test_triplet_loss)
        test_label_loss = sum(test_label_loss) / len(test_label_loss)
        test_total_loss = sum(test_total_loss) / len(test_total_loss)
        test_accuracy = correct / total
        return test_triplet_loss, test_label_loss, test_total_loss, test_accuracy

    def test_epoch_calculate_representation_separation(self, test_data):
        self.eval()
        with torch.no_grad():
            accuracies = []
            for anchor_ims, option_1_ims, option_2_ims, labels, correct_option in test_data:
                anchor_ims = anchor_ims.to(device)
                option_1_ims = option_1_ims.to(device)
                option_2_ims = option_2_ims.to(device)
                anchor_latent, _, _, _ = self.forward(anchor_ims, option_1_ims, option_2_ims)
                anchor_latent = anchor_latent.cpu().numpy()
                anchor_latent = StandardScaler().fit_transform(anchor_latent)
                labels = labels.cpu().numpy()
                lm = linear_model.LogisticRegression()
                lm.fit(anchor_latent, labels)
                accuracies.append(lm.score(anchor_latent, labels))
        accuracy = sum(accuracies) / len(accuracies)
        return accuracy

    def train_epoch(self, train_data, optimizer, train_mode):
        self.train()
        train_triplet_loss = []
        train_label_loss = []
        train_total_loss = []
        correct = 0
        total = 0
        for anchor_ims, option_1_ims, option_2_ims, labels, correct_option in train_data:
            anchor_ims = anchor_ims.to(device)
            option_1_ims = option_1_ims.to(device)
            option_2_ims = option_2_ims.to(device)
            labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = labels.to(device)
            mask_option_1 = correct_option == 0
            mask_option_2 = correct_option == 1
    
            optimizer.zero_grad()
            
            # Create masks for correct options
            # Forward pass with masked options
            anchor_latent_1, positive_latent_1, negative_latent_1, pred_label_1 = self.forward(anchor_ims[mask_option_1], option_1_ims[mask_option_1], option_2_ims[mask_option_1])
            anchor_latent_2, positive_latent_2, negative_latent_2, pred_label_2 = self.forward(anchor_ims[mask_option_2], option_2_ims[mask_option_2], option_1_ims[mask_option_2])
            labels_1 = labels[mask_option_1]
            labels_2 = labels[mask_option_2]
            labels = torch.cat((labels_1, labels_2), dim=0)
            # Combine results
            anchor_latent = torch.cat((anchor_latent_1, anchor_latent_2), dim=0)
            positive_latent = torch.cat((positive_latent_1, positive_latent_2), dim=0)
            negative_latent = torch.cat((negative_latent_1, negative_latent_2), dim=0)
            pred_label = torch.cat((pred_label_1, pred_label_2), dim=0)
            triplet_loss, label_loss, total_loss = self.custom_loss(anchor_latent, positive_latent, negative_latent, labels, pred_label)
            if train_mode == 0:
                triplet_loss.backward()
            elif train_mode == 1:
                label_loss.backward()
            elif train_mode == 2:
                total_loss.backward()
            optimizer.step()
            train_triplet_loss.append(triplet_loss.item())
            train_label_loss.append(label_loss.item())
            train_total_loss.append(total_loss.item())
            total += labels.size(0)
            correct += (torch.argmax(pred_label, dim=1) == torch.argmax(labels, dim=1)).sum().item()
        train_triplet_loss = sum(train_triplet_loss) / len(train_triplet_loss)
        train_label_loss = sum(train_label_loss) / len(train_label_loss)
        train_total_loss = sum(train_total_loss) / len(train_total_loss)
        train_accuracy = correct / total
        return train_triplet_loss, train_label_loss, train_total_loss, train_accuracy

    def training_loop(self, train_data, test_data, train_mode, epochs, optimizer, scheduler):
        train_losses = []
        val_losses = []
        train_triplet_losses = []
        val_triplet_losses = []
        train_label_losses = []
        val_label_losses = []
        train_accuracies = []
        val_accuracies = []
        latent_separation_accuracy = 0
        for epoch in tqdm(range(epochs)):
            train_triplet_loss, train_label_loss, train_total_loss, train_accuracy = self.train_epoch(train_data, optimizer, train_mode)
            test_triplet_loss, test_label_loss, test_total_loss, test_accuracy = self.test_epoch(test_data)
            separation_accuracy = self.test_epoch_calculate_representation_separation(test_data)
            train_losses.append(train_total_loss)
            val_losses.append(test_total_loss)
            train_triplet_losses.append(train_triplet_loss)
            val_triplet_losses.append(test_triplet_loss)
            train_label_losses.append(train_label_loss)
            val_label_losses.append(test_label_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(test_accuracy)
            wandb.log({
                "train triplet loss": train_triplet_loss,
                "train label loss": train_label_loss,
                "validation triplet loss": test_triplet_loss,
                "validation label loss": test_label_loss,
                "total train loss": train_total_loss,
                "total validation loss": test_total_loss,
                "train label accuracy": train_accuracy,
                "validation label accuracy": test_accuracy,
                'latent separation accuracy': separation_accuracy,
                'lr': optimizer.param_groups[0]['lr']
            })
            if scheduler is not None:
                scheduler.step()  # Step the scheduler at the end of each epoch
                # new_lr = optimizer.param_groups[0]['lr']
                # print(f'Epoch {epoch} - New LR: {new_lr}')
        return train_triplet_losses, train_label_losses, val_triplet_losses, val_label_losses, train_losses, val_losses, train_accuracies, val_accuracies

def create_training_dataset(data, label_mapping):
    # Create a copy of the input data
    data_copy = data.copy()
    
    def reshape_array(flat_array):
        return flat_array.reshape(64, 128, 3)
    
    # Validate label_mapping contains non-negative values
    if any(val < 0 for val in label_mapping.values()):
        raise ValueError("Label mapping must contain non-negative values")
    
    # Process images using the copied data
    anchor_ims = torch.tensor(np.stack(data_copy['test_image'].apply(lambda x: reshape_array(np.array(x))).values).transpose(0, 3, 1, 2)).float()
    option_1_ims = torch.tensor(np.stack(data_copy['option_1_image'].apply(lambda x: reshape_array(np.array(x))).values).transpose(0, 3, 1, 2)).float()
    option_2_ims = torch.tensor(np.stack(data_copy['option_2_image'].apply(lambda x: reshape_array(np.array(x))).values).transpose(0, 3, 1, 2)).float()
    
    # Map labels on copied data and validate
    data_copy['trial_type'] = data_copy['trial_type'].map(label_mapping)
    if data_copy['trial_type'].min() < 0:
        raise ValueError("Mapped labels contain negative values")
    
    labels = torch.tensor(data_copy['trial_type']).to(torch.int64)
    correct_options = torch.tensor(data_copy['correct_option']).to(torch.int64)
    resize_transform = transforms.Resize((32, 32))
    
    return TensorDataset(
        resize_transform(anchor_ims),
        resize_transform(option_1_ims),
        resize_transform(option_2_ims),
        labels,
        correct_options
    )

set_A_data = pd.read_parquet(os.path.join(data_dir, 'same_diff_train_exact_match_df_a.parquet'))
set_B_data = pd.read_parquet(os.path.join(data_dir, 'same_diff_train_exact_match_df_b.parquet'))
set_C_data = pd.read_parquet(os.path.join(data_dir, 'same_diff_train_exact_match_df_c.parquet'))
# test_data = pd.read_parquet(os.path.join(data_dir, 'same_diff_train_exact_match_df.parquet'))

plt.imshow(set_A_data['test_image'].iloc[0].reshape(64, 128, 3))

def wandb_init(epochs, lr, train_mode, batch_size, model_number, data_set):
    wandb.login(key='0743dcb3e6cabcef48bfd9b18017ba1cd0722fd7')
    wandb.init(project="ConceptualAlignment2025", entity="sid-academic-team", settings=wandb.Settings(start_method="thread"))
    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_number": model_number,
        "dataset": data_set,
        "train_mode": train_mode,
    }
    train_mode_dict = {0: 'triplet', 1: 'label', 2: 'label_and_triplet'}
    wandb.run.name = f'{data_set}_{train_mode_dict[train_mode]}_{model_number}'
    wandb.run.save()

def main_code(save_dir, num_models, epochs, num_classes, batch_size, lr_dict, latent_dims):
    if os.path.isdir(save_dir):
        pass
    else:
        os.mkdir(save_dir)

    for data_set in ['set_A', 'set_B', 'set_C']:
        for train_mode in tqdm(range(3)):
            for model in range(num_models):
                weights_path = f'../../data/cifar_models/m{model}.pth'
                lr =  lr_dict[train_mode]
                wandb_init(epochs, lr, train_mode, batch_size, model, data_set)
                print(f'LR is {lr}')
                if data_set == 'set_A':
                    train_data = create_training_dataset(set_A_data, label_mapping={'same': 0, 'different': 1})
                elif data_set == 'set_B':
                    train_data = create_training_dataset(set_B_data, label_mapping={'same': 0, 'different': 1})
                elif data_set == 'set_C':
                    train_data = create_training_dataset(set_C_data, label_mapping={'same': 0, 'different': 1})

                # val_data = create_training_dataset(test_data, label_mapping={'same': 0, 'different': 1})
                # Let val_data be a subet of train_data (20%)
                train_data_size = len(train_data)
                indices = list(range(train_data_size))
                split = int(np.floor(0.2 * train_data_size))
                np.random.shuffle(indices)
                train_idx, val_idx = indices[split:], indices[:split]

                # Create new datasets before DataLoader
                train_subset = torch.utils.data.Subset(train_data, train_idx)
                val_subset = torch.utils.data.Subset(train_data, val_idx)

                # Create DataLoaders
                train_loader = torch.utils.data.DataLoader(
                    train_subset, 
                    batch_size=batch_size, 
                    shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    val_subset, 
                    batch_size=batch_size, 
                    shuffle=True
                )

                # Update variable names
                train_data = train_loader
                val_data = val_loader

                train_obj = TrainModels(latent_dims, num_classes, weights_path).to(device)
                optimizer = torch.optim.Adam(train_obj.parameters(), lr=lr, weight_decay=1e-05)
                
                # Define scheduler with 10% warm-up epochs and the rest using cosine annealing
                # warmup_epochs = max(1, int(epochs * 0.1))
                # scheduler = lr_scheduler.SequentialLR(
                #     optimizer,
                #     schedulers=[
                #         lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                #         lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
                #     ],
                #     milestones=[warmup_epochs]
                # )
                scheduler = None
                train_triplet_losses, train_label_losses, val_triplet_losses, val_label_losses, train_losses, val_losses, train_accuracies, val_accuracies = train_obj.training_loop(
                    train_data=train_data,
                    test_data=val_data,
                    epochs=epochs,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_mode=train_mode
                )

                print('validation triplet loss:', val_triplet_losses, 'validation total loss:', val_losses, 'validation accuracy:', val_accuracies)
                train_mode_dict = {0: 'triplet', 1: 'label', 2: 'label_and_triplet'}
                torch.save(train_obj.triplet_lab_model.state_dict(), os.path.join(save_dir, f'{model}_{data_set}_{train_mode_dict[train_mode]}.pth'))
                
                wandb.finish()  # Finish the current WandB run

num_classes = 2
latent_dims = 64
epochs = 1000
lr_dict = {0:0.0005, 1:0.0005, 2:0.0005}
num_models = 10
batch_size = 256
save_dir = save_dir
main_code(save_dir, num_models, epochs, num_classes, batch_size, lr_dict, latent_dims)



#lr for triplet 0.001 might be too high
# lr for label 0.001 might be too low