import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
        def __init__(self):
            super().__init__()
            # self.gamma = nn.Parameter(torch.tensor([.5]))
            self.cross_entropy = nn.CrossEntropyLoss()

        def forward(self, img, pred_img, label, pred_label):
            mse = nn.MSELoss()
            #mse_loss_img = ((img - pred_img)**2).sum()
            mse_loss_img = mse(pred_img, img)
            mse_loss_label = self.cross_entropy(pred_label, label.float())
            # loss = mse_loss_img * torch.sigmoid(self.gamma) + \
                # mse_loss_label * (1 - torch.sigmoid(self.gamma))
            loss = mse_loss_img + mse_loss_label#*torch.sigmoid(self.gamma)
            return mse_loss_img, mse_loss_label, loss

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, num_classes):
        self.num_classes = num_classes
        super().__init__()
        ""
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
    
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_labels_lin = nn.Linear(num_classes,  self.num_classes//2)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(7 * 7 * 32 +  self.num_classes//2, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
        
    def forward(self, x, y=None):
        batch_s = x.size(0)
        img_features = self.encoder_cnn(x)
        img_features = self.flatten(img_features)
        if y== None:
            combined = torch.cat((img_features, torch.zeros(batch_s, self.num_classes//2)), dim = -1)
        else:
            label_features = self.encoder_labels_lin(y)
            combined = torch.cat((img_features, label_features), dim = -1)
 
        out = self.encoder_lin(combined)
        return out

class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 7 * 7 * 32+ num_classes//2),
            nn.ReLU(True)
        )
        self.decoder_labels_lin = (nn.Linear( self.num_classes//2,  self.num_classes))
        self.flatten = nn.Flatten(start_dim=1)
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 7, 7))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, 
            padding=1, output_padding=1,dilation=3)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        img_features = x[:, :-( self.num_classes//2)]
        label_features = x[:, -( self.num_classes//2):]
        img_features = self.unflatten(img_features)
        
        img_features = self.decoder_conv(img_features)
        
        img = torch.sigmoid(img_features)
        label = self.decoder_labels_lin(label_features)
       
        label = F.softmax(label,dim=1)
        
        # x = self.decoder_lin(x)
    
        # img_features = self.unflatten(x)
        # img_features = self.decoder_conv(img_features)
     
        # img = torch.sigmoid(img_features)
        # label = self.decoder_labels_lin(self.flatten(img_features))
        return img, label

class ConvAutoencoder(nn.Module):

    def __init__(self, latent_dims, num_classes, device):
        super(ConvAutoencoder, self).__init__()
        self.device = device
        self.encoder = Encoder(latent_dims, num_classes)
        self.decoder = Decoder(latent_dims, num_classes)
        self.custom_loss = CustomLoss()
        self.num_classes = num_classes
        self.name = 'ConvAutoencoder'
    
    def forward(self, x, y=None):
        z = self.encoder(x, y)  ### latent vector
        return self.decoder(z) ### image and label

    def test_epoch(self, test_data):
    # Set evaluation mode for encoder and decoder
        self.eval()
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            test_img_loss = []
            test_label_loss = []
            total_test_loss = []
            for image_batch, label_batch in test_data:
                total = 0
                correct = 0
                # Move tensor to the proper device
                image_batch = image_batch.to(self.device)
                label_batch = F.one_hot(label_batch, num_classes=self.num_classes)
                label_batch = label_batch.to(self.device)
                pred_img, pred_label = self.forward(image_batch, label_batch.float()) 
                # Append the network output and the original image to the lists
                img_loss, label_loss, total_loss = self.custom_loss(image_batch,
                                                                pred_img, 
                                                                label_batch, 
                                                                pred_label)
                total += label_batch.size(0)
                correct += (torch.argmax(pred_label, dim = 1) == torch.argmax(label_batch, dim = 1)).sum().item()
                test_img_loss.append(img_loss.item())
                test_label_loss.append(label_loss.item())
                total_test_loss.append(total_loss.item())
        test_img_loss = sum(test_img_loss)/len(test_img_loss)
        test_label_loss = sum(test_label_loss)/len(test_label_loss)
        total_test_loss = sum(total_test_loss)/len(total_test_loss)
        test_accuracy = correct/total
        return test_img_loss, test_label_loss, total_test_loss, test_accuracy

    def train_epoch(self, train_data, optimizer, train_mode):
        self.train()
        torch.manual_seed(0)
        train_img_loss = []
        train_label_loss = []
        train_loss = []
        correct = 0
        total = 0
        for image_batch, label_batch in train_data:
            image_batch = image_batch.to(self.device)
            # num_training_examples = label_batch.shape[0]
            # num_non_label_training_examples = num_training_examples*(1-training_label_ratio)
            # non_label_training_idx = random.sample(range(num_training_examples),int(num_non_label_training_examples))
            # label_batch[[non_label_training_idx]] = self.num_classes - 1
           
            label_batch = F.one_hot(label_batch, num_classes=self.num_classes)
            label_batch = label_batch.to(self.device)
            optimizer.zero_grad()
            if train_mode==0:
                pred_img, pred_label = self.forward(image_batch) 
            elif train_mode==1:
                pred_img, pred_label = self.forward(image_batch, label_batch.float()) 
            elif train_mode==2:
                pred_img, pred_label = self.forward(image_batch) 
            # Append the network output and the original image to the lists

           
            img_loss, label_loss, total_loss = self.custom_loss(image_batch,
                                                            pred_img, 
                                                            label_batch, 
                                                            pred_label)
            
            
            if train_mode==0:
                img_loss.backward()
            elif train_mode==1:
                total_loss.backward()
            elif train_mode==2:
                label_loss.backward()

            optimizer.step()
            train_img_loss.append(img_loss.item())
            train_label_loss.append(label_loss.item())
            train_loss.append(total_loss.item())
            total += label_batch.size(0)
            correct += (torch.argmax(pred_label, dim = 1) == torch.argmax(label_batch, dim = 1)).sum().item()
        train_img_loss = sum(train_img_loss)/len(train_img_loss)
        train_label_loss = sum(train_label_loss)/len(train_label_loss)
        train_loss = sum(train_loss)/len(train_loss)
        train_accuracy = correct/total
        return train_img_loss, train_label_loss, train_loss, train_accuracy

    def training_loop(self, train_data, test_data,train_mode,
                      epochs, optimizer):
        train_losses = []
        val_losses = []
        train_img_losses = []
        val_img_losses = []
        train_label_losses = []
        val_label_losses = []
        train_accuracies = []
        val_accuracies = []
        for epoch in tqdm(range(epochs)):
          train_img_loss, train_label_loss, train_loss, train_accuracy =self.train_epoch(train_data, optimizer, 
                                             train_mode)
          val_img_loss, val_label_loss, val_loss, val_accuracy = self.test_epoch(test_data)
          train_losses.append(train_loss)
          val_losses.append(val_loss)
          train_img_losses.append(train_img_loss)
          val_img_losses.append(val_img_loss)
          train_label_losses.append(train_label_loss)
          val_label_losses.append(val_label_loss)
          train_accuracies.append(train_accuracy)
          val_accuracies.append(val_accuracy)
          wandb.log({"train_img_loss": train_img_loss, 
            "train_label_loss":train_label_loss, 
            "val_img_loss":val_img_loss, 
            "val_label_loss":val_label_loss, 
            "train_losses":train_loss, 
            "val_losses":val_loss, 
            "train_accuracy":train_accuracy, 
            "val_accuracy":val_accuracy})
        return train_img_losses, train_label_losses, val_img_losses, val_label_losses ,train_losses, val_losses, train_accuracies, val_accuracies
