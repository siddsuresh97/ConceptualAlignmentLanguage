import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TripletLabelModel(nn.Module):
    def __init__(self, encoded_space_dim=64, num_classes=4):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
    
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        ## changed 32*4*4 to 32*2*2
        self.encoder_lin = nn.Sequential(
            nn.Linear(32*2*2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

        ## triplet projection module
        self.decoder_triplet_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 32),
            nn.ReLU(True)
         
        )
        ##labeling module
        self.decoder_labels_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, num_classes),
        )

        ### initialize weights using xavier initialization
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
        img_features = self.encoder_cnn(x)
        img_features = self.flatten(img_features)
        enc_latent = self.encoder_lin(img_features)
        return enc_latent
        
    # def forward(self, x, y=None):
    #     batch_s = x.size(0)
    #     img_features = self.encoder_cnn(x)

    #     img_features = self.flatten(img_features)

    #     enc_latent = self.encoder_lin(img_features)

    #     triplet_latent = self.decoder_triplet_lin(enc_latent)
    #     label = self.decoder_labels_lin(enc_latent)
    #     # label = F.softmax(label,dim=1)
    #     return enc_latent, label

class CustomLoss(nn.Module):
    def __init__(self, margin=10):
        super(CustomLoss, self).__init__()
        self.margin = margin
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, anchor, positive, negative, label, pred_label):
        cosine_sim = torch.nn.CosineSimilarity(1)
        # distance_positive = torch.tensor(1)-cosine_sim(anchor,positive)
   
        # distance_negative = torch.tensor(1)-cosine_sim(anchor,negative)

        # triplet_loss = torch.maximum(distance_positive - distance_negative + self.margin, torch.tensor(0))
        # triplet_loss = torch.sum(triplet_loss)
        triplet_loss = (nn.TripletMarginWithDistanceLoss( distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)))
        triplet_loss = triplet_loss(anchor, positive, negative)
        label_loss = F.binary_cross_entropy_with_logits(pred_label.float(), label.float())
        total_loss = triplet_loss + label_loss
        return triplet_loss, label_loss, total_loss

    
class NCELoss(nn.Module):
    def __init__(self, temperature=0.07, mode='dot'):
        super(NCELoss, self).__init__()
        self.mode = mode
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        # torch.Size([256, 64]) torch.Size([256, 64]) torch.Size([256, 64])
        if self.mode == 'dot':
            positive_similarity = torch.sum(anchor * positive, dim=1)
            negative_similarity = torch.sum(anchor * negative, dim=1)
        elif self.mode == 'cosine':
            cos = torch.nn.CosineSimilarity(1)
            positive_similarity = cos(anchor, positive)
            negative_similarity = cos(anchor, negative)     
    
        logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity.unsqueeze(1)], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
        
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

class TrainModelsNCE(nn.Module):
    def __init__(self, latent_dims, num_classes, mode='dot'):
        super(TrainModelsNCE, self).__init__()
        self.triplet_lab_model = TripletLabelModel(latent_dims, num_classes)
        self.nce_loss = NCELoss(mode=mode)
        self.num_classes = num_classes

    def forward(self, anchor_im, positive_im, negative_im):
        anchor_latent = self.triplet_lab_model(anchor_im)
        positive_latent = self.triplet_lab_model(positive_im)
        negative_latent = self.triplet_lab_model(negative_im)

        return anchor_latent, positive_latent, negative_latent

    def test_epoch(self, test_data):
    # Set evaluation mode for encoder and decoder
        self.eval()
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            test_nce_loss = []
            # total = 0
            # correct = 0
            for anchor_ims, pos_ims, neg_ims, labels in test_data:
                anchor_ims, pos_ims, neg_ims = anchor_ims.to(device), pos_ims.to(device), neg_ims.to(device)
                labels = F.one_hot(labels, num_classes=self.num_classes)
                labels = labels.to(device)
                
                anchor_latent, positive_latent, negative_latent = self.forward(anchor_ims, pos_ims, neg_ims) 
                loss = self.nce_loss(anchor_latent, positive_latent, negative_latent)
                # wandb.log({"val nce loss": loss})
                # total += labels.size(0)
                # correct += (torch.argmax(pred_label, dim = 1) == torch.argmax(labels, dim = 1)).sum().item()
                test_nce_loss.append(loss.item())
        test_nce_loss = sum(test_nce_loss)/len(test_nce_loss)
        # test_accuracy = correct/total
        return test_nce_loss #, test_accuracy

    def test_epoch_calculate_representation_separation(self, test_data):
    # Set evaluation mode for encoder and decoder
        self.eval()
        with torch.no_grad(): # No need to track the gradients
            accuracies = []
            for anchor_ims, pos_ims, neg_ims, labels in test_data:
                anchor_ims, pos_ims, neg_ims = anchor_ims.to(device), pos_ims.to(device), neg_ims.to(device)
                anchor_latent, _, _, = self.forward(anchor_ims, pos_ims, neg_ims) 
          
                anchor_latent = anchor_latent.cpu().numpy()
                anchor_latent = StandardScaler().fit_transform(anchor_latent)
                labels = labels.cpu().numpy()
        
                lm = linear_model.LogisticRegression()
                lm.fit(anchor_latent, labels)
                accuracies.append(lm.score(anchor_latent, labels))
                # wandb.log({'latent separation accuracy':accuracies[-1]})
        accuracy = sum(accuracies)/len(accuracies)
        return accuracy

    def train_epoch(self, train_data, optimizer):
        self.train()
        train_nce_loss = []
        # correct, total = 0, 0
        for anchor_ims, pos_ims, neg_ims, labels in train_data:
            anchor_ims, pos_ims, neg_ims = anchor_ims.to(device), pos_ims.to(device), neg_ims.to(device)
            labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = labels.to(device)

            optimizer.zero_grad()
            anchor_latent, positive_latent, negative_latent = self.forward(anchor_ims, pos_ims, neg_ims) 
            loss = self.nce_loss(anchor_latent, positive_latent, negative_latent)
            # wandb.log({"train nce loss": loss})
            loss.backward()

            optimizer.step()
            train_nce_loss.append(loss.item())
            # total += labels.size(0)
            # correct += (torch.argmax(pred_label, dim = 1) == torch.argmax(labels, dim = 1)).sum().item()
        train_nce_loss = sum(train_nce_loss)/len(train_nce_loss)
        # train_accuracy = correct/total
        return train_nce_loss #, train_accuracy

    def training_loop(self, train_data, test_data, epochs, optimizer, thres_start=0.02, thres_2=0.005):
        train_losses, val_losses = [], []
        separation_accuracies = []
        for _ in tqdm(range(epochs)):
            train_nce_loss = self.train_epoch(train_data, optimizer)
            test_nce_loss = self.test_epoch(test_data)
            separation_accuracy = self.test_epoch_calculate_representation_separation(test_data)
            train_losses.append(train_nce_loss)
            val_losses.append(test_nce_loss)
            separation_accuracies.append(separation_accuracy)
            wandb.log({
                "train nce loss": train_nce_loss, 
                "val nce loss":test_nce_loss, 
                'latent separation accuracy':separation_accuracy
            })
            
            if len(separation_accuracies) >= 2:  # Ensure we have at least 2 epochs for the mean change calculation
                accuracy_increase_from_start = separation_accuracies[-1] - separation_accuracies[0]
                mean_change = abs(separation_accuracies[-1] - separation_accuracies[-2])
                if accuracy_increase_from_start > thres_start and mean_change < thres_2:
                    print("Early stopping criteria met. Stopping training.")
                    return train_losses, val_losses
            
        return train_losses, val_losses


# class TrainModels(nn.Module):
#     def __init__(self, latent_dims, num_classes):
#         super(TrainModels, self).__init__()
#         self.triplet_lab_model = TripletLabelModel(latent_dims, num_classes)
#         self.custom_loss = CustomLoss()
#         self.num_classes = num_classes
    
#     def forward(self, anchor_im, positive_im, negative_im):
#         anchor_latent, anchor_label = self.triplet_lab_model(anchor_im)
#         positive_latent, _ = self.triplet_lab_model(positive_im)
#         negative_latent, _ = self.triplet_lab_model(negative_im)

#         return anchor_latent, positive_latent, negative_latent, anchor_label

#     def test_epoch(self, test_data):
#     # Set evaluation mode for encoder and decoder
#         self.eval()
#         with torch.no_grad(): # No need to track the gradients
#             # Define the lists to store the outputs for each batch
#             test_triplet_loss = []
#             test_label_loss = []
#             test_total_loss = []
#             total = 0
#             correct = 0
#             for anchor_ims, contrast_ims, labels in test_data:
#                 # Move tensor to the proper device
#                 anchor_ims = anchor_ims.to(device)
#                 contrast_ims = contrast_ims.to(device)
#                 labels = F.one_hot(labels, num_classes=self.num_classes)
#                 labels = labels.to(device)
                
#                 anchor_latent, positive_latent, negative_latent, pred_label = self.forward(anchor_ims, anchor_ims, contrast_ims) 
#                 # Append the network output and the original image to the lists
#                 triplet_loss, label_loss, total_loss = self.custom_loss(anchor_latent,
#                                                                 positive_latent, 
#                                                                 negative_latent, 
#                                                                 labels,
#                                                                 pred_label)
#                 total += labels.size(0)
#                 correct += (torch.argmax(pred_label, dim = 1) == torch.argmax(labels, dim = 1)).sum().item()
#                 test_triplet_loss.append(triplet_loss.item())
#                 test_label_loss.append(label_loss.item())
#                 test_total_loss.append(total_loss.item())
#         test_triplet_loss = sum(test_triplet_loss)/len(test_triplet_loss)
#         test_label_loss = sum(test_label_loss)/len(test_label_loss)
#         test_total_loss = sum(test_total_loss)/len(test_total_loss)
#         test_accuracy = correct/total
#         return test_triplet_loss, test_label_loss, test_total_loss, test_accuracy

#     def test_epoch_calculate_representation_separation(self, test_data):
#     # Set evaluation mode for encoder and decoder
#         self.eval()
#         with torch.no_grad(): # No need to track the gradients
#             accuracies = []
#             for anchor_ims, contrast_ims, labels in test_data:
#                 # Move tensor to the proper device
#                 anchor_ims = anchor_ims.to(device)
#                 contrast_ims = contrast_ims.to(device)
#                 # labels = F.one_hot(labels, num_classes=self.num_classes)
#                 # labels = labels.to(device)
#                 anchor_latent, _, _, _ = self.forward(anchor_ims, anchor_ims,contrast_ims) 
#                 # use sklearn to predict labels from anchor_latent
#                 # calculate accuracy
#                 # x's are anchor_latent and y's are labels
#                 # append accuracy to list
#                 # put anchor_latent and labels on cpu and convert to numpy
          
#                 anchor_latent = anchor_latent.cpu().numpy()
#                 ### standard scale the data in anchor_latent before fitting to the model
#                 anchor_latent = StandardScaler().fit_transform(anchor_latent)
#                 labels = labels.cpu().numpy()
                
#                 lm = linear_model.LogisticRegression()
#                 lm.fit(anchor_latent, labels)
#                 # convert labels to sklearn format
#                 accuracies.append(lm.score(anchor_latent, labels))
#         accuracy = sum(accuracies)/len(accuracies)
#         return accuracy

#     def train_epoch(self, train_data, optimizer, train_mode):
#         self.train()
#         train_triplet_loss = []
#         train_label_loss = []
#         train_total_loss = []
#         correct = 0
#         total = 0
#         for anchor_ims, contrast_ims, labels in train_data:
#             anchor_ims = anchor_ims.to(device)
#             contrast_ims = contrast_ims.to(device)
#             labels = F.one_hot(labels, num_classes=self.num_classes)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             anchor_latent, positive_latent, negative_latent, pred_label = self.forward(anchor_ims, anchor_ims, contrast_ims) 
           
#             triplet_loss, label_loss, total_loss = self.custom_loss(anchor_latent,
#                                                                 positive_latent, 
#                                                                 negative_latent, 
#                                                                 labels,
#                                                                 pred_label)
            
#             if train_mode==0:
#                 triplet_loss.backward()
#             elif train_mode==1:
#                 label_loss.backward()
#             elif train_mode==2:
#                 total_loss.backward()

#             optimizer.step()
#             train_triplet_loss.append(triplet_loss.item())
#             train_label_loss.append(label_loss.item())
#             train_total_loss.append(total_loss.item())
#             total += labels.size(0)
#             correct += (torch.argmax(pred_label, dim = 1) == torch.argmax(labels, dim = 1)).sum().item()
#         train_triplet_loss = sum(train_triplet_loss)/len(train_triplet_loss)
#         train_label_loss = sum(train_label_loss)/len(train_label_loss)
#         train_total_loss = sum(train_total_loss)/len(train_total_loss)
#         train_accuracy = correct/total
#         return train_triplet_loss, train_label_loss, train_total_loss, train_accuracy

#     def training_loop(self, train_data, test_data,train_mode,
#                       epochs, optimizer):
#         train_losses = []
#         val_losses = []
#         train_triplet_losses = []
#         val_triplet_losses = []
#         train_label_losses = []
#         val_label_losses = []
#         train_accuracies = []
#         val_accuracies = []
#         latent_separation_accuracy = 0
#         for epoch in tqdm(range(epochs)):
#           train_triplet_loss, train_label_loss, train_total_loss, train_accuracy =self.train_epoch(train_data, optimizer, 
#                                              train_mode)
#           test_triplet_loss, test_label_loss, test_total_loss, test_accuracy = self.test_epoch(test_data)
#           separation_accuracy = self.test_epoch_calculate_representation_separation(test_data)
#           train_losses.append(train_total_loss)
#           val_losses.append(test_total_loss)
#           train_triplet_losses.append(train_triplet_loss)
#           val_triplet_losses.append(test_triplet_loss)
#           train_label_losses.append(train_label_loss)
#           val_label_losses.append(test_label_loss)
#           train_accuracies.append(train_accuracy)
#           val_accuracies.append(test_accuracy)
#           wandb.log({"train triplet loss": train_triplet_loss, 
#             "train label loss":train_label_loss, 
#             "validation triplet loss":test_triplet_loss, 
#             "validation label loss":test_label_loss, 
#             "total train loss":train_total_loss, 
#             "total validation loss":test_total_loss, 
#             "train label accuracy":train_accuracy, 
#             "validation label accuracy":test_accuracy,
#             'latent separation accuracy':separation_accuracy})
#         return train_triplet_losses, train_label_losses, val_triplet_losses, val_label_losses ,train_losses, val_losses, train_accuracies, val_accuracies
