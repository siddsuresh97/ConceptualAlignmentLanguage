from torch.utils.data import TensorDataset,Dataset
import torch
import os
import wandb
from src import data
from tqdm import tqdm


def wandb_init(epochs, lr, train_mode, batch_size, model_number,data_set):
  wandb.init(project="ConceptualAlignmentLanguage", entity="psych-711")
  wandb.config = {
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": batch_size, 
    # "label_ratio":label_ratio, 
    "model_number": model_number,
    "dataset": data_set,
    "train_mode":train_mode,
  }
  wandb.run.name = f'{data_set}_{train_mode}_{model_number}'
  wandb.run.save()

def train(save_dir, num_models, epochs, num_classes, batch_size,
             lr, latent_dims, model, data_dir):
  if os.path.isdir(os.path.join(save_dir, model.name)):
    pass
  else:
    os.mkdir(os.path.join(save_dir, model.name))
  
  base_set_ims, overlap_20_ims,  overlap_50_ims, validation_ims, base_set_labs, \
           overlap_20_labs, overlap_50_labs, validation_labs = data.shapes3d.load_train_val_data(data_dir)
  for data_set in ['base','overlap_50','overlap_20']:
    for i in tqdm(range(3)):
     # torch.manual_seed(0)
      train_mode=i
      for model_number in range(num_models):
        wandb_init(epochs, lr, train_mode, batch_size, model_number,data_set)

        if data_set=='base':
          train_data = TensorDataset(torch.tensor(base_set_ims.transpose(0,3,1,2)/255).float(), torch.tensor(base_set_labs).to(torch.int64))
        elif data_set=='overlap_50':
          train_data = TensorDataset(torch.tensor(overlap_50_ims.transpose(0,3,1,2)/255).float(), torch.tensor(overlap_50_labs).to(torch.int64))
        elif data_set=='overlap_20':
          train_data = TensorDataset(torch.tensor(overlap_20_ims.transpose(0,3,1,2)/255).float(), torch.tensor(overlap_20_labs).to(torch.int64))


        train_data, val_data = torch.utils.data.random_split(train_data, 
                                                            [18000, 2000])
        train_data = torch.utils.data.DataLoader(train_data, 
                                                batch_size=batch_size,
                                              shuffle=True)
        val_data = torch.utils.data.DataLoader(val_data, 
                                                batch_size=batch_size,
                                              shuffle=True)
        test_data = TensorDataset(torch.tensor(validation_ims.transpose(0,3,1,2)/255).float(), torch.tensor(validation_labs).to(torch.int64))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)
        train_img_loss, train_label_loss, val_img_loss, \
        val_label_loss ,train_losses, val_losses,  train_accuracy, \
        val_accuracy= model.training_loop(train_data = train_data,
                                                            test_data = val_data,
                                                            epochs = epochs,
                                                            optimizer = optimizer, 
                                                            train_mode = train_mode)




          #### To fix:

        # min_x, max_x, min_y, max_y = plot_latent_with_label(autoencoder, 
        #                                                     batch_size, 
        #                                                     data=val_data, 
        #                                                     random_labels = False,
        #                                                     num_classes = num_classes,
        #                                                     num_batches=100)
        # plt.clf()
        # plot_reconstructed_with_labels(autoencoder = autoencoder, 
        #                                r0=(min_x, max_x),
        #                               r1=(min_y, max_y), 
        #                                n=24, random_labels = False)
        # plt.clf()
        # min_x, max_x, min_y, max_y = plot_latent_with_label(autoencoder, 
        #                                                     batch_size, 
        #                                                     data=val_data, 
        #                                                     random_labels = True,
        #                                                     num_classes = num_classes,
        #                                                     num_batches=100)
        # plt.clf()
        # plot_reconstructed_with_labels(autoencoder = autoencoder, 
        #                                r0=(min_x, max_x),
        #                               r1=(min_y, max_y), 
        #                                n=24, random_labels = True)
        # plt.clf()
        # wandb.log({"train_img_loss": train_img_loss, 
        #           "train_label_loss":train_label_loss, 
        #           "val_img_loss":val_img_loss, 
        #           "val_label_loss":val_label_loss, 
        #           "train_losses":train_losses, 
        #           "val_losses":val_losses, 
        #           "train_accuracy":train_accuracy, 
        #           "val_accuracy":val_accuracy})
        torch.save(model.state_dict(), os.path.join(os.path.join(save_dir, model.name),f'{data_set}_{train_mode}_{model_number}'))
        
      
