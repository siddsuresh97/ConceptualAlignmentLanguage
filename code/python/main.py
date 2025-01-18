import argparse, os
import torch
from src.data import shapes3d
from src import training

DEFAULT_DIR = os.path.join(os.path.abspath(os.getcwd()), '../')
DEFAULT_DATASET_DIR = os.path.join(DEFAULT_DIR, "data/3dshapes")
DEFAULT_MODEL_WEIGHTS_DIR =  os.path.join(DEFAULT_DIR, "results")
DEFAULT_MODEL = 'ConvAutoencoder'
DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('--download_dataset', '-v', type=str, 
                        default = "F", help=""" Specify if you want to download the 3dshapes.h5""")
    parser.add_argument('--dataset_download_dir', type=str, 
                         default = DEFAULT_DATASET_DIR, help=""" Specify where you want to download the 3dshapes.h5""")
    parser.add_argument('--train_val_create', type=str, 
                         default = "F", help=""" Create train_val splits""")
    parser.add_argument('--model_save_dir', type=str, 
                         default = DEFAULT_MODEL_WEIGHTS_DIR, help=""" Specify where you want to save trained model weights""")
    parser.add_argument('--num_models', default = 1,
                        type=int, help="""num of models to train""")
    parser.add_argument('--num_classes', default = 4,
                        type=int, help="""num of pred classes in the dataset""")
    parser.add_argument('--batch_size', default = 64,
                        type=int, help="""training and val batch_size""")
    parser.add_argument('--epochs', default = 30,
                        type=int, help="""number of training epochs of each model""")
    parser.add_argument('--lr', default = 0.005,
                        type=int, help="""lr of ADAM optimizer""")
    parser.add_argument('--latent_dims', default = 10,
                        type=int, help="""Dim of autoencoder bottleneck layer""")
    parser.add_argument('--model', default = DEFAULT_MODEL,
                        type=str, help="""Type of model architecture""")
    parser.add_argument('--device', default = DEVICE,
                        type=str, help="""GPU DEVICE""")
    parser.add_argument('--train', default = "F",
                        type=str, help="""Specify if you want to train the models""")
    args = parser.parse_args()
    if args.download_dataset== "T":
        shapes3d.download_dataset(save_dir = args.dataset_download_dir)

    if args.train_val_create == "T":
        shapes3d.create_train_val_data(dataset_dir = args.dataset_download_dir, data_dir = os.path.join(DEFAULT_DIR, 'data'))
    if args.train == "T":
        training.train(args.model_save_dir, args.num_models, args.epochs, args.num_classes, args.batch_size,
             args.lr, args.latent_dims, args.model, args.device, 
             os.path.join(DEFAULT_DIR, 'data'))
    else:
        print('Did not train any model. If you want to train models, specify --train flag as "T" ')


if __name__=="__main__":
    main()