import os
import h5py
import requests
from tqdm import tqdm
import numpy as np


CURRENT_DIR = os.path.join(os.path.abspath(os.getcwd()))

def download_dataset(save_dir = os.path.join(CURRENT_DIR, '../../data/3dshapes')):
    if os.path.isdir(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    print('## DOWNLOADING DATA ####')
    response = requests.get("https://storage.googleapis.com/3d-shapes/3dshapes.h5")
    open(os.path.join(save_dir, '3dshapes.h5'), "wb").write(response.content)


def create_train_val_data(dataset_dir = os.path.join(CURRENT_DIR, '../../data/3dshapes'), data_dir = os.path.join(CURRENT_DIR, '../../data')):
    dataset = h5py.File(os.path.join(dataset_dir,'3dshapes.h5'), 'r')
    images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
    labels = dataset['labels']  # array shape [480000,6], float64
    image_shape = images.shape[1:]  # [64,64,3]
    label_shape = labels.shape[1:]  # [6]
    n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000

    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                        'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                            'scale': 8, 'shape': 4, 'orientation': 15}


    base_set_ims = np.empty((20000,64,64,3),dtype='uint8')
    overlap_50_ims = np.empty((20000,64,64,3),dtype='uint8')
    overlap_20_ims = np.empty((20000,64,64,3),dtype='uint8')
    validation_ims = np.empty((4000,64,64,3),dtype='uint8')

    base_set_labs=np.empty((20000),dtype='uint8')
    overlap_50_labs=np.empty((20000),dtype='uint8')
    overlap_20_labs=np.empty((20000),dtype='uint8')
    validation_labs = np.empty((4000),dtype='uint8')


    for i in tqdm(range(4)):
        shape_inds = np.argwhere(labels[:,4] == i)
        np.random.seed(seed=i)
        sub_inds = np.random.choice(shape_inds.flatten(),10000,replace=False)
        base_set_ims[(i*5000)+0:(i*5000)+5000,:,:,:] = images[sorted(sub_inds[0:5000]),:,:,:]
        base_set_labs[(i*5000)+0:(i*5000)+5000] = i
        print('base_done')
        overlap_50_ims[(i*5000)+0:(i*5000)+2500,:,:,:] = images[sorted(sub_inds[0:2500]),:,:,:]
        overlap_50_ims[(i*5000)+2500:(i*5000)+5000,:,:,:] = images[sorted(sub_inds[5000:7500]),:,:,:]
        overlap_50_labs[(i*5000)+0:(i*5000)+5000] = i


        overlap_20_ims[(i*5000)+0:(i*5000)+1000,:,:,:] = images[sorted(sub_inds[0:1000]),:,:,:]
        overlap_20_ims[(i*5000)+1000:(i*5000)+5000,:,:,:] = images[sorted(sub_inds[5000:9000]),:,:,:]
        overlap_20_labs[(i*5000)+0:(i*5000)+5000] = i

        validation_ims[(i*1000)+0:(i*1000)+1000,:,:,:] = images[sorted(sub_inds[9000:10000]),:,:,:]
        validation_labs[(i*1000)+0:(i*1000)+1000] = i

    np.save(os.path.join(data_dir,'base_set.npy'),base_set_ims)
    np.save(os.path.join(data_dir,'overlap_20.npy'),overlap_20_ims)
    np.save(os.path.join(data_dir,'overlap_50.npy'),overlap_50_ims)
    np.save(os.path.join(data_dir,'validation_set.npy'),validation_ims)

    np.save(os.path.join(data_dir,'base_set_labs.npy'),base_set_labs)
    np.save(os.path.join(data_dir,'overlap_20_labs.npy'),overlap_20_labs)
    np.save(os.path.join(data_dir,'overlap_50_labs.npy'),overlap_50_labs)
    np.save(os.path.join(data_dir,'validation_labs.npy'),validation_labs)

def load_train_val_data(data_dir = os.path.join(CURRENT_DIR,'../../data/3dshapes')):
    base_set_ims = np.load(os.path.join(data_dir,'base_set.npy'))
    overlap_20_ims = np.load(os.path.join(data_dir,'overlap_20.npy'))
    overlap_50_ims = np.load(os.path.join(data_dir,'overlap_50.npy'))
    validation_ims = np.load(os.path.join(data_dir,'validation_set.npy'))

    base_set_labs = np.load(os.path.join(data_dir,'base_set_labs.npy'))
    overlap_20_labs = np.load(os.path.join(data_dir,'overlap_20_labs.npy'))
    overlap_50_labs = np.load(os.path.join(data_dir,'overlap_50_labs.npy'))
    validation_labs = np.load(os.path.join(data_dir,'validation_labs.npy'))
    
    return base_set_ims, overlap_20_ims,  overlap_50_ims, validation_ims, base_set_labs, \
           overlap_20_labs, overlap_50_labs, validation_labs
