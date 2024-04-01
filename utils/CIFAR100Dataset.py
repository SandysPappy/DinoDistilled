from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from imutils import paths
from PIL import Image
from tqdm import tqdm
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import time
from tqdm import tqdm
from utils import utils
import torch.distributed as dist
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

# custom dataset class
class CIFAR100Dataset(Dataset):
    def __init__(self, root="./data/", preprocessin_fn = None,subset="train"):

        self.preprocessin_fn = ToTensor()
        if preprocessin_fn is not None:
            self.preprocessin_fn = preprocessin_fn

        is_subset_train = True
        if subset=="test":
            is_subset_train = False
        
        self.dataset = CIFAR100(root=root,download=True,train =is_subset_train, transform=self.preprocessin_fn)
        
        string_classes = self.dataset.classes
        # LabelClassName = self.class_id_to_str[self.labels[idx]] 
        # LabelClasId = self.class_str_to_id[LabelClassName]


        self.class_id_to_str = {}
        self.class_str_to_id = {}
        self.image_features= []
        self.labels = []
        self.image_features =[]

        for _, index in self.dataset:
            label = string_classes[index]
            self.labels.append(index)
            if index not in self.class_id_to_str:
                self.class_id_to_str[index] = label
            if label not in self.class_str_to_id:
                self.class_str_to_id[label] = index


        self.isDataTransformed = False
        self.isImageFeaturesExtracted = False

    def getOriginalImage(self, idx):
        Images, labels_ = self.dataset[idx]
        return Images
    
    def getImagePath(self, idx):
        return None
    
    def ExtractImageFeatures(self, preprocsessor, model, device):
        print("Extracting Image features")
        for img_idx in range(len(self.dataset)):
            image, label_ =  self.dataset[img_idx]
            with torch.no_grad():
                inputs = preprocsessor(images=image, return_tensors="pt", do_rescale=False)
                inputs = inputs.to(device)
                outputs = model(**inputs)
                last_hidden_states = outputs[0]
                features = last_hidden_states[:,0,:] # batch size, 257=CLS_Token+256,features_length
                features = features.reshape(features.size(0), -1)
                self.image_features.append(features)
        print("Extracting Image features done")

        if len(self.image_features)==len(self.images):
            self.isImageFeaturesExtracted = True
        else:
            print("Image features are extracted but their lenght doesnt match with total images.")
        
    
    def transformEEGDataDino(self, model, device, pass_eeg=False,preprocessor=None, min_time=0,max_time=460, do_z2_score_norm=False, keep_features_flat=False):
        print(f"Transforming Image data to dino features EEG, pass_eeg is {pass_eeg}")
        # assert pass_eeg==False
        model = model.to(device)

        for i, data in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            image, label = data
            t0 = time.perf_counter()
            model_inputs = None
            if pass_eeg:
                eeg = self.image_features[i].float()
                eeg = eeg.cpu().numpy()
                eeg = self.resizeEEGToImageSize(eeg)
                if do_z2_score_norm:
                    fmean = np.mean(eeg)
                    fstd = np.std(eeg)
                    eeg = (eeg - fmean)/fstd
                model_inputs = torch.from_numpy(eeg)
            else:
                pass
                model_inputs = image
                # print("model_inputs shape", model_inputs.shape)
                # model_inputs = Image.fromarray(model_inputs)
                # if self.preprocessin_fn is not None:
                #     model_inputs = self.preprocessin_fn(model_inputs)
                # else:
                #     if preprocessor is not None:
                #         model_inputs = preprocessor(model_inputs)

            with torch.no_grad():
                feats = model(model_inputs.unsqueeze(0).to(device))
                if not keep_features_flat:
                    dino_f = feats.cpu().numpy()
                    dino_f = dino_f.reshape(128, -1)
                    dino_f = dino_f[:,min_time:max_time]
                    self.image_features.append(torch.from_numpy(dino_f).float())
                else:
                    self.image_features.append(feats.float())

        self.isDataTransformed = True
        print("Transforming Image data to dino EEG features (done)")

    
    @torch.no_grad()
    def extract_features(self,model, data_loader, use_cuda=True, multiscale=False):
        metric_logger = utils.MetricLogger(delimiter="  ")
        features = None
        # for samples, index in metric_logger.log_every(data_loader, 10):
        for EEG,labels,image, index, Image_features in metric_logger.log_every(data_loader, 10):
            samples = image
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            if multiscale:
                feats = utils.multi_scale(samples, model)
            else:
                feats = model(samples).clone()

            # init storage feature matrix
            if dist.get_rank() == 0 and features is None:
                features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
                if use_cuda:
                    features = features.cuda(non_blocking=True)
                print(f"Storing features into tensor of shape {features.shape}")

            # get indexes from all processes
            y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
            y_l = list(y_all.unbind(0))
            y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
            y_all_reduce.wait()
            index_all = torch.cat(y_l)

            # share features between processes
            feats_all = torch.empty(
                dist.get_world_size(),
                feats.size(0),
                feats.size(1),
                dtype=feats.dtype,
                device=feats.device,
            )
            output_l = list(feats_all.unbind(0))
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()

            # update storage feature matrix
            if dist.get_rank() == 0:
                if use_cuda:
                    features.index_copy_(0, index_all, torch.cat(output_l))
                else:
                    features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
        
        for f in features:
            self.image_features.append(f.cpu().numpy())

        self.isDataTransformed = True
        #return features
    
    
    def __len__(self):
        return len(self.dataset)

    
    def __getitem__(self, idx):
        image, label =  self.dataset[idx]
        # Image_features = []

        LabelClassName = self.class_id_to_str[label] 
        LabelClasId = self.class_str_to_id[LabelClassName]

        # if self.preprocessin_fn is not None:
        #     image = self.preprocessin_fn(image)
        
        # if self.isDataTransformed:
        #     Image_features = self.image_features[idx]

        if len(self.image_features)==len(self):
            Image_features = self.image_features[idx]

        return Image_features,{"ClassName": LabelClassName, "ClassId": LabelClasId},image, idx