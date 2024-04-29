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
# custom dataset class
class Caltech101Dataset(Dataset):
    def __init__(self, images_path, preprocessin_fn = None, filter_label=None, test_split=0.2, random_seed=43, subset="train", isInferenceMode=True):

        image_paths = list(paths.list_images(images_path))
        self.str_labels = []
        self.images = []
        self.image_features = []
        self.isInferenceMode = isInferenceMode

        for img_path in tqdm(image_paths):
            label = img_path.split(os.path.sep)[-2]
            if label == "BACKGROUND_Google":
                continue

            if filter_label is not None:
                if label == filter_label: 
                    self.images.append(img_path.replace("\\", "/"))
                    self.str_labels.append(label)
            else:
                self.images.append(img_path.replace("\\", "/"))
                self.str_labels.append(label)

        lb = LabelEncoder()
        self.labels = lb.fit_transform(self.str_labels)

        self.class_str_to_id = {}
        for intlab, strlab in zip(self.labels,self.str_labels):
            if strlab not in self.class_str_to_id:
                self.class_str_to_id[strlab] = intlab

        self.class_id_to_str = {y: x for x, y in self.class_str_to_id.items()}

        # Create a StratifiedShuffleSplit instance
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_seed)

        # Get the indices for the splits
        for train_index, test_index in sss.split(self.images, self.labels):
            self.train_images = [self.images[i] for i in train_index]
            self.train_labels = [self.labels[i] for i in train_index]
            self.test_images = [self.images[i] for i in test_index]
            self.test_labels = [self.labels[i] for i in test_index]

        
        print(len(set(self.test_labels)), len(set(self.train_labels)))
        assert len(set(self.test_labels))==len(set(self.train_labels))

        if subset == "train":
            self.images = self.train_images
            self.labels = self.train_labels
        elif subset == "test":
            self.images = self.test_images
            self.labels = self.test_labels

        self.preprocessin_fn = preprocessin_fn
        self.isDataTransformed = False
        self.isImageFeaturesExtracted = False

    def getOriginalImage(self, idx):
        ImagePath = self.images[idx][:]
        imageOriginal = Image.open(ImagePath).convert('RGB')
        return imageOriginal
    
    def getImagePath(self, idx):
        ImagePath = self.images[idx][:]
        return ImagePath
    
    def ExtractImageFeatures(self, preprocsessor, model, device):
        print("Extracting Image features")
        for img_idx in range(len(self.images)):
            image =  self.getOriginalImage(img_idx)
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
        
    
    @torch.no_grad()
    def extract_features(self,model, data_loader, use_cuda=True, multiscale=False):
        metric_logger = utils.MetricLogger(delimiter="  ")
        features = None
        # for samples, index in metric_logger.log_every(data_loader, 10):
        for ImageFeatures,labels,image, index in metric_logger.log_every(data_loader, 10):

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
    
    def transformEEGDataDino(self, model, device, pass_eeg=False,preprocessor=None, min_time=0,max_time=460, do_z2_score_norm=False, keep_features_flat=False):
        print(f"Transforming Image data to dino features EEG, pass_eeg is {pass_eeg}")
        # assert pass_eeg==False
        model = model.to(device)

        for i, image_path in tqdm(enumerate(self.images), total=len(self.images)):
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

                model_inputs = self.getOriginalImage(i)

                if self.preprocessin_fn is not None:
                    model_inputs = self.preprocessin_fn(model_inputs)
                else:
                    if preprocessor is not None:
                        model_inputs = preprocessor(model_inputs)

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

    
    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        image =  self.getOriginalImage(idx)
        img_features = []

        LabelClassName = self.class_id_to_str[self.labels[idx]] 
        LabelClasId = self.class_str_to_id[LabelClassName]

        if self.preprocessin_fn is not None:
            image = self.preprocessin_fn(image)

        if self.isDataTransformed:
            img_features = self.image_features[idx]

        if self.isInferenceMode:
            return img_features,{"ClassName": LabelClassName, "ClassId": LabelClasId},image, idx
        else:
            return img_features,LabelClasId,image, idx