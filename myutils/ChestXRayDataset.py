from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms 
import glob
from PIL import Image
import os
import torch
from myutils import utils
import torch.distributed as dist

class ChestXRayDataset(Dataset):
    """ ChestXRay dataset """

    def __init__(self, img_preprocessing_fn, 
                 labels={ "NORMAL": 0 , "PNEUMONIA": 1},
                 rootPath="./datasets/chest_xray/train/", 
                 seed=123,
                 inference_mode=True):
        self.labels = []
        self.images = []
        self.image_features = []
        self.inference_mode = inference_mode

        self.transform = transforms.Compose([ 
            # transforms.PILToTensor(),
            transforms.Resize((224,224), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        
        for label, labelInt in labels.items():
            print(f"Looking for images in : {rootPath}/{label}/*.jpeg")
            # images_list = glob.glob(f"{rootPath}/{label}/*.JPEG")
            images_list = glob.glob(f"{rootPath}/{label}/*.jpeg")
            # images_list += glob.glob(f"{rootPath}/{label}/*.png")
            # images_list += glob.glob(f"{rootPath}/{label}/*.PNG")

            print(f"{label} Images found: {len(images_list)}")

            for imagePath in images_list:
                self.labels.append(labelInt)
                self.images.append(imagePath)


        self.img_preprocessing_fn = img_preprocessing_fn

        
        self.classes = list(set(self.labels))
        self.classes_zeros_list = [0 for k in self.classes]

        self.class_id_to_str = {}
        self.class_str_to_id = {}

        for key,val in labels.items():
            self.class_id_to_str[val] = key
        for key,val in labels.items():
            self.class_str_to_id[key] = val

        self.isDataTransformed = False

    
    @torch.no_grad()
    def extract_features(self,model, data_loader, use_cuda=True, multiscale=False):
        metric_logger = utils.MetricLogger(delimiter="  ")
        features = None
        # for samples, index in metric_logger.log_every(data_loader, 10):
        for Image_features,labels,image, index in metric_logger.log_every(data_loader, 10):
            if image is None:
                continue
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

    
    def getOriginalImage(self, idx):
        ImagePath = self.images[idx][:]
        imageOriginal = Image.open(ImagePath).convert('RGB')
        return imageOriginal
    
    def getImagePath(self, idx):
        ImagePath = self.images[idx][:]
        return ImagePath
    
    def create_one_hot_encoding(self, label):
        index_of_label = self.classes.index(label)
        zerosList = self.classes_zeros_list.copy()
        zerosList[index_of_label] = 1
        return zerosList

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (Image_features,{"ClassName": LabelClassName, "ClassId": Image_features},image, idx)
        """

        image, LabelClassId, Image_features = [], [], []


        image = self.getOriginalImage(idx)
        if image is not None:

            label = self.labels[idx]
            LabelClassName = self.class_id_to_str[label] 
            LabelClassId = self.class_str_to_id[LabelClassName]

            if self.img_preprocessing_fn is not None:
                image = self.img_preprocessing_fn(image)
            
            if self.isDataTransformed:
                Image_features = self.image_features[idx]
            
            if self.inference_mode:
                return Image_features,{"ClassName": LabelClassName, "ClassId": LabelClassId},image, idx
            else:
                return Image_features,LabelClassId,image, idx
        else:
            print(f"image path: {self.getImagePath(idx)} is None")

    
