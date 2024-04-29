import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torchvision import models
from utils.DinoModel import DinoModel, dino_args
from utils.ChestXRayDataset import ChestXRayDataset
from utils.Caltech101Dataset import Caltech101Dataset
from utils.CIFAR100Dataset import CIFAR100Dataset
from utils.CIFAR10Dataset import CIFAR10Dataset
from utils import utils
from utils.utils import NpEncoder
import time
import os
import json
import numpy as np
import faiss
import timm

def initDinoV1Model(model_to_load, FLAGS, checkpoint_key="teacher", use_back_bone_only=False):
    dino_args.pretrained_weights = model_to_load
    dino_args.output_dir = FLAGS.log_dir
    dino_args.checkpoint_key = checkpoint_key
    dino_args.use_cuda = torch.cuda.is_available()
    dinov1_model = DinoModel(dino_args, use_only_backbone=use_back_bone_only)
    dinov1_model.eval()
    return dinov1_model

# Define the student model for knowledge distillation
class StudentModel(nn.Module):
    def __init__(self, num_features=384):
        super(StudentModel, self).__init__()
        self.efficientnet = models.efficientnet_v2_s(weights=None)
        # self.efficientnet.classifier[-1].out_features = num_features
        self.efficientnet.classifier = torch.nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_features, bias=True)
        )
        """
        self.efficientnet.classifier =  Sequential(
        (0): Dropout(p=0.2, inplace=True)
        (1): Linear(in_features=1280, out_features=1000, bias=True) # Replaced 1000 by 384
        )
        """

    def forward(self, x):
        x = self.efficientnet(x)
        return x

# Define the student model for knowledge distillation
class EfficientNetV2Embeddings(nn.Module):
    def __init__(self, output_dim=384, dropout_rate=0.5):
        super(EfficientNetV2Embeddings, self).__init__()
        self.base_model = timm.create_model('efficientnetv2_rw_s', pretrained=True, features_only=True)
        feature_dim = self.base_model.feature_info[-1]['num_chs']
        self.batch_norm = nn.BatchNorm1d(feature_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding_layer = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        features = self.base_model(x)[-1]
        pooled_features = features.mean([2, 3])
        norm_features = self.batch_norm(pooled_features)
        dropped_out_features = self.dropout(norm_features)
        embeddings = self.embedding_layer(dropped_out_features)
        return embeddings
    

if __name__=="__main__":


    parser = argparse.ArgumentParser('Efficient DINO')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=32,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='DINO',
                        help='Directory to put logging.')
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        help='type of mode train or test')
    parser.add_argument('--dino_base_model_weights',
                        type=str,
                        default="./models/pretrains/dino_deitsmall8_pretrain_full_checkpoint.pth",
                        help='dino based model weights')
    parser.add_argument('--dino_custom_model_weights',
                        type=str,
                        default="./weights/dinoxray/checkpoint.pth",
                        help='dino based model weights')
    parser.add_argument('--efficient_distilled_model_weights',
                        type=str,
                        default="./weights/best_model_epoch_39_effiv2_dinov2.pth",
                        help='efficient distilled model weights')
    parser.add_argument('--dataset_root',
                        type=str,
                        default="./datasets/chest_xray",
                        help='dataset directory root')
    parser.add_argument('--search_gallery',
                        type=str,
                        default="train",
                        help='dataset in which images will be searched')
    parser.add_argument('--topK',
                        type=int,
                        default=5,
                        help='Top-k paramter, defaults to 5')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")


    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    

    utils.init_distributed_mode(args=FLAGS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    os.makedirs(f"{FLAGS.log_dir}/output", exist_ok=True)

    TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL = 0.7
    SEED_FOR_RANDOM_SPLIT = 43

    # Load DINOv1 model (you need to replace this with your own DINOv2 model)
    # dinov1_model = initDinoV1Model(model_to_load=FLAGS.dino_custom_model_weights,FLAGS=FLAGS,checkpoint_key="teacher", use_back_bone_only=True)
    # dinov1_model.to(device)

    # Initialize the student model
    student_model = EfficientNetV2Embeddings(output_dim=384)
    # student_model = StudentModel(num_features=384)  # dinov1 small
    if os.path.exists(FLAGS.efficient_distilled_model_weights):
        efficient_weights = torch.load(FLAGS.efficient_distilled_model_weights, map_location="cpu")
        student_model.load_state_dict(state_dict=efficient_weights)
        print(f"Weights loaded from: {FLAGS.efficient_distilled_model_weights}")
    else:
        print(f"[ERROR] Weights {FLAGS.efficient_distilled_model_weights} not found!!")

    student_model.to(device)


    # Reinit dataset for EfficientNet
    transforms_efficientnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256, antialias=True),       
        transforms.CenterCrop(224),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
    ])
    
    Datasets_To_test = ["caltech101", "cifar10", "cifar100", "chestxray"]

    for selectedDataset in Datasets_To_test:
        if selectedDataset=="caltech101" :
            dataset = Caltech101Dataset(filter_label=None,images_path="./datasets/caltech101/101_ObjectCategories",preprocessin_fn=transforms_efficientnet,subset="train",test_split=TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL, random_seed=SEED_FOR_RANDOM_SPLIT)
            test_dataset = Caltech101Dataset(filter_label=None,images_path="./datasets/caltech101/101_ObjectCategories",preprocessin_fn=transforms_efficientnet,subset="test",test_split=TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL, random_seed=SEED_FOR_RANDOM_SPLIT)
        elif selectedDataset=="cifar10":
            dataset = CIFAR10Dataset(root="./data/", preprocessin_fn=transforms_efficientnet,subset="train")
            test_dataset = CIFAR10Dataset(root="./data/", preprocessin_fn=transforms_efficientnet,subset="test")
        elif selectedDataset=="cifar100":
            dataset = CIFAR100Dataset(root="./data/", preprocessin_fn=transforms_efficientnet,subset="train")
            test_dataset = CIFAR100Dataset(root="./data/", preprocessin_fn=transforms_efficientnet,subset="test")
        elif selectedDataset=="chestxray":
            dataset = ChestXRayDataset(img_preprocessing_fn=transforms_efficientnet,rootPath=f"{FLAGS.dataset_root}/train")
            test_dataset = ChestXRayDataset(img_preprocessing_fn=transforms_efficientnet,rootPath=f"{FLAGS.dataset_root}/test")

        
        output_dir = f"{FLAGS.log_dir}/Dataset_{selectedDataset}"
        os.makedirs(output_dir,exist_ok=True)

        with open(f'{output_dir}/commandline_args.txt', 'w') as f:
            json.dump(FLAGS.__dict__, f, indent=2)

        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
        data_loader_train = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_query = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            pin_memory=True,
            drop_last=False,
        )


        time_t0 = time.perf_counter()

        dataset.extract_features(student_model,data_loader=data_loader_train)
        test_dataset.extract_features(student_model,data_loader=data_loader_query)

        print(f"Train : {len(dataset)}  features: {len(dataset.image_features)}")
        print(f"Test : {len(test_dataset)}  features: {len(test_dataset.image_features)}")

        gallery_features = []
        query_features = []
        for i in range(len(dataset)):
            gallery_features.append(dataset.image_features[i])
        for i in range(len(test_dataset)):
            query_features.append(test_dataset.image_features[i])

        gallery_features = torch.from_numpy(np.array(gallery_features))
        query_features = torch.from_numpy(np.array(query_features))

        gallery_features = gallery_features.reshape(gallery_features.size(0), -1)
        query_features = query_features.reshape(query_features.size(0), -1)

        d = gallery_features.size(-1)    # dimension
        nb = gallery_features.size(0)    # database size
        nq = query_features.size(0)      # nb of queries
        
        index = faiss.IndexFlatL2(d)   # build the index
        print(index.is_trained)
        index.add(gallery_features)    # add vectors to the index
        print(index.ntotal)

        topK  = FLAGS.topK
        k = FLAGS.topK                          # we want to see 4 nearest neighbors
        D, I = index.search(gallery_features[:5], k) # sanity check
        print(I)
        print(D)
        D, I = index.search(query_features, k)     # actual search
        print(I[:5])                   # neighbors of the 5 first queries
        # print(I[-5:])                # neighbors of the 5 last queries

        class_scores = {"data" :{}, "metadata": {}}
        class_scores["metadata"] = {"flags": FLAGS}
        print_done = False

        
        for query_idx, search_res in enumerate(I):
            # print(search_res)
            labels = []
            test_intlabel = test_dataset.labels[query_idx]
            test_strlabel = test_dataset.class_id_to_str[test_intlabel]

            cosine_similarities = []
            cosine_similarities_labels_int = []
            cosine_similarities_labels_str = []
            cosine_similarities_labels_classid = []
            cosine_similarities_images = []

            test_intlabel = test_dataset.labels[query_idx]
            test_strlabel = test_dataset.class_id_to_str[test_intlabel]

            img_f, test_label, test_image, test_idx = test_dataset[query_idx]
            #originalImage = test_dataset.getOriginalImage(test_idx)
            originalImage = test_dataset.getImagePath(test_idx)

            if test_label["ClassName"] not in class_scores["data"]:
                class_scores["data"][test_label["ClassName"]] = {"TP": 0, 
                                                        "classIntanceRetrival": 0,
                                                        "TotalRetrival": 0,
                                                        "TotalClass": 0, 
                                                        "input_images": [],
                                                        "GroundTruths": [], 
                                                        "Predicted":[], 
                                                        "Topk": {
                                                            "labels": [], 
                                                            "scores": [],
                                                            "images": []
                                                            },
                                                        "Recall": "",
                                                        "Precision": ""
                                                        }
                
            for search_res_idx in search_res:
                intlabel = dataset.labels[search_res_idx]
                strLabel = dataset.class_id_to_str[intlabel]
                cosine_similarities_labels_str.append(strLabel)
                cosine_similarities_labels_int.append(intlabel)
                    
            cosine_similarities.append(list(D[query_idx]))
            unique, counts = np.unique(cosine_similarities_labels_str, return_counts=True)
            count = 0
            count_label = ""
            
            for u, c in zip(unique, counts):
                if u==test_strlabel:
                    count = c
                    count_label = u
            
            classIntanceRetrival = count
            TotalRetrival = topK


            if test_label["ClassName"] in cosine_similarities_labels_str:
                class_scores["data"][test_label["ClassName"]]["TP"] +=1
                class_scores["data"][test_label["ClassName"]]["classIntanceRetrival"] +=classIntanceRetrival
                class_scores["data"][test_label["ClassName"]]["Predicted"].append(test_label["ClassId"])
            else:
                class_scores["data"][test_label["ClassName"]]["Predicted"].append(test_dataset.class_str_to_id[cosine_similarities_labels_str[0]])

                
            class_scores["data"][test_label["ClassName"]]["TotalRetrival"] +=TotalRetrival
            class_scores["data"][test_label["ClassName"]]["TotalClass"] +=1

            class_scores["data"][test_label["ClassName"]]["Topk"]["labels"].append(list(cosine_similarities_labels_str))
            class_scores["data"][test_label["ClassName"]]["Topk"]["scores"].append(list(cosine_similarities))
            class_scores["data"][test_label["ClassName"]]["Topk"]["images"].append(list(cosine_similarities_images))
            
            class_scores["data"][test_label["ClassName"]]["input_images"].append(originalImage)
            class_scores["data"][test_label["ClassName"]]["GroundTruths"].append(test_label["ClassId"])

            TP  = class_scores["data"][test_label["ClassName"]]['TP']
            TotalClass = class_scores["data"][test_label["ClassName"]]['TotalClass']
            classIntanceRetrival = class_scores["data"][test_label["ClassName"]]['classIntanceRetrival']
            TotalRetrival = class_scores["data"][test_label["ClassName"]]['TotalRetrival']

            class_scores["data"][test_label["ClassName"]]["Recall"] = round(((TP*100)/TotalClass), 2)
            class_scores["data"][test_label["ClassName"]]["Precision"] = round(((classIntanceRetrival*100)/TotalRetrival), 2)


        Recall_Total = []
        Precision_Total = []
        for key, cls_data in class_scores["data"].items():
            print(f"Class : {key} Recall: [{cls_data['Recall']}] Precision: [{cls_data['Precision']}]" )
            Recall_Total.append(cls_data["Recall"])
            Precision_Total.append(cls_data["Precision"])

        Recall_Total = np.array(Recall_Total).mean()
        Precision_Total = np.array(Precision_Total).mean()
        print(f"Overall Recall :{Recall_Total} Overall Precision: {Precision_Total}")
        
        
        time_tn = time.perf_counter()
        outputPath = f"{output_dir}/{selectedDataset}_Scores.pth"
        class_scores["metadata"] = {"processing_time": f"{time_tn-time_t0:.2f}s"}
        
        torch.save(class_scores, outputPath)

        with open(f"{output_dir}/{selectedDataset}_Scores.txt", 'w') as f:
            json.dump(class_scores, f, indent=2, cls=NpEncoder)

        pthFiles = [outputPath]
        csv_file = open(f"{output_dir}/{selectedDataset}_.csv", "w")
        csv_file.write(f"srno, label, imagenet_label, Total class images,Total class image Retr, TP,Total Images Retr, Recall, Precision")
        cnt = 1
        for pth in pthFiles:
            class_metrics = torch.load(pth)
            filename = pth.split("train")[-1].split(".")[0]
            filename  = filename[1:]
            for key,val1 in class_metrics.items():
                if key=="data":
                    val1 = dict(sorted(val1.items()))
                    for classN, classData in val1.items():
                        TP  = classData['TP']
                        TotalClass = classData['TotalClass']
                        classIntanceRetrival = classData['classIntanceRetrival']
                        TotalRetrival = classData['TotalRetrival']
                        Recall = classData['Recall']
                        Precision = classData['Precision']
                        print(f"Class:{classN} TP: [{classData['TP']}] TotalClass: [{classData['TotalClass']}] classIntanceRetrival: [{classData['classIntanceRetrival']}] TotalRetrival: [{classData['TotalRetrival']}] ")
                        csv_file.write(f"\n {cnt}, {filename}, {classN}, {TotalClass},{TotalRetrival},{TP},{classIntanceRetrival},{Recall},{Precision}")
                        cnt +=1
        csv_file.write(f"\n\n,,,,,,,{Recall_Total},{Precision_Total}")                    
        csv_file.close()
        
        print(f"Completed in : {time_tn-time_t0:.2f}")