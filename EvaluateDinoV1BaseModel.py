from utils.Caltech101Dataset import Caltech101Dataset
from utils.CIFAR100Dataset import CIFAR100Dataset
from utils.CIFAR10Dataset import CIFAR10Dataset
from utils.ChestXRayDataset import ChestXRayDataset
from utils import utils
from utils.DinoModel import DinoModel, dino_args
import torch
import argparse
import time 
import faiss
import numpy as np
import json
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def initDinoV1Model(model_to_load, FLAGS, checkpoint_key="teacher", use_back_bone_only=False):
    dino_args.pretrained_weights = model_to_load
    dino_args.output_dir = FLAGS.log_dir
    dino_args.checkpoint_key = checkpoint_key
    dino_args.use_cuda = torch.cuda.is_available()
    dinov1_model = DinoModel(dino_args, use_only_backbone=use_back_bone_only)
    dinov1_model.eval()
    return dinov1_model


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
                        type=int, default=4,
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

    TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL = 0.7
    SEED_FOR_RANDOM_SPLIT = 43


    # Datasets_To_test = ["caltech101", "cifar10", "cifar100", "chestxray"]
    # Datasets_To_test = ["chestxray", "cifar100", "cifar10"]
    Datasets_To_test = ["cifar10"]
    # dinov1_model = initDinoV1Model(model_to_load=FLAGS.dino_base_model_weights,FLAGS=FLAGS,checkpoint_key="teacher")
    dinov1_model = initDinoV1Model(model_to_load=FLAGS.dino_custom_model_weights,FLAGS=FLAGS,checkpoint_key="teacher", use_back_bone_only=True)

    
    for selectedDataset in Datasets_To_test:
        if selectedDataset=="caltech101" :
            dataset = Caltech101Dataset(filter_label=None,images_path="./datasets/caltech101/101_ObjectCategories",preprocessin_fn=dinov1_model.dinov1_transform,subset="train",test_split=TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL, random_seed=SEED_FOR_RANDOM_SPLIT)
            test_dataset = Caltech101Dataset(filter_label=None,images_path="./datasets/caltech101/101_ObjectCategories",preprocessin_fn=dinov1_model.dinov1_transform,subset="test",test_split=TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL, random_seed=SEED_FOR_RANDOM_SPLIT)
        elif selectedDataset=="cifar10":
            dataset = CIFAR10Dataset(root="./data/", preprocessin_fn=None,subset="train")
            test_dataset = CIFAR10Dataset(root="./data/", preprocessin_fn=None,subset="test")
        elif selectedDataset=="cifar100":
            dataset = CIFAR100Dataset(root="./data/", preprocessin_fn=None,subset="train")
            test_dataset = CIFAR100Dataset(root="./data/", preprocessin_fn=None,subset="test")
        elif selectedDataset=="chestxray":
            dataset = ChestXRayDataset(img_preprocessing_fn=dinov1_model.dinov1_transform,rootPath="./datasets/chest_xray/train")
            test_dataset = ChestXRayDataset(img_preprocessing_fn=dinov1_model.dinov1_transform,rootPath="./datasets/chest_xray/test")

        
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

        dataset.extract_features(dinov1_model,data_loader=data_loader_train)
        test_dataset.extract_features(dinov1_model,data_loader=data_loader_query)


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


