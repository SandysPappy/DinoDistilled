from utils.Caltech101Dataset import Caltech101Dataset
from utils.CIFAR100Dataset import CIFAR100Dataset
from utils.CIFAR10Dataset import CIFAR10Dataset
from utils.utils import init_distributed_mode
from utils.DinoModel import DinoModel, dino_args
import torch
import argparse
import time 
import faiss
import numpy as np
import json
import os
import torchvision
import torchvision.transforms as T
import torch.distributed as dist
from PIL import Image
from TinyViT.data import build_transform
from TinyViT.config import get_config
from TinyViT.models import build_model
from TinyViT.logger import create_logger
from TinyViT.utils import load_checkpoint, load_pretrained, save_checkpoint

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def initDinoV1Model(model_to_load, FLAGS, checkpoint_key="teacher"):
    dino_args.pretrained_weights = model_to_load
    dino_args.output_dir = FLAGS.log_dir
    dino_args.checkpoint_key = checkpoint_key
    dino_args.use_cuda = torch.cuda.is_available()
    dinov1_model = DinoModel(dino_args)
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
                        type=int, default=1,
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
    parser.add_argument('--search_gallery',
                        type=str,
                        default="train",
                        help='dataset in which images will be searched')
    parser.add_argument('--topK',
                        type=int,
                        default=5,
                        help='Top-k paramter, defaults to 5')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='Number gpu.')
    parser.add_argument("--dist_url", 
                        default="env://", 
                        type=str, 
                        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", 
                        default=0, 
                        type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--pretrained',
                        type=str,
                        default="./TinyViT/models/pretrained/tiny_vit_21m_22k_distill.pth",
                        help='TinyViT pretrained weight')
    parser.add_argument('--output',
                        type=str,
                        default="./TinyViT_log",
                        help='TinyViT log output')
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    init_distributed_mode(FLAGS)

    TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL = 0.7
    SEED_FOR_RANDOM_SPLIT = 43

    Datasets_To_test = ["caltech101"]
    #Datasets_To_test = ["caltech101", "cifar10", "cifar100"]
    dinov1_model = initDinoV1Model(model_to_load=FLAGS.dino_base_model_weights,FLAGS=FLAGS,checkpoint_key="teacher")

    for selectedDataset in Datasets_To_test:
        if selectedDataset=="caltech101" :
            dataset = Caltech101Dataset(filter_label=None,images_path="./data/caltech-101/101_ObjectCategories",preprocessin_fn=dinov1_model.dinov1_transform,subset="train",test_split=TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL, random_seed=SEED_FOR_RANDOM_SPLIT)
            test_dataset = Caltech101Dataset(filter_label=None,images_path="./data/caltech-101/101_ObjectCategories",preprocessin_fn=dinov1_model.dinov1_transform,subset="test",test_split=TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL, random_seed=SEED_FOR_RANDOM_SPLIT)
        elif selectedDataset=="cifar10":
            dataset = CIFAR10Dataset(root="./data/", preprocessin_fn=None,subset="train")
            test_dataset = CIFAR10Dataset(root="./data/", preprocessin_fn=None,subset="test")
        elif selectedDataset=="cifar100":
            dataset = CIFAR100Dataset(root="./data/", preprocessin_fn=None,subset="train")
            test_dataset = CIFAR100Dataset(root="./data/", preprocessin_fn=None,subset="test")

        
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


        time_t0 = time.perf_counter()
        '''
        dataset.extract_features(dinov1_model,data_loader=data_loader_train)
        #print(dataset.image_features)
       
        gallery_features = []
        for i in range(len(dataset)):
            gallery_features.append(dataset.image_features[i])

        gallery_features = torch.from_numpy(np.array(gallery_features))

        gallery_features = gallery_features.reshape(gallery_features.size(0), -1)
        '''
        config = get_config(FLAGS)
        TinyViT_model = build_model(config)

        logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

        load_pretrained(config, TinyViT_model, logger)
        TinyViT_model.eval()

        image = dataset[0][2]
        toPIL = T.ToPILImage() 
        image = toPIL(image)
        transform = build_transform(is_train=False, config=config)
        batch = transform(image)[None]

        with torch.no_grad():
            logits = TinyViT_model(batch)

        probs = torch.softmax(logits, -1)
        scores, inds = probs.topk(5, largest=True, sorted=True)
        print('=' * 30)
        for score, ind in zip(scores[0].numpy(), inds[0].numpy()):
            print(f'{ind}: {score:.2f}')



        
