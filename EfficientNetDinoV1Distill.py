import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torchvision import models
from utils.DinoModel import DinoModel, dino_args
from utils.ChestXRayDataset import ChestXRayDataset
from utils import utils
import time
import os


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
        self.efficientnet = models.efficientnet_b0(pretrained=False)
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
    os.makedirs(f"{FLAGS.log_dir}/output", exist_ok=True)

    TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL = 0.7
    SEED_FOR_RANDOM_SPLIT = 43

    # Load DINOv1 model (you need to replace this with your own DINOv2 model)
    dinov1_model = initDinoV1Model(model_to_load=FLAGS.dino_custom_model_weights,FLAGS=FLAGS,checkpoint_key="teacher", use_back_bone_only=True)

    dataset = ChestXRayDataset(img_preprocessing_fn=dinov1_model.dinov1_transform,rootPath=f"{FLAGS.dataset_root}/train")
    test_dataset = ChestXRayDataset(img_preprocessing_fn=dinov1_model.dinov1_transform,rootPath=f"{FLAGS.dataset_root}/test")

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


    """
    Extract DINO features
    """
    time_t0 = time.perf_counter()

    dataset.extract_features(dinov1_model,data_loader=data_loader_train)
    test_dataset.extract_features(dinov1_model,data_loader=data_loader_query)

    time_t1 = time.perf_counter()

    print(f"Feature extraction done in : {time_t1-time_t0}s")

    #Save Features In a list 
    train_image_features = dataset.image_features
    test_image_features = test_dataset.image_features


    # Reinit dataset for EfficientNet
    transforms_efficientnet = transforms.Compose([
        transforms.Resize((224,224), antialias=True), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ChestXRayDataset(img_preprocessing_fn=transforms_efficientnet,rootPath="./datasets/chest_xray/train")
    test_dataset = ChestXRayDataset(img_preprocessing_fn=transforms_efficientnet,rootPath="./datasets/chest_xray/test")

    # Reassign DINO features to dataset
    dataset.image_features = train_image_features 
    dataset.isDataTransformed = True # else image features will be empty
    test_dataset.image_features = test_image_features
    test_dataset.isDataTransformed = True # else image features will be empty

    del train_image_features, test_image_features
    print(f"Train DINO image features : {len(dataset.image_features)}  Test features: {len(test_dataset.image_features)}")

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    class HyperParams:
        learning_rate=0.001
        T=2
        soft_target_loss_weight=0.25
        ce_loss_weight=0.75
    

    # Initialize the student model
    student_model = StudentModel(num_features=384)  # dinov1 small

    # Define loss function (you can use other loss functions as well)
    ce_loss = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=HyperParams.learning_rate)

    # Training loop

    best_val_loss = None

    TotalBatches = len(data_loader_train)/FLAGS.batch_size

    for epoch in range(FLAGS.num_epochs):
        running_loss= 0.0
        student_model.train()
        for batch_idx, (img_f, label, image, idx) in enumerate(data_loader_train):
            optimizer.zero_grad()

            teacher_logits = img_f # pre exctracted features from dinov1

            # Forward pass with the student model
            student_logits = student_model(image)


            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / HyperParams.T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / HyperParams.T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (HyperParams.T**2)


            # Calculate the true label loss
            label_loss = ce_loss(student_logits, label)

            # Weighted sum of the two losses
            loss = HyperParams.soft_target_loss_weight * soft_targets_loss + HyperParams.ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if TotalBatches % 10 == 0:
                print(f"EPOCH[{epoch}] Batch {batch_idx}, Loss: {loss.item()}")
        
        student_model.eval()
       
        test_loss = 0.0
        for batch_idx, (img_f, label, image, idx) in enumerate(data_loader_test):
            student_logits = student_model(image)
            label_loss = ce_loss(student_logits, label)
            test_loss += loss.item()
        
        current_test_loss = test_loss/len(data_loader_test)
        current_train_loss = running_loss/len(data_loader_train)
        if best_val_loss is None:
            best_val_loss = current_test_loss
        else:
            if current_test_loss<best_val_loss:
                torch.save(student_model.state_dict(), 'output/student_model_efficientnetb0_best_test_loss.pth')
        
        print(f"EPOCH[{epoch}] Train loss: {current_train_loss} Test loss: {current_test_loss}")

    # Save the trained student model
    torch.save(student_model.state_dict(), 'output/student_model_efficientnetb0_final.pth')
    print("Student model trained and saved.")
