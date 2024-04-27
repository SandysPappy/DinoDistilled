import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from myutils.Caltech101Dataset import Caltech101Dataset
from myutils import utils
from myutils.DinoModel import DinoModel, dino_args
import time
import os
from TinyViT.models.tiny_vit import _create_tiny_vit 

from myutils.ChestXRayDataset import ChestXRayDataset
# Path to the dataset
dataset_path = "datasets/caltech101/101_ObjectCategories"

def initDinoV2Model(model="dino_vits8"):
    dinov2_model = torch.hub.load("facebookresearch/dino", model)
    return dinov2_model

from TinyViT.models.tiny_vit import tiny_vit_5m_224  # Directly import the specific model function

class StudentTinyViT(nn.Module):
    def __init__(self, num_classes=384, pretrained=False):
        super(StudentTinyViT, self).__init__()
        # Directly create an instance of tiny_vit_5m_224
        self.tinyvit = tiny_vit_5m_224(pretrained=False)
        # Access the appropriate attribute for the number of in_features
        num_features = self.tinyvit.head.in_features
        # Replace the classifier head with a new one suited to your number of classes
        self.tinyvit.head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.tinyvit(x)

def initDinoV1Model(model_to_load, FLAGS, checkpoint_key="teacher"):
    dino_args.pretrained_weights = model_to_load
    dino_args.output_dir = FLAGS.log_dir
    dino_args.checkpoint_key = checkpoint_key
    dino_args.use_cuda = torch.cuda.is_available()
    dinov1_model = DinoModel(dino_args, use_only_backbone=True)
    dinov1_model.eval()
    return dinov1_model


def get_dino_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Efficient DINO')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir', type=str, default='DINO', help='Directory to put logging.')
    parser.add_argument('--mode', type=str, default="train", help='Type of mode: train or test')
    parser.add_argument('--dino_custom_model_weights', type=str, default="./weights/dinoxray/checkpoint.pth", help='DINO custom model weights')
    parser.add_argument('--search_gallery', type=str, default="train", help='Dataset in which images will be searched')
    parser.add_argument('--topK', type=int, default=5, help='Top-k parameter, defaults to 5')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="URL used to set up distributed training.")
    parser.add_argument("--local_rank", default=0, type=int, help="Local rank, do not set this argument manually if using distributed training.")
    parser.add_argument('--dino_base_model_weights',
                        type=str,
                        default="./models/pretrains/dino_deitsmall8_pretrain_full_checkpoint.pth",
                        help='dino based model weights') 
    parser.add_argument('--dataset_root',
                        type=str,
                        default="./datasets/chest_xray",
                        help='dataset directory root')

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    utils.init_distributed_mode(args=FLAGS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    dinov1_model = initDinoV1Model(model_to_load=FLAGS.dino_base_model_weights,FLAGS=FLAGS,checkpoint_key="teacher")
    dinov1_model = dinov1_model.to(device)
    dinov1_model.eval()
    
    if os.path.exists(FLAGS.dino_custom_model_weights):
        state_dict = torch.load(FLAGS.dino_custom_model_weights, map_location=device)
        dinov1_model.load_state_dict(state_dict,strict=False)
    '''
    dataset = Caltech101Dataset(filter_label=None, preprocessin_fn=dinov1_model.dinov1_transform, subset="train", images_path=dataset_path, random_seed=43)
    test_dataset = Caltech101Dataset(filter_label=None, preprocessin_fn=dinov1_model.dinov1_transform, subset="test", images_path=dataset_path, random_seed=43)
    '''

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

    # Feature extraction
    time_t0 = time.perf_counter()
    dataset.extract_features(dinov1_model, data_loader=data_loader_train)
    test_dataset.extract_features(dinov1_model, data_loader=data_loader_query)
    time_t1 = time.perf_counter()
    print(f"Feature extraction done in : {time_t1 - time_t0}s")

    # Save Features In a list 
    train_image_features = dataset.image_features
    test_image_features = test_dataset.image_features

    print("len:", len(train_image_features))
    print("image feature size:", train_image_features[0].shape)
    print(train_image_features[0])

    transforms_tinyvit = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    '''
    dataset = Caltech101Dataset(filter_label=None, preprocessin_fn=transforms_tinyvit,subset="train", images_path=dataset_path, random_seed=43)
    test_dataset = Caltech101Dataset(filter_label=None, preprocessin_fn=transforms_tinyvit,subset="test", images_path=dataset_path, random_seed=43)
    '''
    
    dataset = ChestXRayDataset(img_preprocessing_fn=transforms_tinyvit,rootPath=f"{FLAGS.dataset_root}/train", inference_mode=False)
    test_dataset = ChestXRayDataset(img_preprocessing_fn=transforms_tinyvit,rootPath=f"{FLAGS.dataset_root}/test", inference_mode=False)

    # Reassigning DINO features to dataset
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

    student_model = StudentTinyViT(num_classes=384, pretrained=True).to(device)  # Ensure the model is created with the pretrained flag
    student_model =student_model.to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=HyperParams.learning_rate)

    best_val_loss = None

    # Training loop
    for epoch in range(FLAGS.num_epochs):
        running_loss = 0.0
        student_model.train()
        for batch_idx, (img_f, labels, images, idx) in enumerate(data_loader_train):
            images = images.to(device)  # Move images to the correct device
            #labels = labels["ClassId"]
            labels = labels.to(device)  # Move labels to the correct device
            #labels = labels.reshape(4,1)

            optimizer.zero_grad()
            # Assuming teacher_logits are included in your dataset and handled correctly
            # If teacher_logits are not part of your dataset, you will need to adjust this
            teacher_logits = img_f

            student_logits = student_model(images)
                
            #print("teacher logit:", teacher_logits.size())
            #print("student logit:", student_logits.size())
            #print("label size:", labels.size())

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / HyperParams.T, dim=-1).to(device)
            soft_prob = nn.functional.log_softmax(student_logits / HyperParams.T, dim=-1).to(device) 

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (HyperParams.T**2)

            label_loss = ce_loss(student_logits, labels)
            loss = HyperParams.soft_target_loss_weight * soft_targets_loss + HyperParams.ce_loss_weight * label_loss
             # Update this if you use soft targets
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"EPOCH[{epoch}] Batch {batch_idx}, Loss: {loss.item()}")
            
        student_model.eval()

        test_loss = 0.0
        for batch_idx, (img_f, labels, images, idx) in enumerate(data_loader_test):
            images = images.to(device)
            #labels = labels["ClassId"]
            labels = labels.to(device)
            student_logits = student_model(images)
            label_loss = ce_loss(student_logits, labels)
            test_loss += label_loss.item()

        current_test_loss = test_loss/len(data_loader_test)
        current_train_loss = running_loss/len(data_loader_train)
        if best_val_loss is None:
            best_val_loss = current_test_loss
        else:
            if current_test_loss<best_val_loss:
                torch.save(student_model.state_dict(), f"{FLAGS.log_dir}/output/v1student_model_tinyvit_best_test_loss.pth")

        print(f"EPOCH[{epoch}] Train loss: {running_loss / len(data_loader_train)} Test loss: {test_loss / len(data_loader_test)}")

    os.makedirs("output", exist_ok=True)
    torch.save(student_model.state_dict(), f"{FLAGS.log_dir}/output/v1student_model_tinyvit.pth")
    print("Student model trained and saved.")




