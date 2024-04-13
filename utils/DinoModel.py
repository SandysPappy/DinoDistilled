import dino.vision_transformer as vits
from dino.vision_transformer import DINOHead
import dino.utils as utils
import torch
import torch.nn as nn
from torchvision import models as torchvision_models
import sys
from torchvision import transforms as pth_transforms


class dino_args:
    arch = "vit_small"
    patch_size = 8
    n_last_blocks = 4
    avgpool_patchtokens = False
    pretrained_weights = "./trainedWeights/teacher_only_eeg/checkpoint0240.pth"
    checkpoint_key = "teacher"
    num_workers = 0
    out_dim  = 65536
    use_bn_in_head = True
    use_fp16 =True
    optimizer = "adamw"
    output_dir = "./output"
    drop_path_rate = 0.1
    norm_last_layer= True
    gpu  =0
    seed = 43
    dist_url = "env://"
    local_crops_number = 8
    warmup_teacher_temp = 0.4
    epochs = 100
    teacher_temp = 0.4
    warmup_teacher_temp_epochs = 10
    use_cuda = True


class DinoModel(nn.Module):

    def __init__(self,args, use_only_backbone=False) -> None:
        super().__init__()

        self.use_only_backbone = use_only_backbone
        
        # ============ building network ... ============
        if "vit" in args.arch:
            self.backbone = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
            print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
        elif "xcit" in args.arch:
            self.backbone = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        elif args.arch in torchvision_models.__dict__.keys():
            self.backbone = torchvision_models.__dict__[args.arch](num_classes=0)
        else:
            print(f"Architecture {args.arch} non supported")
            sys.exit(1)
        if args.use_cuda:
            self.backbone.cuda()
        self.backbone.eval()


        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        # remove `head.` prefix for dino head
        state_dict = {k.replace("head.", ""): v for k, v in state_dict.items()}

        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))

        if not self.use_only_backbone:
            self.head = DINOHead(self.backbone.embed_dim, out_dim=65536,use_bn=False)
            msg = self.head.load_state_dict(state_dict, strict=False)
            print('Pretrained weights for Dino Head found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
            if args.use_cuda:
                self.head.cuda()
            self.head.eval()

        self.dinov1_transform = pth_transforms.Compose([
            # pth_transforms.ToTensor(),    
            pth_transforms.Resize((224,224)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        
    def forward(self,x,transform_inputs=False,return_back_bone_only=False):
        if transform_inputs:
            x = self.dinov1_transform(x)
        x = self.backbone(x)
        if self.use_only_backbone:
            return x
        if return_back_bone_only: 
            return x
        x = self.head(x)
        return x