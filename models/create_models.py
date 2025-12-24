from .ViT import ViT
from .wide_resnet import WideResNet
from .resnet import ResNet,BasicBlock,BottleNeck

def get_model(args):
    if args.model == 'resnet18':
        model=ResNet(BasicBlock, [2, 2, 2, 2])
    
    elif args.model == "wide_resnet28-10":
         model=WideResNet(depth=28, widen_factor=10)

    elif args.model == "ViT-T":
         model =ViT(img_size=32, patch_size = 4, num_classes=100, dim=192,
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA)
    
    elif args.model == "ViT-S":
         model =ViT(img_size=32, patch_size = 4, num_classes=100, dim=384,
                    mlp_dim_ratio=2, depth=12, heads=6, dim_head=384//6,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA)
    
    elif args.model == "resnet50":
        model=ResNet(BottleNeck, [3, 4, 6, 3])

    
    return model