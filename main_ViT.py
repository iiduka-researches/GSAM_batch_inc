import argparse
import os
import numpy as np
import time
import torch
import torch.nn.functional as F
import random
from models.create_models import get_model
from utils.initialize import initialize
from utils.dataset import get_dataset
from utils.sampler import RASampler
from utils.loss_function import LabelSmoothingCrossEntropy
from utils.log import Log
from utils.mix import mixup_criterion,mixup_data,cutmix_data
from gsam.gsam import GSAM
from gsam.scheduler import CosineScheduler,ProportionScheduler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--model', default="ViT-T",choices=["ViT-T","ViT-S"], type=str)
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--seed', type=int, default=0, help='seed') 
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--batch_list", default=[64,128,256,512], type=list, help="This is the list used when batch increasing.")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs.")
    parser.add_argument('--method', default='fix',choices=['fix','cosine','batch'], type=str,help="Specify whether to fix lr and batch, increase batch, or use cosine.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--rho", default=0.6, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--gsam_alpha", default=0.1, type=int, help="Rho parameter for SAM.")
    
    parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--ls', action='store_false', help='label smoothing')
    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')
    
    #
    parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--beta', default=1.0, type=float,help='hyperparameter beta (default: 1)')
    parser.add_argument('--alpha', default=1.0, type=float,help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mix_prob', default=0.5, type=float,help='mixup probability')
    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    parser.add_argument('--is_LSA', action='store_true', help='Locality Self-Attention')
    parser.add_argument('--is_SPT', action='store_true', help='Shifted Patch Tokenization')

    return parser.parse_args()
    


def train_and_val(args,train_loader,test_loader,model,scheduler,rho_scheduler,optimizer,log,epochs,batch_size):
    criterion = LabelSmoothingCrossEntropy().cuda(args.gpu)
    val_criterion = LabelSmoothingCrossEntropy(train=False).cuda(args.gpu)
    for epoch in range(epochs):
        model.train()
        log.train(len_dataset=len(train_loader))

        for batch in train_loader:
            inputs, targets = (b.cuda(args.gpu) for b in batch)

            def loss_fn(predictions, targets, **kwargs):
                # デフォルト値の設定 (必要に応じて)
                criterion = kwargs.get('criterion', LabelSmoothingCrossEntropy())  
                y_a = kwargs.get('y_a', None)
                y_b = kwargs.get('y_b', None)
                lam = kwargs.get('lam', 0.4)

                
                if y_a is not None and y_b is not None and lam is not None:
                    # MixUp
                    loss = mixup_criterion(criterion, predictions, y_a, y_b, lam)
                else:
                    loss = criterion(predictions, targets)

                return loss
            
            r = np.random.rand(1)
            if r < args.mix_prob:
                switching_prob = np.random.rand(1)

                # Cutmix
                if switching_prob < 0.5:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(inputs, targets, args.alpha)
                    inputs[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced

                    optimizer.set_closure(loss_fn, inputs,targets, **{'criterion': criterion, 'y_a': y_a, 'y_b': y_b, 'lam': lam})

                else:
                    inputs, y_a, y_b, lam = mixup_data(inputs, targets, args.beta)

                    optimizer.set_closure(loss_fn, inputs, targets,**{'criterion': criterion, 'y_a': y_a, 'y_b': y_b, 'lam': lam})

            else:
                    optimizer.set_closure(loss_fn, inputs, targets)
            
            predictions, loss = optimizer.step()

            with torch.no_grad():
                maxk = 1
                _,pred = predictions.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                log(model, loss.cpu().repeat(batch_size), correct.cpu(), scheduler.lr())
                scheduler.step()
                optimizer.update_rho_t()
        
        model.eval()
        log.eval(len_dataset=len(test_loader))

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = (b.cuda(args.gpu) for b in batch)

                predictions = model(inputs)
                loss = val_criterion(predictions, targets)
                _,pred = predictions.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                log(model, loss.cpu(), correct.cpu())
    

    return model,log

if __name__ == '__main__':
    args = get_args()
    global save_path
    
    initialize(args)
    
    model_name = args.model
    model_name += f"{args.dataset}-Seed{args.seed}"
    save_path = os.path.join(os.getcwd(), 'save', model_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    model=get_model(args)
    model.cuda(args.gpu)
    trainset,testset=get_dataset(args)
    
    log = Log(log_each=10)

    

    base_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    test_loader = torch.utils.data.DataLoader(testset,batch_size=args.batch_size,shuffle=False,pin_memory=True,drop_last=False,num_workers=2,
        )

    if args.method=="fix":
        train_loader = torch.utils.data.DataLoader(
            trainset,  num_workers=0, pin_memory=False,
            batch_sampler=RASampler(len(trainset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
        
        warmup_steps=int(args.warmup * len(train_loader))

        scheduler = CosineScheduler(T_max=args.epochs*len(train_loader), max_value=args.learning_rate, min_value=args.learning_rate, optimizer=base_optimizer,warmup_steps=warmup_steps,init_value=args.learning_rate*0.01)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.learning_rate, min_lr=args.learning_rate,max_value=args.rho, min_value=args.rho)
        optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.gsam_alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)
    elif args.method=="cosine":
        train_loader = torch.utils.data.DataLoader(
            trainset,  num_workers=0, pin_memory=False,
            batch_sampler=RASampler(len(trainset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
        warmup_steps=int(args.warmup * len(train_loader))

        scheduler = CosineScheduler(T_max=args.epochs*len(train_loader), max_value=args.learning_rate, min_value=args.learning_rate*0.01, optimizer=base_optimizer,warmup_steps=warmup_steps,init_value=args.learning_rate*0.01)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.learning_rate, min_lr=args.learning_rate*0.01,max_value=args.rho, min_value=args.rho)
        optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.gsam_alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)
    
    if args.method=="batch":
        batch_epoch=int(args.epochs/len(args.batch_list))
        for index, b in enumerate(args.batch_list):
            train_loader = torch.utils.data.DataLoader(
                trainset,  num_workers=0, pin_memory=False,
                batch_sampler=RASampler(len(trainset), b, 1, args.ra, shuffle=True, drop_last=True))
            
            if index==0:
                warmup_steps=int(args.warmup * len(train_loader))
                init=args.learning_rate*0.01
            else:
                warmup_steps=0
                init=args.learning_rate

            scheduler=CosineScheduler(T_max=batch_epoch*len(train_loader), max_value=args.learning_rate, min_value=args.learning_rate, optimizer=base_optimizer,warmup_steps=warmup_steps,init_value=init)
            rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.learning_rate, min_lr=args.learning_rate,max_value=args.rho, min_value=args.rho)
            optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.gsam_alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)
            model,log=train_and_val(args,train_loader,test_loader,model,scheduler,rho_scheduler,optimizer,log,epochs=batch_epoch,batch_size=b)
    
    else:
        model,log=train_and_val(args,train_loader,test_loader,model,scheduler,rho_scheduler,optimizer,log,epochs=args.epochs,batch_size=args.batch_size)
    
    log.flush()

    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pth'))