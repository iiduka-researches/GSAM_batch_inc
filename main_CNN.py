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
from utils.loss_function import crossentropy
from utils.log import Log
from gsam.gsam import GSAM
from gsam.scheduler import CosineScheduler,ProportionScheduler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--model', default='resnet18',choices=['resnet18', "wide_resnet28-10","resnet50"], type=str)
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--seed', type=int, default=0, help='seed') 
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--batch_list", default=[16,32,64,128,256], type=list, help="This is the list used when batch increasing.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument('--method', default='fix',choices=['fix','cosine','batch'], type=str,help="Specify whether to fix lr and batch, increase batch, or use cosine.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--gsam_alpha", default=0.02, type=int, help="Rho parameter for SAM.")

    return parser.parse_args()
    


def train_and_val(args,train_loader,test_loader,model,scheduler,rho_scheduler,optimizer,log,epochs,batch_size):
    for epoch in range(epochs):
        model.train()
        log.train(len_dataset=len(train_loader))

        for batch in train_loader:
            inputs, targets = (b.cuda(args.gpu) for b in batch)

            def loss_fn(predictions, targets):
                return crossentropy(predictions, targets).mean()
            
            optimizer.set_closure(loss_fn, inputs, targets)
            predictions, loss = optimizer.step()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu().repeat(batch_size), correct.cpu(), scheduler.lr())
                scheduler.step()
                optimizer.update_rho_t()
        
        model.eval()
        log.eval(len_dataset=len(test_loader))

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = (b.cuda(args.gpu) for b in batch)

                predictions = model(inputs)
                loss = crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
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

    base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    test_loader = torch.utils.data.DataLoader(testset,batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=2,
        )

    if args.method=="fix":
        train_loader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=2,
        )

        scheduler = CosineScheduler(T_max=args.epochs*len(train_loader), max_value=args.learning_rate, min_value=args.learning_rate, optimizer=base_optimizer,warmup_steps=0,init_value=args.learning_rate)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.learning_rate, min_lr=args.learning_rate,max_value=args.rho, min_value=args.rho)
        optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.gsam_alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)
    elif args.method=="cosine":
        train_loader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=2,
        )

        scheduler = CosineScheduler(T_max=args.epochs*len(train_loader), max_value=args.learning_rate, min_value=args.learning_rate*0.01, optimizer=base_optimizer,warmup_steps=0,init_value=args.learning_rate)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.learning_rate, min_lr=args.learning_rate*0.01,max_value=args.rho, min_value=args.rho)
        optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.gsam_alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)
    
    if args.method=="batch":
        batch_epoch=int(args.epochs/len(args.batch_list))
        for b in args.batch_list:
            train_loader = torch.utils.data.DataLoader(trainset,batch_size=b,shuffle=True,drop_last=True,num_workers=2,
            )
            scheduler=CosineScheduler(T_max=batch_epoch*len(train_loader), max_value=args.learning_rate, min_value=args.learning_rate, optimizer=base_optimizer,warmup_steps=0,init_value=args.learning_rate)
            rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.learning_rate, min_lr=args.learning_rate,max_value=args.rho, min_value=args.rho)
            optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.gsam_alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)
            model,log=train_and_val(args,train_loader,test_loader,model,scheduler,rho_scheduler,optimizer,log,epochs=batch_epoch,batch_size=b)
    
    else:
        model,log=train_and_val(args,train_loader,test_loader,model,scheduler,rho_scheduler,optimizer,log,epochs=args.epochs,batch_size=args.batch_size)
    
    log.flush()

    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pth'))

        

