import torch
import random
import argparse
import numpy as np

from utils import *
from datasets.cardiac_dig import CardiacDigProvider
from models.CardiacDig_model import RobustNet


# Use GPU if available.
device = torch.device("cuda")
print(device, " will be used.\n")


def train(model, train_loader, criterion, optimizer, scheduler):
    
    avg_loss, avg_mtloss = [], []
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        img, label, d1annot, d2annot, d3annot, pix = data
        #print(img.shape, label.shape, d1annot.shape, pix.shape)
        imgin = img.to(torch.float32).to(device)
        outd1, outd2, outd3, mtloss = model(imgin)
        #print(outd1.shape)
        label = label.to(device)
        d1annot, d2annot, d3annot = d1annot.to(device), d2annot.to(device), d3annot.to(device)
        loss, mtloss = criterion(outd1, outd2, outd3, label, mtloss, d1annot, d2annot, d3annot)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item() - mtloss.item()*0.001)
        avg_mtloss.append(mtloss.item()*0.001)
    scheduler.step()
    avg_loss = np.mean(np.array(avg_loss))    
    return avg_loss


def test(model, test_loader, criterion_test):
    
    all_loss, all_out, all_label = [], [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, label, d1annot, d2annot, d3annot, pix = data
            imgin = img.to(device).float()
            outd1, outd2, outd3, _ = model(imgin)
            label = label.to(device)

            d1annot, d2annot, d3annot = d1annot.to(device), d2annot.to(device), d3annot.to(device)
            loss, out, label = criterion_test(outd1, outd2, outd3, label, d1annot, d2annot, d3annot)
            
            pix = torch.reshape(pix, (-1, 1)).numpy()
            loss[:, :2] = loss[:, :2] * pix[0, 0] * pix[0, 0] * 80 * 80.0
            out[:, :2] = out[:, :2] * pix[0, 0] * pix[0, 0] * 80 * 80.0
            label[:, :2] = label[:, :2] * pix[0, 0] * pix[0, 0] * 80 * 80.0
            
            loss[:, 2:] = loss[:, 2:] * pix[0, 0] * 80.0
            out[:, 2:] = out[:, 2:] * pix[0, 0] * 80.0
            label[:, 2:] = label[:, 2:] * pix[0, 0] * 80.0
            
            all_loss.append(loss)
            all_out.append(out)
            all_label.append(label)
    
    all_loss = torch.cat(all_loss, dim=0).cpu().numpy()
    all_out = torch.cat(all_out, dim=0).cpu().numpy()
    all_label = torch.cat(all_label, dim=0).cpu().numpy()
    
    return all_loss, all_out, all_label


def valid(model, valid_loader, criterion_test):
    
    all_loss, all_out, all_label = [], [], []
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            img, label, d1annot, d2annot, d3annot, pix = data
            imgin = img.to(device).float()
            outd1, outd2, outd3, _ = model(imgin)
            label = label.to(device)

            d1annot, d2annot, d3annot = d1annot.to(device), d2annot.to(device), d3annot.to(device)
            loss, out, label = criterion_test(outd1, outd2, outd3, label, d1annot, d2annot, d3annot)
            
            pix = torch.reshape(pix, (-1, 1)).numpy()
            loss[:, :2] = loss[:, :2] * pix[0, 0] * pix[0, 0] * 80 * 80.0
            out[:, :2] = out[:, :2] * pix[0, 0] * pix[0, 0] * 80 * 80.0
            label[:, :2] = label[:, :2] * pix[0, 0] * pix[0, 0] * 80 * 80.0
            
            loss[:, 2:] = loss[:, 2:] * pix[0, 0] * 80.0
            out[:, 2:] = out[:, 2:] * pix[0, 0] * 80.0
            label[:, 2:] = label[:, 2:] * pix[0, 0] * 80.0
            
            all_loss.append(loss)
            all_out.append(out)
            all_label.append(label)
    
    all_loss = torch.cat(all_loss, dim=0).cpu().numpy()
    all_out = torch.cat(all_out, dim=0).cpu().numpy()
    all_label = torch.cat(all_label, dim=0).cpu().numpy()
    
    return all_loss, all_out, all_label


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cardiac_dig', help="dataset")
    parser.add_argument("--dataset_dir", default="./data/ACDC/")
    parser.add_argument("--save_path", default="./pretrained/")
    
    parser.add_argument("--cross_valid", default=2)
    
    parser.add_argument("--batch_size", default=4, help="batch size")
    parser.add_argument("--lr", default=1e-3, help="learning rate")
    parser.add_argument("--lr_gamma", default=0.2) 
    parser.add_argument("--num_epochs", default=400)
    parser.add_argument("--img_size", default=80)
    parser.add_argument("--save_interval", default=10)
    parser.add_argument("--multistep", default=[150, 250, 350])
    
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--checkpoint", default=None)      
    args = parser.parse_args()
    

    model = RobustNet(args).to(device)

    pretrained_path = 'checkpoint/model_best_test_{}.pt'.format(args.dataset)
    if os.path.exists(pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        model.load_state_dict(pretrained_dict['model'])
        model.task_cov_var.data = torch.tensor(pretrained_dict['task']).to(device)
        model.class_cov_var.data = torch.tensor(pretrained_dict['class']).to(device)
        model.feature_cov_var.data = torch.tensor(pretrained_dict['feature']).to(device)
    else:
        model.apply(weights_init)

    criterion = TensorNormalLoss()
    criterion_test = L1TestLoss()

    # Optimizer
    temporal_params = list(map(id, model.temporal.parameters()))
    base_params = filter(lambda p: id(p) not in temporal_params, model.parameters())
    optimizer = torch.optim.Adam([
                 {'params': model.temporal.parameters(), 'lr': args.lr},
                 {'params': base_params, 'lr': args.lr}], weight_decay=0.0001)
    # optimizer = optim.Adam([{'params': model.parameters()}], lr=params['lr'], betas=(params['beta1'], params['beta2']))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                        milestones=args.multistep, gamma=args.lr_gamma)

    print("-"*25)
    print("Starting Training Loop...\n")
    print("-"*25)

    best_mae = 1000000.
    for epoch in range(args.num_epochs):
    #for path in paths1:
        if(args.dataset == 'cardiac_dig'):
            dataset = CardiacDigProvider(args.batch_size, args.cross_valid)
        train_loader, test_loader, valid_loader = dataset.train, dataset.test, dataset.valid
        
        model.train()
        train_loss = train(model, train_loader, criterion, optimizer, scheduler)
        
        if epoch % 3==2:
            model.update_cov()
        
        model.eval()
        
        all_loss, all_out, all_label = valid(model, valid_loader, criterion_test)
        avg_loss = all_loss.mean(axis=0)
        if epoch % 10 == 0:
            print('Epoch: ', epoch)
            print('Valid Loss: ', avg_loss[:2].mean(), avg_loss[2:5].mean(), avg_loss[5:].mean())
        
        os.makedirs(args.save_path, exist_ok=True)
        if epoch > 1 and epoch % args.save_interval == 0:
            file_path = os.path.join('pretrained/cv{}-epoch{}.pkl'.format(args.cross_valid, epoch))
            save(model, file_path)
        
        if avg_loss[:2].mean() < best_mae:
            best_mae = avg_loss[:2].mean()
            file_path = os.path.join('pretrained/cv{}-test_best.pkl'.format(args.cross_valid))
            save(model, file_path)
    
    best_path = os.path.join(args.save_path, str(args.cross_valid) + '-test_best.pkl')
    if os.path.exists(best_path):
        load_checkpoint(model, best_path, device)
        
    all_loss, all_out, all_label = test(model, test_loader, criterion_test)
    avg_loss = all_loss.mean(axis=0)
    print('Test: ')
    print('Detailed: ', avg_loss)
    print('Test Loss: ', avg_loss[:2].mean(), avg_loss[2:5].mean(), avg_loss[5:].mean())


if __name__=='__main__':
    
    # Set random seed for reproducibility.
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Random Seed: ", seed)
    
    main()
    

