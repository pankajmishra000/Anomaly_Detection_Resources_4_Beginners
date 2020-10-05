
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from mnist_old import Mnist
from fashion_mnist import Fashion_Mnist as FMnist
from k_mnist import KMnist
from autoencoder_basic import autoencoder
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os

def train(epoch):
    model.train()
    loss_list = []
    for batch_idx, (data, _) in enumerate(m.train_loader):
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        
        recon = model(data)
        
        loss = model.loss_function(recon, data)

        loss.backward()
        loss_list.append( loss.item() )
        optimizer.step()
    #print('\nEpoch: {} -> Train loss: {:.4f}\n'.format(epoch, np.mean(loss_list)), end='')

def test_normal(epoch):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for i, (data, _) in enumerate(m.test_loader):
            data = data.to(device)
            data = data.view(data.size(0), -1)
            
            recon = model(data)
        
            loss = model.loss_function(recon, data, domean=False)

            loss_list.append(loss)
    loss_list = torch.cat(loss_list)
    #print('\nEpoch: {} -> Test normal loss: {:.4f}\n'.format(epoch, loss_list.mean()), end='')
    return loss_list

def test_anom(epoch):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for i, (data, _) in enumerate(m.test_anom_loader):
            data = data.to(device)
            data = data.view(data.size(0), -1)
            
            recon = model(data)
        
            loss = model.loss_function(recon, data, domean=False)


            loss_list.append(loss)
    loss_list = torch.cat(loss_list)
    #print('\nEpoch: {} -> Test anomalous loss: {:.4f}\n'.format(epoch, loss_list.mean()))
    return loss_list

def show_grid(img_grid):
    img_grid = img_grid.detach().cpu().numpy()
    plt.figure()
    plt.imshow(np.transpose(img_grid, (1,2,0)))
    #plt.show()

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"   
#    norm_class = 8
    batch_size = 32
    learning_rate = 0.001
    model_name = "ae_kmnist_anomaly_1"
    # field names  
    fields = ['Class', 'Epoch', 'Accuracy']                
    # name of csv file  
    filename = f'{model_name}_results.csv' # can also add path here like = './xyz/pqr/test_results.csv'
    t = open(f'{model_name}_results.txt', 'a')

    class_l = list()
    epoch_l = list()
    acc_l = list()

        
    accuracy_list = []
    auc_list = []
    for norm_class in range(10):
        
        print(f'\n ------------------- Norm class: {norm_class} -------------------\n')
        t.write(f'\n ------------------- Norm class: {norm_class} -------------------\n')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = autoencoder().to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
#        m = Mnist(batch_size=batch_size, norm_dgt=norm_class, nu=0.01) # MNIST
#        m = FMnist(batch_size=batch_size, norm_class=norm_class, nu=0.01) # FASHION MNIST
        m = KMnist(batch_size=batch_size, norm_class=norm_class, nu=0.01) # K_MNIST
        
        best_acc = 0
        for epoch in range(200):
    #        print("Epoch: ", epoch)
            train(epoch)
            loss_normal = test_normal(epoch).detach().cpu().numpy()
            loss_anom = test_anom(epoch).detach().cpu().numpy()
    
            # eval results
            roc_data = np.concatenate((loss_normal, loss_anom))
            roc_targets = np.concatenate((np.zeros(len(loss_normal)), np.ones(len(loss_anom))))
            fpr, tpr, thresholds = roc_curve( roc_targets, roc_data )
            roc_auc = auc(fpr, tpr)
            auc_list.append(roc_auc)
            
            # compute best classification
            idx = np.argmax(tpr-fpr)
            best_thresh = thresholds[idx]
            err = ((roc_data > best_thresh) != roc_targets).sum()
            acc = 1 - err/roc_data.shape[0]
            accuracy_list.append(acc)
            
            print(f'\nEpoch: {epoch} Accuracy: {acc:0.4f}\n')
            t.write(f'\nEpoch: {epoch} -> Accuracy : {acc:0.4f}\n')
            class_l.append(norm_class)
            epoch_l.append(epoch)
            acc_l.append(acc)
            
            # check and save the model with the best accuracy
#            if best_acc < acc:
                #torch.save(model.state_dict(), f'{model_name}_class_{norm_class}'+'.pt')                
#                best_acc  = acc
        
        t.write(f'\nBest accuracy:{best_acc}\n')
        torch.cuda.empty_cache()
        
    print("average accuracy: ", np.mean(accuracy_list))
    print("average AUC: ", np.mean(auc_list))
    t.write(f'\n----------------------- MNIST DATASET ---------------\n')
    t.write(f'\nAverage accuracy: {np.mean(accuracy_list)}\n')
    t.write(f'\nAverage AUC: {np.mean(auc_list)}\n')
    t.close()
    
    print(f'\n-----Writing Results to CSV file {filename}------')
    df = pd.DataFrame(list(zip(class_l,epoch_l,acc_l)), columns=fields)
#    df.set_index('Class', inplace=True)
    df.to_csv(filename)
    print(f'Done! Check file {filename}')
