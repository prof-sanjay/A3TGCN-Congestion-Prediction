#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_result(test_result,test_label1,path):
    ##all test result visualization
    fig1 = plt.figure(figsize=(7,2))
#    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[:,0]
    a_true = test_label1[:,0]
    
    plt.plot(a_pred,'r-',label='Prediction')
    plt.plot(a_true,'b-',label='Ground Truth')

    plt.xlabel("Test Sample Index")
    plt.ylabel("Traffic Speed")
    plt.title("Prediction vs. Ground Truth")
    plt.tight_layout()
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_all.jpg')
    plt.show()
    ## oneday test result visualization
    fig1 = plt.figure(figsize=(7,1.5))
#    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[0:96,0]
    a_true = test_label1[0:96,0]
    plt.plot(a_pred,'r-',label="Prediction")
    plt.plot(a_true,'b-',label="Ground Truth")

    plt.xlabel("Time Step")
    plt.ylabel("Traffic Speed")
    plt.title("One-Day Traffic Prediction")

    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_oneday.jpg')
    plt.show()



    
def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,test_r2,path):
    ###train_rmse & test_rmse 
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse, 'r-', label="train_rmse")
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/rmse.jpg')
    plt.show()
    #### train_loss & train_rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_loss,'b-', label='train_loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss.jpg')
    plt.show()

    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse,'b-', label='train_rmse')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_rmse.jpg')
    plt.show()

    ### accuracy
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_acc, 'b-', label="test_acc")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc.jpg')
    plt.show()
    ### rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_rmse.jpg')
    plt.show()
    ### mae
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mae, 'b-', label="test_mae")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae.jpg')
    plt.show()


        # RMSE vs epoch

    fig = plt.figure(figsize=(7,4))
    plt.plot(test_rmse, color='red', linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Epoch")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(path + '/rmse_vs_epoch.jpg', dpi=500, bbox_inches='tight')
    plt.show()


    #MAE vs epoch

    fig = plt.figure(figsize=(7,4))
    plt.plot(test_mae, color='blue', linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE vs Epoch")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(path + '/mae_vs_epoch.jpg', dpi=500, bbox_inches='tight')
    plt.show()


    #R2 vs epoch

    fig = plt.figure(figsize=(7,4))
    plt.plot(test_r2, color='green', linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel(r"$R^2$")
    plt.title(r"$R^2$ vs Epoch")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(path + '/r2_vs_epoch.jpg', dpi=500, bbox_inches='tight')
    plt.show()


