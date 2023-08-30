# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:22:31 2022

@author: thura
"""

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.animation as animation
import numpy as np
from metrics import clustering_metrics
from sklearn.cluster import KMeans

def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations=100, custom_label=''):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([
        0, np.max(minibatch_losses[num_losses:])*1.5
        ])

    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()
    
    
def plot_accuracy(train_acc, valid_acc):

    num_epochs = len(train_acc)

    plt.plot(np.arange(1, num_epochs+1), 
             train_acc, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    
    
def plot_generated_images1 (data_loader, model, device, 
                          unnormalizer=None,
                          figsize=(20, 2.5), n_images=15, modeltype='autoencoder'):

    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    
    for batch_idx, (features, _) in enumerate(data_loader):
        
        features = features.to(device)

        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]
        
        with torch.no_grad():
            if modeltype == 'autoencoder':
                decoded_images = model(features)[:n_images]
            elif modeltype == 'VAE':
                encoded, z_mean, z_log_var, decoded_images = model(features)[:n_images]
            else:
                raise ValueError('`modeltype` not supported')

        orig_images = features[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')
                
def plot_generated_images(data_loader, model, device,figsize=(20, 2.5), n_images=15):                    
                          
    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    J1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25]
    J2=[2, 21, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 23, 8, 25, 12]
    train_features, train_labels = next(iter(data_loader))
    o = [0,1,2]  

    point = train_features.permute(1,2,0)
    print("shape of point", point.shape)
    point = point.reshape(30,75)
    print("shape of point after permute and reshape", point.shape)
    ok = point.reshape(30,25,3)    
    prmd = ok

    x = prmd[:,:,0]
    y = prmd[:,:,1]
    z = prmd[:,:,2]
    def update_points_lines(num):
            # Remove the previous frame
            ax.cla()
            # set x,y,z limits
            ax.set_xlim(np.min(z), np.max(z))
            ax.set_ylim(np.min(x), np.max(x))
            ax.set_zlim(np.min(y), np.max(y))

            # load data
            df = prmd[num]

            # draw points
            # ax.scatter3D(df[:, o[0]], df[:, o[1]], df[:, o[2]], cmap='Greens', linewidth=0.2)

            # draw line
            line = [[] for _ in range(len(J1))]
            for j in range(len(J1)):
                point1 = df[J1[j]-1]
                point2 = df[J2[j]-1]
                line[j] = ax.plot3D([point1[o[0]],point2[o[0]]],[point1[o[1]],point2[o[1]]],[point1[o[2]],point2[o[2]]], 'gray')
            return line
    n =0
    m = 5
    for i  in range(n,m):
    
    # x = x.permute(1,2,0)
    # x = x.reshape(30,75)
    # x = x.reshape(30,25,3)
        df = ok[i,:,:]
        fig = plt.figure(tight_layout=True)
        ax = plt.axes(projection='3d')
        point_ani = ax.scatter3D(df[:, o[0]], df[:, o[1]], df[:, o[2]], cmap='tab10', linewidth =0.2)
        line = [[] for _ in range (len(J1))]
        for j in range(len(J1)):
            point1 = df[J1[j]-1]
            point2 = df[J2[j]-1]
            line[j] = ax.plot3D([point1[o[0]],point2[o[0]]],[point1[o[1]],point2[o[1]]],[point1[o[2]],point2[o[2]]], 'blue')    

        ani = animation.FuncAnimation(fig, update_points_lines, range(1, len(prmd)), interval=100, blit=False)
    plt.show()
    
    #train_features, train_labels = next(iter(data_loader))
    x = train_features.cuda() 
    #x = torch.from_numpy(x)
    z = model(x)
    print("encoded data shape", z.shape)
    z = z.permute(1,2,0)
    print("encoded data after permute 1", z.shape)
    z = z.to('cpu').detach().numpy()
      
    print("z data shape", z.shape)
       
    z = z.reshape(30,75)#(30,300)forConvolutionalAutoencoder
    z = z.reshape(30,25,3)#(30,25,3,4)
    #z = z.reshape(30,25,12)
    print("encoded data shape after permute 2", z.shape)
    n =0
    m = 5
    for i  in range(n,m):
        df = z[i,:,:]
        fig = plt.figure(tight_layout=True)
        ax = plt.axes(projection='3d')
        point_ani = ax.scatter3D(df[:, o[0]], df[:, o[1]], df[:, o[2]], cmap='green', linewidth =0.2)
        line = [[] for _ in range (len(J1))]
        for j in range(len(J1)):
            point1 = df[J1[j]-1]
            point2 = df[J2[j]-1]
            line[j] = ax.plot3D([point1[o[0]],point2[o[0]]],[point1[o[1]],point2[o[1]]],[point1[o[2]],point2[o[2]]], 'green')    

        ani = animation.FuncAnimation(fig, update_points_lines, range(1, len(prmd)), interval=100, blit=False)
        
    plt.show()
    
def plot_vae_generated_images(data_loader, model, device,figsize=(20, 2.5), n_images=15):                    
                          
    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    J1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25]
    J2=[2, 21, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 23, 8, 25, 12]
    train_features, train_labels = next(iter(data_loader))
    o = [0,1,2]  

    point = train_features.permute(1,2,0)
    print("shape of point", point.shape)
    point = point.reshape(30,75)
    print("shape of point after permute and reshape", point.shape)
    ok = point.reshape(30,25,3)    
    prmd = ok

    x = prmd[:,:,0]
    y = prmd[:,:,1]
    z = prmd[:,:,2]
    def update_points_lines(num):
            # Remove the previous frame
            ax.cla()
            # set x,y,z limits
            ax.set_xlim(np.min(z), np.max(z))
            ax.set_ylim(np.min(x), np.max(x))
            ax.set_zlim(np.min(y), np.max(y))

            # load data
            df = prmd[num]

            # draw points
            # ax.scatter3D(df[:, o[0]], df[:, o[1]], df[:, o[2]], cmap='Greens', linewidth=0.2)

            # draw line
            line = [[] for _ in range(len(J1))]
            for j in range(len(J1)):
                point1 = df[J1[j]-1]
                point2 = df[J2[j]-1]
                line[j] = ax.plot3D([point1[o[0]],point2[o[0]]],[point1[o[1]],point2[o[1]]],[point1[o[2]],point2[o[2]]], 'gray')
            return line
    n =0
    m = 5
    for i  in range(n,m):
    
    # x = x.permute(1,2,0)
    # x = x.reshape(30,75)
    # x = x.reshape(30,25,3)
        df = ok[i,:,:]
        fig = plt.figure(tight_layout=True)
        ax = plt.axes(projection='3d')
        point_ani = ax.scatter3D(df[:, o[0]], df[:, o[1]], df[:, o[2]], cmap='tab10', linewidth =0.2)
        line = [[] for _ in range (len(J1))]
        for j in range(len(J1)):
            point1 = df[J1[j]-1]
            point2 = df[J2[j]-1]
            line[j] = ax.plot3D([point1[o[0]],point2[o[0]]],[point1[o[1]],point2[o[1]]],[point1[o[2]],point2[o[2]]], 'blue')    

        ani = animation.FuncAnimation(fig, update_points_lines, range(1, len(prmd)), interval=100, blit=False)
    plt.show()
    
    #train_features, train_labels = next(iter(data_loader))
    x = train_features.cuda() 
    #x = torch.from_numpy(x)
    a,b,c,d = model(x)
    print("encoded data shape", d.shape)
    d = d.permute(1,2,0)
    print("encoded data after permute 1", z.shape)
    d = d.to('cpu').detach().numpy()
      
    print("d data shape", z.shape)
       
    d = d.reshape(30,75)#(30,300)forConvolutionalAutoencoder
    d = d.reshape(30,25,3)#(30,25,3,4)
    # z = z.reshape(30,25,12)
    print("encoded data shape after permute 2", z.shape)
    n =0
    m = 5
    for i  in range(n,m):
        df = d[i,:,:]
        fig = plt.figure(tight_layout=True)
        ax = plt.axes(projection='3d')
        point_ani = ax.scatter3D(df[:, o[0]], df[:, o[1]], df[:, o[2]], cmap='green', linewidth =0.2)
        line = [[] for _ in range (len(J1))]
        for j in range(len(J1)):
            point1 = df[J1[j]-1]
            point2 = df[J2[j]-1]
            line[j] = ax.plot3D([point1[o[0]],point2[o[0]]],[point1[o[1]],point2[o[1]]],[point1[o[2]],point2[o[2]]], 'green')    

        ani = animation.FuncAnimation(fig, update_points_lines, range(1, len(prmd)), interval=100, blit=False)
        
    plt.show()
   

         
                
def plot_latent_space_with_labels(num_classes, data_loader, model, device):
    d = {i:[] for i in range(num_classes)}

    model.eval()
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)
            
            out = model.encoder(features)
            out = out.cpu()
            nsamples, nx, ny = out.shape
            embedding = out.reshape((nsamples,nx*ny))
            
            tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embedding)
            for i in range(num_classes):
                if i in targets:
                    mask = targets == i
                    d[i].append(tsne[mask].to('cpu').numpy())

    colors = list(mcolors.CSS4_COLORS.items())#mcolors.TABLEAU_COLORS.items()
    
    for i in range(num_classes):
        d[i] = np.concatenate(d[i])
        plt.scatter(
            d[i][:, 0], d[i][:, 1],
            color=colors[i][1],
            label=f'{i}',
            alpha=0.5)

    plt.legend()

def plot_latent1(autoencoder, data_loader, num_batches=100):  
    np.random.seed(123)
    autoencoder = autoencoder.cuda()
    for i, (x, y) in enumerate(data_loader):
        x = x.cuda() 
        x = x.view(-1,2250).cuda()
        y = y.cpu()
        z_mean, z_log_var, z = autoencoder.encoder(x)
        print("encoded data shape", z.shape)
        #z = z.cpu().detach().numpy()
        tsne = TSNE().fit_transform(z.cpu().detach().numpy())
        plt.scatter(x =tsne[:, 0], y =tsne[:, 1])#, c = km.labels_
        if i > num_batches:
            plt.colorbar()
            break     

def plot_latent(autoencoder, data_loader, num_batches=100):            
     for i, (x, y) in enumerate(data_loader):
        x = x.to('cpu') 
        y = y.to('cpu')
        z = autoencoder.encoder(x)
        print("encoded data shape", z.shape)
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], cmap='tab10')#c=y,
        if i > num_batches:
            plt.colorbar()
            break 
def plot_knn_latent(autoencoder, data_loader, num_batches=100):            
     for i, (x, y) in enumerate(data_loader):
        x = x.to('cpu') 
        y = y.to('cpu')
        z = autoencoder.encoder(x)
        z = z.reshape(30,75)
        print("encoded data shape", z.shape)
        z = z.to('cpu').detach().numpy()
        ACC_H2 = []
        NMI_H2 = []
        PUR_H2 = []
        for i in range(10):
            kmeans = KMeans(n_clusters=max(np.int_(y).flatten()))
            Y_pred_OK = kmeans.fit_predict(z)
            Y_pred_OK = np.array(Y_pred_OK)
            labels = np.array(y).flatten()
            AM = clustering_metrics(y, labels)
            ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
            ACC_H2.append(ACC.item)
            print("ACC",ACC)
            NMI_H2.append(NMI)
            PUR_H2.append(PUR)
        plt.scatter(AM, ACC, cmap='tab10')#c=y,
        if i > num_batches:
            plt.colorbar()
            break  
def plot_vae_latent(autoencoder, data_loader, num_batches=100):
    autoencoder = autoencoder.cuda()
    for i, (x, y) in enumerate(data_loader):
        x = x.view(-1,2250).cuda()
        y = y.cpu()
        z_mean, z_log_var, encoded = autoencoder.encoder(x)
        print("encoded data shape", encoded.shape)
        z = torch.as_tensor(encoded).clone()
       # torch.as_tensor(np.array(pil_img).astype('float'))
        z = z.cpu().detach().numpy()#z.cpu().detach().numpy()
        plt.scatter(z[:, 0], z[:, 1],c=y, cmap='tab10')#, cmap='tab10'
        if i > num_batches:
            plt.colorbar()
            break 
def plot_vae_latent_colorbar(model, data_loader, num_classes=60):
    d = {i:[] for i in range(num_classes)}
    for i, (x, y) in enumerate(data_loader):
        x = x.cuda()
        x = x.view(-1,2250).cuda()
        y = y.to('cpu') 
        z_mean, z_log_var, encoded  = model.encoder(x)
        print(encoded.shape)
        z = encoded.to('cpu').detach().numpy()
        for i in range(num_classes):
            if (i-1) in y:
                mask = y == i
                d[i].append(z[mask])

    colors = list(mcolors.CSS4_COLORS.items())#mcolors.CSS4_COLORS #mcolors.TABLEAU_COLORS
    for i in range(num_classes):
        #d[i] = np.concatenate(z[i],axis =1)
        plt.scatter(z[:, 0], z[:, 1],color=colors[i][0],
        label=f'{i}',
        alpha=0.5)
    plt.colorbar()
    plt.legend()
    plt.show()    
def plot_images_sampled_from_vae(model, device, latent_size, unnormalizer=None, num_images=10):

    with torch.no_grad():

        ##########################
        ### RANDOM SAMPLE
        ##########################    

        rand_features = torch.randn(num_images, latent_size).to(device)
        new_images = model.decoder(rand_features)
        color_channels = new_images.shape[1]
        image_height = new_images.shape[2]
        image_width = new_images.shape[3]

        ##########################
        ### VISUALIZATION
        ##########################

        image_width = 28

        fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 2.5), sharey=True)
        decoded_images = new_images[:num_images]

        for ax, img in zip(axes, decoded_images):
            curr_img = img.detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax.imshow(curr_img)
            else:
                ax.imshow(curr_img.view((image_height, image_width)), cmap='binary') 

def plot_VAE_generated_images(data_loader, model, device,figsize=(20, 2.5), n_images=15):                    
                          
    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    J1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25]
    J2=[2, 21, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 23, 8, 25, 12]
    train_features, train_labels = next(iter(data_loader))
    o = [0,1,2]  

    point = train_features.permute(1,2,0)
    print("shape of point", point.shape)
    point = point.reshape(30,75)
    print("shape of point after permute and reshape", point.shape)
    ok = point.reshape(30,25,3)    
    prmd = ok

    x = prmd[:,:,0]
    y = prmd[:,:,1]
    z = prmd[:,:,2]
    def update_points_lines(num):
            # Remove the previous frame
            ax.cla()
            # set x,y,z limits
            ax.set_xlim(np.min(z), np.max(z))
            ax.set_ylim(np.min(x), np.max(x))
            ax.set_zlim(np.min(y), np.max(y))

            # load data
            df = prmd[num]

            # draw points
            # ax.scatter3D(df[:, o[0]], df[:, o[1]], df[:, o[2]], cmap='Greens', linewidth=0.2)

            # draw line
            line = [[] for _ in range(len(J1))]
            for j in range(len(J1)):
                point1 = df[J1[j]-1]
                point2 = df[J2[j]-1]
                line[j] = ax.plot3D([point1[o[0]],point2[o[0]]],[point1[o[1]],point2[o[1]]],[point1[o[2]],point2[o[2]]], 'gray')
            return line
    n =0
    m = 5
    for i  in range(n,m):
    
    # x = x.permute(1,2,0)
    # x = x.reshape(30,75)
    # x = x.reshape(30,25,3)
        df = ok[i,:,:]
        fig = plt.figure(tight_layout=True)
        ax = plt.axes(projection='3d')
        point_ani = ax.scatter3D(df[:, o[0]], df[:, o[1]], df[:, o[2]], cmap='tab10', linewidth =0.2)
        line = [[] for _ in range (len(J1))]
        for j in range(len(J1)):
            point1 = df[J1[j]-1]
            point2 = df[J2[j]-1]
            line[j] = ax.plot3D([point1[o[0]],point2[o[0]]],[point1[o[1]],point2[o[1]]],[point1[o[2]],point2[o[2]]], 'blue')    

        ani = animation.FuncAnimation(fig, update_points_lines, range(1, len(prmd)), interval=100, blit=False)
    plt.show()
    
    #train_features, train_labels = next(iter(data_loader))
    #x = train_features.cuda()
    x = train_features.view(-1,2250).to(device)
 
    z_mean, z_log_var, encoded, z = model(x)
    print("encoded data shape", z.shape)
    z = z.permute(1,0)
    print("encoded data after permute 1", z.shape)
    z = z.to('cpu').detach().numpy()
      
    print("z data shape", z.shape)
       
    z = z.reshape(30,75)#(30,300)forConvolutionalAutoencoder
    z = z.reshape(30,25,3)#(30,25,3,4)
    # z = z.reshape(30,25,12)
    print("encoded data shape after permute 2", z.shape)
    n =0
    m = 5
    for i  in range(n,m):
        df = z[i,:,:]
        fig = plt.figure(tight_layout=True)
        ax = plt.axes(projection='3d')
        point_ani = ax.scatter3D(df[:, o[0]], df[:, o[1]], df[:, o[2]], cmap='green', linewidth =0.2)
        line = [[] for _ in range (len(J1))]
        for j in range(len(J1)):
            point1 = df[J1[j]-1]
            point2 = df[J2[j]-1]
            line[j] = ax.plot3D([point1[o[0]],point2[o[0]]],[point1[o[1]],point2[o[1]]],[point1[o[2]],point2[o[2]]], 'green')    

        ani = animation.FuncAnimation(fig, update_points_lines, range(1, len(prmd)), interval=100, blit=False)
        
    plt.show()
             
def draw_figure(embed_array, label_array, name, epoch):
    embed_array = torch.cat(embed_array, dim=0).cpu().numpy()
    label_array = torch.cat(label_array, dim=0).cpu().numpy()
    
    
    embed_to2_arr = TSNE(n_components=2).fit_transform(embed_array)
        
    style_list = [['b', 'o'], ['b', '^'], ['b', '.'], ['b', '+'], ['b', '_'], ['b', 's'], ['b', '*'], ['b', '8'],
                  ['g', 'o'], ['g', '^'], ['g', '.'], ['g', '+'], ['g', '_'], ['g', 's'], ['g', '*'], ['g', '8'],
                  ['r', 'o'], ['r', '^'], ['r', '.'], ['r', '+'], ['r', '_'], ['r', 's'], ['r', '*'], ['r', '8'],
                  ['y', 'o'], ['y', '^'], ['y', '.'], ['y', '+'], ['y', '_'], ['y', 's'], ['y', '*'], ['y', '8'],
                  ['k', 'o'], ['k', '^'], ['k', '.'], ['k', '+'], ['k', '_'], ['k', 's'], ['k', '*'], ['k', '8'],
                  ['c', 'o'], ['c', '^'], ['c', '.'], ['c', '+'], ['c', '_'], ['c', 's'], ['c', '*'], ['c', '8'],
                  ['m', 'o'], ['m', '^'], ['m', '.'], ['m', '+'], ['m', '_'], ['m', 's'], ['m', '*'], ['m', '8'],
                  ['gray', 'o'], ['gray', '^'], ['gray', '.'], ['gray', '+'], ['gray', '_'], ['gray', 's'], ['gray', '*'], ['gray', '8'],]
    

    fig = plt.figure(0)
    # plt.subplot(2,1,1)
    km = KMeans(n_clusters=30).fit(embed_array)
    for i in range(label_array.shape[0]):
        label = label_array[i]
        # if label > 10: continue
        
        color, marker = style_list[label - 1]
        plt.scatter(embed_to2_arr[i][0], embed_to2_arr[i][1], c=color, marker=marker, alpha=0.8)
        
   
    plt.show()
    