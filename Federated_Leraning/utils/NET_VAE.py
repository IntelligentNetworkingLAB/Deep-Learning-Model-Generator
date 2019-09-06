import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import torch

from utils.options import args_parser
from data.preprocess import data_preprocessing, load_train_data, load_tr_te_data
from model.Nets import VAE
from model.Update import *
from SOCKET.Server import * 
from socket import *

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    print(torch.cuda.is_available())

    if args.device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
   
    
    #isNotProcessed == 1 for preprocessing
    if args.preprocess == 1 :
        #Data Preprocessing
        data_preprocessing(args.data_dir)
    

    pro_dir = os.path.join(args.data_dir, 'pro_sg')
  
    #getting user's unique id
    unique_sid = list()
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip()) #line.strip() 양쪽 공백과 \n을 삭제해준다.

    n_items = len(unique_sid)
       
    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), n_items)

    #10 user's data distribute
    user_dir = os.path.join(args.data_dir, 'pro_sg/user_data')
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    for i in args.user_IDs:
        #store user's data
        pd.DataFrame(train_data[i].indices).to_csv(os.path.join(user_dir, 'user{}.csv'.format(i)), index=False)

    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'), os.path.join(pro_dir, 'validation_te.csv'), n_items)
    
    #number of data size
    N = train_data.shape[0]
    N_vad = vad_data_tr.shape[0]

    idxlist = range(N)

    # training batch size
    batches_per_epoch = int(np.ceil(float(N) / args.batch_size)) #np.ceil() 은 요소들을 올림 ######################################################################

    N_vad = vad_data_tr.shape[0]

    idxlist_vad = range(N_vad)
    
    #Model size
    p_dims = [200, 600, n_items]

    #build model
    model = VAE(p_dims).to(args.device)
    print(model)

    #training
    ndcgs_list = []
    r20_list = []
    r50_list = []
    loss_list = []

    best_ndcg = -np.inf
    #for annealing
    update_count = 0.0

    #Socket
    runServer()

    for epoch in range(args.n_epochs) :
        #FL participants
        idxs_users = np.random.choice(N, args.n_participants, replace=False)
        #data of FL participants
        data = torch.FloatTensor(train_data[idxs_users].toarray()).to(args.device)

        idx_client = 0
            
        total_loss = 0 

        local_up = Local_Update(args = args)

        w, loss, update_count = local_up.train(net = model, train_data = data, update_count = update_count)
     
        model.load_state_dict(w)
   
        if epoch % args.check_point == 0:
            # compute validation NDCG            
            ndcg_,r20, r50 = evaluate(args, model, vad_data_tr, vad_data_te, update_count)
        
            ndcgs_list.append(ndcg_)
            r20_list.append(r20)
            r50_list.append(r50)
            loss_list.append(loss.item())

            print("||epochs::",epoch,"\t||recall20::",r20,"\t||recall50::",r50,"\t||ndcg::", ndcg_,"\t||loss::", loss.item(),"||")

            # update the best model (if necessary)
            if ndcg_ > best_ndcg:
                best_ndcg = ndcg_
    
    plt.figure(figsize=(12, 3))
    plt.plot(ndcgs_list)
    plt.plot(r20_list)
    plt.plot(r50_list)
    plt.ylabel("Validation NDCG@100")
    plt.xlabel("Epochs")        
    plt.savefig('./log/performance.png')

    
    plt.figure(figsize=(12, 3))
    plt.plot(loss_list)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")        
    plt.savefig('./log/loss.png')        