import numpy as np
import torch
import torch.nn.functional as F
import bottleneck as bn
from time import sleep
class Local_Update(object):
    def __init__(self, args):
        self.args = args
  
    def train(self, net, train_data, update_count, neg_list, kl_list):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
        loss = []
        
        if self.args.total_anneal_steps > 0:
            anneal = min(self.args.anneal_cap, 1. * update_count / self.args.total_anneal_steps)
        else:
            anneal = self.args.anneal_cap
        
        optimizer.zero_grad()
            
        logits, mu, logvar = net(train_data)
      
        #loss definition (neg_ELBO = loss)
        neg_ll = -torch.mean(torch.sum(F.log_softmax(logits, 1) * train_data, -1))
        neg_ll = ( neg_ll * (self.args.n_participants - len(neg_list)) + sum(neg_list) ) / self.args.n_participants
        
        KL = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        KL = ( KL * (self.args.n_participants - len(kl_list)) + sum(kl_list) ) / self.args.n_participants

        neg_ELBO = neg_ll + anneal * KL
        loss.append(neg_ELBO)
        neg_ELBO.backward()
          
        optimizer.step()

        update_count += 1
        
        
        return net.state_dict(), sum(loss)/len(loss) , update_count

def evaluate(args, model, data_tr, data_te, update_count):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]

    n100_list = []
    r20_list = []
    r50_list = []
    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size_vad):
            end_idx = min(start_idx + args.batch_size_vad, e_N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = torch.FloatTensor(data.toarray()).to(args.device)

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 
                               1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data_tensor)

            
            #total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)
            #print("n100::",n100)
       

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)
          

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall