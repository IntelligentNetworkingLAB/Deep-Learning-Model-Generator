import socket
import torch
import os
import pandas as pd
import numpy as np
from threading import Thread
from torch import nn
import torch.nn.functional as F
import argparse
from scipy import sparse

HOST = 'localhost'
PORT = 9009
user_id = -1

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--data_dir', type=str, default='data/ml-20m/pro_sg/user_data', help='Movielens-20m dataset location')
args = parser.parse_args()



class VAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """
    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(VAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
        self.dims = self.q_dims + self.p_dims[1:] #p_dims[1:] : [600, 20108] 즉 1번째 index부터 끝까지
        self.tanh = nn.Tanh()
        #encoder
        self.encoder_input_layer = nn.Linear(self.dims[0], self.dims[1],bias=True)
        self.encoder_hidden_layer = nn.Linear(self.dims[1], 2 * self.dims[2],bias=True) #200개는 평균, 200개는 표준편차 400개가 output

        #decoder
        self.decoder_input_layer = nn.Linear(self.dims[2], self.dims[3],bias=True) #평균 + 표준편차 * epsilon 해서 200개가 input
        self.decoder_output_layer = nn.Linear(self.dims[3], self.dims[4],bias=True)
    
    def forward(self, input):
        mu, logvar = self.encode(input)

        if self.training:  
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
         
        else:
            
            z= mu

        return self.decode(z), mu, logvar
    
    def encode(self, input):
        x = F.normalize(input) #dim=0 이냐 1이냐가 문제다
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.tanh(self.encoder_input_layer(x))
        x = self.encoder_hidden_layer(x)
        mu = x[:, :self.q_dims[-1]]
        logvar = x[:, self.q_dims[-1]:]

        return mu, logvar

    
    def decode(self, z):
        x = self.tanh(self.decoder_input_layer(z))
        output = self.decoder_output_layer(x) 
        return output


def train(net, train_data):
   
   net.train()
   logits, mu, logvar = net(train_data)
   return logits, mu, logvar 


def load_train_data(csv_file, n_items):
 
   tp = pd.read_csv(csv_file)

   n_users = 1

   rows, cols = np.zeros(len(tp['sid'])), tp['sid']
 
   data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, n_items))
 
   return data

def rcvMsg(sock):
   
   
   data = sock.recv(1024)
   print(data.decode())
   count = 0
   if not os.path.exists('download'):
      os.makedirs('download')
   
   

   while True:
      try:
         #Getting file size
         data = sock.recv(1024)
         model_size = int.from_bytes(data, byteorder='big') 
         
         #downloaded file size
         down_size = 0
         print("model_size", model_size)


         #data = sock.recv(1024)

         #if not data:
         #   break
         
         with open('download/downModel{}'.format(user_id), 'wb') as f:
            try:
               while True:
                  data = sock.recv(1024)
                  down_size += len(data)
                  f.write(data)
                  if down_size == model_size: break
               '''
               while data: 
                  #if data == b'/end' : break
                  f.write(data)
                  if model_size=
                  data = sock.recv(1024)
               '''
            except Exception as e:
               print(e)
         f.close
         #print("data", data)
         p_dims = [200, 600, 20108]

         #loss
         model = VAE(p_dims).to('cpu')
         #.to(args.device)
      
         try:
            model.load_state_dict(torch.load('download/downModel{}'.format(user_id)))
         except Exception as e:
               print(e)

         os.remove('download/downModel{}'.format(user_id))

         train_data = load_train_data(os.path.join(args.data_dir, 'user{}.csv'.format(user_id)), 20108)
         
         data = torch.FloatTensor(train_data.toarray())
         #.to(args.device)
        
         logits, mu, logvar = train(model, data)
        
         neg_ll = -torch.mean(torch.sum(F.log_softmax(logits, 1) * data, -1))
     
         KL = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
 
         print("neg_ll :", neg_ll, "KL :", KL)
         try:
            with torch.no_grad():
               np_neg = neg_ll.detach().numpy()
               np_KL = KL.detach().numpy()
               #np_neg = neg_ll.cpu().numpy()
               #np_KL = KL.cpu().numpy()
         except Exception as e:
            print(e)
         b_neg = np_neg.tobytes()
         b_KL = np_KL.tobytes()
         sock.send(b_neg)########################################
         
         sock.send(b_KL)#########################################

      except:
         pass
 
def runChat():
   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.connect((HOST, PORT))
      t = Thread(target=rcvMsg, args=(sock,))
      t.daemon = True
      t.start()
      count = 0
      while True:
         msg = input()
         if count == 0:
            global user_id
            user_id = msg
          
            count+=1
         if msg == '/quit':
            sock.send(msg.encode())
            break
 
         sock.send(msg.encode())
      
     
             
if __name__ == '__main__':       
   
   #args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')      
   runChat()



