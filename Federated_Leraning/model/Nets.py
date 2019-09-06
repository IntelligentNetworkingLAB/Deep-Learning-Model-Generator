import torch
from torch import nn
import torch.nn.functional as F

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