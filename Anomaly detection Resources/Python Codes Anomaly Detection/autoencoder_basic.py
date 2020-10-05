import torch

import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
#import pytorch_ssim

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        block = BasicBlock
        
        self.encoder_1 = block(28 * 28, 128)        
        self.encoder_2 = block(128, 64)  
        self.encoder_3 = block(64, 12)
        self.encoder_4 = block(12, 3)
        
        self.decoder_4 = block(3, 12)    
        self.decoder_3 = block(12, 64)
        self.decoder_2 = block(64, 128)
        self.decoder_1 = block(128, 28 * 28)
        
#        self.ssim_loss = pytorch_ssim.SSIM(window_size=11)
#        self.ssim_loss_nomean = pytorch_ssim.SSIM(window_size=11, size_average=False)

        # Defininf the forward feed
    
    def forward(self, x):
        #encoder
        
        # 784
        e1 = self.encoder_1(x)
        # 128
        e2 = self.encoder_2(e1)
        # 64
        e3 = self.encoder_3(e2)
        # 12
        e4 = self.encoder_4(e3)
        # 3
        
        #decoder
        # 3
        d3 = self.decoder_4(e4)
        # 12
        d2 = self.decoder_3(d3)
        # 64
        d1 = self.decoder_2(d2)
        # 128
        y = self.decoder_1(d1)
        # 784
#        y = torch.sigmoid(y)
#        x = self.encoder(x)
#        x = self.decoder(x)
        return y

    def to_img(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        # print (x)
        return x

    def loss_function(self, recon_x, x, domean=True):

        if domean:
            #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
            MSE = F.mse_loss(recon_x, x.view(x.size(0), -1))
            #SSIM= -self.ssim_loss(self.to_img(recon_x), self.to_img(x))
        else:
            MSE = F.mse_loss(recon_x, x.view(x.size(0), -1), reduction='none')
            MSE = MSE.mean(dim=1)
            #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='none')
            #BCE = BCE.mean(dim=1)
            #SSIM = -self.ssim_loss_nomean(self.to_img(recon_x), self.to_img(x))


        return MSE

"""
    def loss_function_bce(self, recon_x, x, domean=True):
        if domean:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='none')
            BCE = BCE.mean(dim=1)
        return BCE
"""

#
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicBlock, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x
    
if __name__=='__main__':
    model = autoencoder()
    print(model)