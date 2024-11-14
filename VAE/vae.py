import torch 
from .base import BaseVAE
from torch import nn 
from torch.nn import functional as F 
from .types_ import * 





class VanillaVAE(BaseVAE):

    def __init__(self, 
                 in_channels: int,
                 latent_dim:int,
                 hidden_dims: List = None,
                 **kwargs) -> None: 
        
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        
        module = []




    def encoder(self, input:Tensor) -> List[Tensor]:

        """ 
        Encodes the input by passing through the encoder network 
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encode [N x C x H x W] = (batch_size, channels, Height, width)
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # split the result into mu and var components 
        # of the latent Gaussian distribution 
        mu = self.fc_mu(result)   # fully connected mean
        log_var = self.fc_var(result)  # fully connected variance 

        return [mu, log_var]
    


    def decoder(self, z: Tensor) -> Tensor:

        """ 
        Maps the given latent code into the image space 
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B, C, H, W]
        """ 

        result = self.decoder_input(z)
        result = self.view(-1, 512, 2, 2)   # (batch_size, channels, Height , width)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    

    def loss_function(self, *args, **kwargs) -> dict:

        """
        Compute the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons = args[0]
        input = args[1]
        mu = args[2] # mean 
        log_var = args[3]   # log variance 

        kid_weigth = kwargs['M_N']   # Account for the minibatch samples from the dataset 
        recons_loss = F.mse_loss(recons, input)


        kid_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kid_weigth *  kid_loss

        return {
            'loss': loss,
            'Reconstruction_loss': recons_loss.detach(),
            'KLD': -kid_loss.detach()
        }
    

    
    


