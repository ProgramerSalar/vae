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
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder 
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )

            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder 
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),

                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),

            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3,
                      padding=1),
                      
            nn.Tanh()
        )



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
    

    def reparametrize(self, mu: Tensor, logvar: Tensor) -> Tensor:

        """ 
        Reparametrization trick to sample from N(mu, var) from N(0, 1)
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]   -> (batch_size, dimension_size)
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu 
    

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparametrize(mu, log_var)
        return [self.decode(z), input, mu, log_var]
    

    def sample(self,
               num_samples: int,
               current_device: int,
               **kwargs) -> Tensor:
        
        """ 
        Samples from the latent space and return the corresponding image space map.
        :param num_samples: (Int) number of sampels 
        :param current_device: (Int) Device to run the model 
        :return: (Tensor)
        """

        z = torch.randn(num_samples,
                        self.latent_dim)
        
        z = z.to(current_device)

        samples = self.decode(z)
        return samples
    

    def generate(self, x: Tensor, **kwargs) -> Tensor: 

        """ 
        Given an input image x, return the reconstructed image 
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



    
    


