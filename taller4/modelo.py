import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latend_dim = latent_dim
        # usando nn.Sequential definan las capas del stacked autoencoder, 
        # tanto en el encoder, como en el decoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(len(x), -1)  #asi x tiene en su primera dimension al batch que se le pasa, y el -1 indica que en esa dimension están todos los pixeles

        x = self.encoder(x)
        x = self.decoder(x)
        
        x = x.view(len(x), 1, 28, 28)
        return x