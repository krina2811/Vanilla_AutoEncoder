import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Autoencoder(nn.Module):
    """
        Autoencoder represents a deep convolutional autoencoder archi with mirrored encoder and decoder components
    """

    def __init__(self, input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        
        super(Autoencoder, self).__init__()

        self.input_shape = input_shape  # [ch, h, w]
        self.conv_filters = conv_filters
        self.conv_kernals = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self._num_conv_layers = len(conv_filters)

        self.decoder_input_shape = None
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64*3*3, self.latent_space_dim)
        self.linear2 = nn.Linear(self.latent_space_dim, 64*3*3)



    def _build_encoder(self):
        enc_layers = []
        for i in range(self._num_conv_layers):
            if i == 0:
                enc_layers.append(torch.nn.Conv2d(1, self.conv_filters[i], self.conv_kernals[i], self.conv_strides[i]))
            else:
                enc_layers.append(torch.nn.Conv2d(self.conv_filters[i-1], self.conv_filters[i], self.conv_kernals[i], self.conv_strides[i]))
            enc_layers.append(nn.ReLU(inplace=True))     
        return nn.Sequential(*enc_layers)



    def _build_decoder(self):
        dec_layers = []
        for i in reversed(range(1, self._num_conv_layers)):
            print("i", i)
            if i==1 or i==2: ######3 if conv layer has stride=2 we need output_padding=1
                dec_layers.append(nn.ConvTranspose2d(self.conv_filters[i], self.conv_filters[i-1], self.conv_kernals[i], self.conv_strides[i], output_padding=1))
            else:
                dec_layers.append(nn.ConvTranspose2d(self.conv_filters[i], self.conv_filters[i-1], self.conv_kernals[i], self.conv_strides[i]))
            dec_layers.append(nn.ReLU(inplace=True))
        
        dec_layers.append(nn.ConvTranspose2d(self.conv_filters[0], 1, self.conv_kernals[0], self.conv_strides[0]))
        return nn.Sequential(*dec_layers)

    def reconstruct(self, images):
        x = self.encoder(images)
        self.decoder_input_shape = x.shape
        x = self.flatten(x)
        latent_representations = self.linear1(x)
        
        dec_input = self.linear2(latent_representations)
        dec_input = dec_input.view(self.decoder_input_shape)
        reconstructed_images = self.decoder(dec_input)
        return reconstructed_images, latent_representations

    def forward(self, x):
        x = self.encoder(x)        
        self.decoder_input_shape = x.shape  
        x = self.flatten(x)
        x = self.linear1(x)
        
        dec_input = self.linear2(x) 
        dec_input = dec_input.view(self.decoder_input_shape)
        x = self.decoder(dec_input)
        # print(x.shape)
        # exit()
        return x
        
        