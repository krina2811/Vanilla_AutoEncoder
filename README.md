# Vanilla_AutoEncoder

An autoencoder is an artificial neural network used for unsupervised learning tasks (i.e., no class labels or labeled data) such as dimensionality reduction, feature extraction, and data compression. They seek to:

-> Accept an input set of data (i.e., the input)
<br>
-> Internally compress the input data into a latent space representation (i.e., a single vector that compresses and quantifies the input)
<br>
-> Reconstruct the input data from this latent representation (i.e., the output)
<br>

--------------- An autoencoder consists of the following two primary components:---------------------

Encoder: The encoder compresses input data into a lower-dimensional representation known as the latent space or code. This latent space, often called embedding, aims to retain as much information as possible, allowing the decoder to reconstruct the data with high precision. If we denote our input data as x and the encoder as E, then the output latent space representation, s, would be s=E(x).


Decoder: The decoder reconstructs the original input data by accepting the latent space representation s. If we denote the decoder function as D and the output of the detector as o, then we can represent the decoder as o = D(s).
Both encoder and decoder are typically composed of one or more layers, which can be fully connected, convolutional, or recurrent, depending on the input data’s nature and the autoencoder’s architecture.


 
