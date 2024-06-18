import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Autoencoder import Autoencoder
import torch


def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        image = image.detach().cpu()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        reconstructed_image = reconstructed_image.detach().cpu()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    model = Autoencoder(input_shape=(1, 28, 28),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2)

    model.load_state_dict(torch.load('./checkpoint/1_vanilla_Autoencoder/vanilla_autoencoder.pth'))
    model.eval()

    transform = transforms.ToTensor()
    mnist_Data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(dataset= mnist_Data,
                                                  batch_size=1000,
                                                  shuffle = True)
    
    examples = enumerate(test_data_loader)
    batch_idx, (x_test, y_test) = next(examples)
    
    num_sample_images_to_show = 8
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    reconstructed_images, _ = model.reconstruct(sample_images)   
    plot_reconstructed_images(sample_images, reconstructed_images)

    num_images = 6000
    sample_images, sample_labels = select_images(x_test, y_test, num_images)
    _, latent_representations = model.reconstruct(sample_images)
    plot_images_encoded_in_latent_space(latent_representations.detach().cpu(), sample_labels.detach().cpu())


