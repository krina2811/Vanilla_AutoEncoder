import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from Autoencoder import Autoencoder
from Auto_Encoder_jay import AutoEncoder, Encoder, Decoder
import os

transform = transforms.ToTensor()
mnist_Data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset= mnist_Data,
                                                  batch_size=32,
                                                  shuffle = True)



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    save_dir = "./checkpoint/1_vanilla_Autoencoder"
    os.makedirs(save_dir, exist_ok=True)

    model = Autoencoder(input_shape=(1, 28, 28),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2)


    model.to(device)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Train the autoencoder
    num_epochs = 50
    for epoch in range(num_epochs):
        for data in data_loader:
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)          
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
        
        print('Epoch [{}/{}] -------------------------------------------, Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        if epoch % 5== 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"vanilla_autoencoder_{epoch}.pth"))
            
 