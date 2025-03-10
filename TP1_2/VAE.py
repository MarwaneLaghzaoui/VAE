
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm



# Preprocessing 
transform = transforms.Compose([
    transforms.Pad(2),  # Pad 28x28 images to 32x32
    transforms.ToTensor(),  # Convert images to tensors
])

batch_size = 64


#load data
train_dataset = datasets.FashionMNIST(root="./data", train= True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class_labels = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}

#create VAE model

class VAE(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # ------------------
        #     1. Encoder
        # ------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU()
        )

        flattened_size, decode_shape = self.calculate_flattened_size(self.encoder, input_shape)

        self.fc_mu = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_size, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_size, latent_dim)
        )

        # ------------------
        #     3. Decoder
        # ------------------
        self.decoder = nn.Sequential(
            # La première couche prend latent_dim et le transforme en 128x4x4
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=1),  # 1x1 -> 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.Sigmoid()
        )


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
        
        z = mu + sigma * eps, where eps ~ N(0, I)

        Args:
            mu (Tensor): Mean of the latent distribution.
            logvar (Tensor): Log-variance of the latent distribution.

        Returns:
            z (Tensor): Latent variable sampled from N(mu, sigma^2).
        """
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        eps = torch.randn_like(std)    # Sample noise from a normal distribution
        return mu + eps * std

    def calculate_flattened_size(self, model, input_shape):
       #same as the last one 
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = model(dummy_input)
            return output.numel(), output.shape

    def forward(self, x):
        """
        Forward pass for the VAE:
        
        1) Encode input into latent distribution parameters (mu, logvar).
        2) Sample z using the reparameterization trick.
        3) Decode z back to a reconstructed image.

        Args:
            x (Tensor): Input images.

        Returns:
            recon_x (Tensor): Reconstructed images.
            mu (Tensor): Mean of latent distribution.
            logvar (Tensor): Log-variance of latent distribution.
            z (Tensor): Sampled latent variable.
        """
        # Encode input
        x_encoded = self.encoder(x)

        # Compute mu and logvar
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)

        # Sample from the latent distribution
        z = self.reparameterize(mu, logvar)

        # Redimensionner z correctement avant de passer dans le décodeur
        # Cela dépend de la taille de l'encodeur et de la dimension latente
        batch_size = z.size(0)
        z = z.view(batch_size, self.latent_dim, 1, 1)

        # Decode latent vector to reconstruct the input
        recon_x = self.decoder(z)

        return recon_x, mu, logvar, z



#######################
# 1. Define VAE Loss
#######################
def vae_loss(recon_x, x, mu, logvar, beta=1):
    """
    Compute the Variational Autoencoder (VAE) loss function.
    A combination of:
      - Reconstruction loss (using BCE or MSE)
      - KL divergence regularizer

    Args:
      recon_x (Tensor): Reconstructed images from the decoder.
      x (Tensor): Original input images.
      mu (Tensor): Mean of the latent distribution.
      logvar (Tensor): Log-variance of the latent distribution.
      beta (float): Weight for the KL term (β-VAE concept).

    Returns:
      total_loss (Tensor): Sum of reconstruction loss and β * KL divergence.
    """

    # 1) Reconstruction Loss
    #    Measures how closely recon_x matches x. 
    #    Typically use Binary Cross Entropy (BCE) if inputs are normalized [0,1].
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')


    # 2) KL Divergence
    #    Encourages the approximate posterior (q(z|x)) to be close to a 
    #    standard normal prior p(z) ~ N(0,I).
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
     # Return the total VAE loss
    return recon_loss + beta * kl_div

#######################
# 2. Training Function
#######################
def train_model(model, train_loader, val_loader, epochs, beta=1):
    """
    Trains a VAE model with a given β for the KL term.

    Args:
      model (nn.Module): VAE instance (encoder + decoder).
      train_loader (DataLoader): Dataloader for training set.
      val_loader (DataLoader): Dataloader for validation set.
      epochs (int): Number of training epochs.
      beta (float): Weight for the KL divergence in the VAE loss.
    """

    # Choose an optimizer, e.g., Adam
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Set model to train mode
        total_loss = 0
        
        # Use tqdm to create a progress bar
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for x, _ in tqdm_loader:
            # 1) Zero out gradients from previous iteration
            optimizer.zero_grad()

            # 2) Move the batch to the correct device (CPU, CUDA, or MPS)
            #move to cpu
            x = x.to("cpu")

            # 3) Forward pass: encode -> reparameterize -> decode
            recon_x, mu, logvar, _ = model(x)

            # 4) Compute VAE loss
            loss = vae_loss(recon_x, x, mu, logvar, beta=beta)

            # 5) Backpropagation
            loss.backward()
            # 5) update grad
            optimizer.step()

            # Accumulate total loss for this batch
            total_loss += loss.item()

            # Update tqdm progress bar with the current batch loss
            tqdm_loader.set_postfix(loss=loss.item())

        # Average loss over all training samples
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Evaluate on validation data
        avg_val_loss, _ = evaluate_model(model, val_loader)

        # Print epoch statistics
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Store the average loss for this epoch
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
    return train_losses, val_losses



#######################
# 3. Evaluation Function
#######################
def evaluate_model(model, loader ,beta=1):
    """
    Evaluates the VAE on a validation or test dataset.

    Args:
      model (nn.Module): VAE instance (encoder + decoder).
      loader (DataLoader): Dataloader for validation/test set.

    Returns:
      avg_loss (float): Average loss across all validation samples.
      ce_loss_placeholder (float): Placeholder if you want to track
                                   additional metrics or losses.
beta (float): Weight for the KL divergence in the VAE loss.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    ce_loss_placeholder = 0  # Example placeholder for separate metrics

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Evaluating"):
            x = x.to("cpu")  # Move data to the same device as the model

            # Forward pass to get reconstruction and latent variables
            recon_x, mu, logvar, _ = model(x)

            # Compute VAE loss (without adjusting gradients)
            loss = vae_loss(recon_x, x, mu, logvar, beta=1)  # e.g., same β as training

            total_loss += loss.item()
            # Optionally compute or track other metrics here, e.g.:
            # ce_loss_placeholder += some_other_metric(...)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, ce_loss_placeholder

def plot_training_loss(train_losses, val_losses):
    """
    Plot the training and validation loss across epochs.

    - train_losses: List of training losses per epoch.
    - val_losses: List of validation losses per epoch.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="s")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()


# Create a new VAE model
model = VAE()
# Train the model for 10 epochs
train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=10, beta=1)
# Plot the training and validation loss
plot_training_loss(train_losses, val_losses)


# Save the trained model
#torch.save(model.state_dict(), "vae_model.pth")

# Load the trained model
model = VAE()
model.load_state_dict(torch.load("vae_model.pth"))
model.eval()

# Evaluate the model on the validation set
avg_val_loss, _ = evaluate_model(model, val_loader)
print(f"Validation Loss: {avg_val_loss:.4f}")

# Generate new samples from the VAE
def generate_samples(model, num_samples=10):
    """
    Generate new samples from the VAE by sampling from the latent space.

    Args:
      model (nn.Module): Trained VAE model.
      num_samples (int): Number of samples to generate.

    Returns:
      samples (Tensor): Generated samples from the VAE.
    """
    with torch.no_grad():
        # Sample from a standard normal distribution
        z = torch.randn(num_samples, model.latent_dim, 1, 1)
        # Decode the samples to generate new images
        samples = model.decoder(z)
    return samples

# Generate 10 new samples from the VAE
samples = generate_samples(model, num_samples=10)

# Plot the generated samples
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(samples[i].squeeze(), cmap="gray")
    plt.title(f"Sample {i + 1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

#visualize the latent space
def plot_latent_space(model, loader, num_batches=100):
    """
    Plot the latent space of a trained VAE using a subset of the data.

    Args:
      model (nn.Module): Trained VAE model.
      loader (DataLoader): Dataloader for the dataset.
      num_batches (int): Number of batches to use for visualization.
    """
    model.eval()  # Set model to evaluation mode
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break

            x = x.to("cpu")
            _, mu, _, _ = model(x)
            latent_vectors.append(mu)
            labels.append(y)

        latent_vectors = torch.cat(latent_vectors, dim=0)
        labels = torch.cat(labels, dim=0)

    # Plot the latent space using the first two dimensions
    plt.figure(figsize=(8, 6))
    for i in range(10):
        indices = labels == i
        plt.scatter(latent_vectors[indices, 0], latent_vectors[indices, 1], label=class_labels[i])
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Latent Space Visualization")
    plt.legend()
    plt.grid()
    plt.show()

# Visualize the latent space of the VAE
plot_latent_space(model, val_loader, num_batches=100)
