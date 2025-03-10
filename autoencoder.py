import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), latent_dim=2):
        """
        Initialize the Autoencoder model.
        - input_shape: The shape of the input images (1, 32, 32) for grayscale images.
        - latent_dim: The number of neurons in the latent space.
        """
        super(Autoencoder, xxx).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # First convolutional layer: Convert 1-channel input into 32 feature maps.
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.xxx(), # acivation function 

            # Second convolutional layer: Reduce spatial dimensions further.
            nn.xxx(32, 64, kernel_size=3, stride=2, padding=1),
            nn.xxx(), #acivation function 

            # Third convolutional layer: Extract higher-level features.
            nn.Conv2d(kernel_size=3,)  # Add another Conv2D layer with 128 filters, kernel size 3, stride 2, padding 1
            nn.xxx()  #acivation function 

        )

        # Dynamically calculate the flattened size after encoding
        flattened_size, decode_shape = self.calculate_flattened_size(self.encoder, input_shape)

        # Latent space (fully connected layer)
        self.fc = nn.Sequential(
            nn.xxx(start_dim=1),  # Flatten the encoded features
            nn.Linear(flattened_size, xxx)  # Replace xxx with the latent dimension
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(xxx, flattened_size),  # Map latent space back to the feature map
            nn.Unflatten(decode_shape[0],decode_shape[1:]),  # Reshape to match the encoded feature map

            # Transposed Convolution layers (Decoder)
            nn.xxx(xxx, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Add a conv transpose 2d
            nn.xxx(), # acivation function 


            # Next deconvolution layer
            xxx,  # Add a ConvTranspose2d layer reducing from 64 channels to 32 channels and  kernel_size=3, stride=2, padding=1, output_padding=1


            nn.xxx(), # acivation function 

            # Final deconvolution layer: Convert back to single-channel grayscale image
            nn.xxx(xxx, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.xxx()  # acivation function 
 Output values should be between 0 and 1
        )

    def calculate_flattened_size(self, model, input_shape):
        """
        Helper function to determine the flattened size after the encoder.
        - model: The encoder model.
        - input_shape: The shape of the input tensor.
        Returns:
        - The flattened size (number of features) and output shape.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Create a batch of one image
            output = model(dummy_input)
            return output.numel(), output.shape  # Return total number of elements and shape

    def forward(self, x):
        """
        Forward pass through the Autoencoder.
        - x: Input image tensor.
        Returns:
        - Reconstructed image
        - Latent space representation
        """
        x = self.xxx(x)  # Pass through the encoder
        latent = self.fc(x)  # Project into latent space
        x = self.xxx(latent)  # Decode the latent representation
        return x, latent

# Test dimension consistency
x_sample = torch.randn(1, 1, 32, 32)  # Example input
model_test = Autoencoder()
image_test,latent_space = model_test(x_sample)
assert image_test.shape == x_sample.shape, "Output dimensions do not match input dimensions!"