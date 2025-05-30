import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def resnet34_(in_channels: int, n_classes: int, dropout: float = 0.5, weights: torch.nn.Module = None) -> torch.nn.Module:
    """
    Constructs a modified ResNet-34 model with a custom input layer and fully connected layer.

    Args:
        in_channels (int): Number of input channels for the convolutional layer.
        n_classes (int): Number of output classes for the final layer.
        dropout (float, optional): Dropout rate for the fully connected layers. Default is 0.5.
        weights (torch.nn.Module, optional): Pre-trained weights for the ResNet-34 model. Default is None.

    Returns:
        torch.nn.Module: A modified ResNet-34 model.
    """
    model = models.resnet34(weights=weights)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(
                            nn.BatchNorm1d(512),
                            nn.Dropout(p=dropout, inplace=False),
                            nn.Linear(in_features=512, out_features=64, bias=False),
                            nn.ReLU(inplace=True),
                            
                            nn.BatchNorm1d(64),
                            nn.Dropout(p=dropout, inplace=False),
                            nn.Linear(in_features=64, out_features=n_classes, bias=True)
                            )
    return model


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = False) -> None:
        """
        Initializes a ResNet block with optional upsampling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            upsample (bool, optional): If True, applies upsampling. Default is False.
        """
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        
        if upsample:
            self.upsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResNet block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size,
                              C is the number of channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: Output tensor after passing through the ResNet block, with the same shape as input.
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.upsample:
            identity = self.upsample_layer(x)
        
        out += identity
        out = self.relu(out)
        
        return out
class Encoder(nn.Module):
    def __init__(self, num_channels: int = 3) -> None:
        """
        Initializes the Encoder with a modified ResNet-18 architecture.

        Args:
            num_channels (int, optional): Number of input channels. Default is 3.
        """
        super(Encoder, self).__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size,
                              C is the number of channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: Encoded feature tensor.
        """
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, num_channels: int = 3) -> None:
        """
        Initializes the Decoder with a series of transposed convolutional layers.

        Args:
            num_channels (int, optional): Number of output channels. Default is 3.
        """
        super(Decoder, self).__init__()
        self.decoder_linear = nn.Linear(512, 512 * 7 * 7)  # Upsample to 7x7 feature map

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_channels, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.Sigmoid()
        )

    def forward(self, feature_vectors: torch.Tensor, target_size: torch.Size) -> torch.Tensor:
        """
        Forward pass for the Decoder.

        Args:
            feature_vectors (torch.Tensor): Input feature tensor of shape (N, 512).
            target_size (torch.Size): Target size for the output tensor, typically (N, C, H, W).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (N, C, H, W).
        """
        batch_size = feature_vectors.size(0)
        h = self.decoder_linear(feature_vectors).view(batch_size, 512, 7, 7)
        h = self.decoder(h)
        
        # Final interpolation to match the target size
        h = F.interpolate(h, size=(target_size[2], target_size[3]), mode='bilinear', align_corners=False)
        return h


class VideoRegressionModel(nn.Module):
    def __init__(self, num_frames: int, num_channels: int, hidden_dim: int, num_layers: int, num_heads: int, mlp_dim: int, output_dim: int) -> None:
        """
        Initializes the VideoRegressionModel with an encoder, decoder, and transformer.

        Args:
            num_frames (int): Number of frames in the input video.
            num_channels (int): Number of channels in each frame.
            hidden_dim (int): Dimension of the hidden layer in the transformer.
            num_layers (int): Number of layers in the transformer encoder and decoder.
            num_heads (int): Number of attention heads in the transformer.
            mlp_dim (int): Dimension of the feedforward network model in the transformer.
            output_dim (int): Dimension of the output regression value.
        """
        super(VideoRegressionModel, self).__init__()
        self.num_frames = num_frames
        self.num_channels = num_channels

        self.encoder = Encoder(num_channels)
        self.decoder = Decoder(num_channels)

        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
                                          dim_feedforward=mlp_dim, dropout=0.1, activation='relu', batch_first=True)

        self.feature_projection = nn.Linear(512, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the VideoRegressionModel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, num_channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the regression output tensor of shape (batch_size, output_dim)
                                               and the reconstructed video tensor of shape (batch_size, num_frames, num_channels, height, width).
        """
        features = []
        reconstructed_frames = []
        for i in range(self.num_frames):
            frame = x[:, i]
            frame_features = self.encoder(frame)
            
            # Project features to match transformer input dimension
            projected_features = self.feature_projection(frame_features)
            
            reconstructed_frame = self.decoder(frame_features, frame.size())
            features.append(projected_features)
            reconstructed_frames.append(reconstructed_frame)
        
        features = torch.stack(features, dim=1)
        reconstructed_video = torch.stack(reconstructed_frames, dim=1)

        outputs = self.transformer(features, features)
        outputs = outputs.mean(dim=1) 
        regression_output = self.fc(outputs)
        
        return regression_output, reconstructed_video


# # Example usage
# model = VideoRegressionModel(num_frames=24, num_channels=1, hidden_dim=512, 
#                              num_layers=4, num_heads=8, mlp_dim=2048, output_dim=1)

# # Test with a sample input
# input_data = torch.randn(2, 24, 1, 250, 400)  # Batch size 2, 24 frames, 3 channels, 224x224 resolution
# target = torch.randn(2, 1)  # Batch size 2, 1 output value

# output, reconstructed_video = model(input_data)
# print(f"Input shape: {input_data.shape}")
# print(f"Output shape: {output.shape}")
# print(f"Reconstructed video shape: {reconstructed_video.shape}")
