import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def resnet34_(in_channels, n_classes, dropout=0.5, weights=None):
    model = models.resnet34(weights=weights)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(
                            nn.BatchNorm1d(512),
                            nn.Dropout(p=dropout, inplace=False),
                            nn.Linear(in_features = 512, out_features=64, bias=False),
                            nn.ReLU(inplace=True),
                            
                            nn.BatchNorm1d(64),
                            nn.Dropout(p=dropout, inplace=False),
                            nn.Linear(in_features=64, out_features=n_classes, bias=True)
                            )
    return model


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
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

    def forward(self, x):
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
    def __init__(self, num_channels=3):
        super(Encoder, self).__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.encoder.fc = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, num_channels=3):
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

    def forward(self, feature_vectors, target_size):
        batch_size = feature_vectors.size(0)
        h = self.decoder_linear(feature_vectors).view(batch_size, 512, 7, 7)
        h = self.decoder(h)
        
        # Final interpolation to match the target size
        h = F.interpolate(h, size=(target_size[2], target_size[3]), mode='bilinear', align_corners=False)
        return h


class VideoRegressionModel(nn.Module):
    def __init__(self, num_frames, num_channels, hidden_dim, num_layers, num_heads, mlp_dim, output_dim):
        super(VideoRegressionModel, self).__init__()
        self.num_frames = num_frames
        self.num_channels = num_channels

        self.encoder = Encoder(num_channels)
        self.decoder = Decoder(num_channels)

        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
                                          dim_feedforward=mlp_dim, dropout=0.1, activation='relu', batch_first=True)

        self.feature_projection = nn.Linear(512, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
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
