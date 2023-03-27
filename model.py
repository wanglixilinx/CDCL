import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import icnr_init
from .sesr import SESR

class ConvBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size=3, with_bn=False):
        layers = [nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size//2, bias=(not with_bn))]
        if with_bn: layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.ReLU())

        nn.init.kaiming_uniform_(layers[0].weight)
        if not with_bn:
            nn.init.normal_(layers[0].bias, mean=0.0, std=0.01)

        super().__init__(*layers)

class ConvNextBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super().__init__()
        assert input_channels==output_channels, "For ConvNextBlock input_channels has to be equal to output_channels."
        self.dwpw = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size//2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, output_channels, kernel_size=1, bias=True),
            nn.GELU()
        )
        self.dwpw_ib = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size, padding=kernel_size//2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, 2*output_channels, kernel_size=1, groups=2, bias=True),
            nn.GELU(),
            nn.Conv2d(2*output_channels, output_channels, kernel_size=1, groups=2, bias=True),
        )

    def forward(self, x):
        mid = x + self.dwpw(x)
        out = mid + self.dwpw_ib(mid)
        return out

class UNetTiny(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.encoder1 = nn.Sequential(
            ConvBlock(input_channels, 16),
            ConvBlock(16, 16),
        )
        self.encoder2 = nn.Sequential(
            ConvBlock(16, 32),
            ConvBlock(32, 32),
        )
        self.encoder3 = nn.Sequential(
            ConvBlock(32, 48),
            ConvBlock(48, 48),
        )
        self.bottleneck = nn.Sequential(
            ConvBlock(48, 64),
            ConvBlock(64, 64),
        )
        self.decoder3 = nn.Sequential(
            ConvBlock(64+48, 48, kernel_size=1),
            ConvBlock(48, 48)
        )
        self.decoder2 = nn.Sequential(
            ConvBlock(48+32, 32, kernel_size=1),
            ConvBlock(32, 32)
        )
        self.decoder1 = nn.Sequential(
            ConvBlock(32+16, 32, kernel_size=1),
            ConvBlock(32, 32)
        )
    
    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(F.max_pool2d(skip1, kernel_size=2))
        skip3 = self.encoder3(F.max_pool2d(skip2, kernel_size=2))
        up3 = self.bottleneck(F.max_pool2d(skip3, kernel_size=2))
        up2 = self.decoder3(torch.cat([F.interpolate(up3, scale_factor=2, mode='nearest'), skip3], dim=1))
        up1 = self.decoder2(torch.cat([F.interpolate(up2, scale_factor=2, mode='nearest'), skip2], dim=1))
        out = self.decoder1(torch.cat([F.interpolate(up1, scale_factor=2, mode='nearest'), skip1], dim=1))
        return out

class UNetBaseline(nn.Module): # ~25k MACs/pixel
    def __init__(self, input_channels):
        super().__init__()
        self.input_conv = nn.Conv2d(input_channels, 16, kernel_size=1)
        self.encoder1 = nn.Sequential(
            ConvBlock(16, 16),
            ConvBlock(16, 16),
        )
        self.encoder2 = nn.Sequential(
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
        )
        self.bottleneck = nn.Sequential(
            ConvBlock(32, 48),
            ConvBlock(48, 48),
            ConvBlock(48, 48),
        )
        self.decoder2 = nn.Sequential(
            ConvBlock(48+32, 32, kernel_size=1),
            ConvBlock(32, 32),
            ConvBlock(32, 32)
        )
        self.decoder1 = nn.Sequential(
            ConvBlock(32+16, 16, kernel_size=1),
            ConvBlock(16, 16),
            ConvBlock(16, 16)
        )
    
    def forward(self, x):
        x = self.input_conv(x)
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(F.max_pool2d(skip1, kernel_size=2))
        up2 = self.bottleneck(F.max_pool2d(skip2, kernel_size=2))
        up1 = self.decoder2(torch.cat([F.interpolate(up2, scale_factor=2, mode='nearest'), skip2], dim=1))
        out = self.decoder1(torch.cat([F.interpolate(up1, scale_factor=2, mode='nearest'), skip1], dim=1))
        return out

class FilterNetwork(nn.Module):
    def __init__(self, input_channels, recurrent_channels):
        super().__init__()
        self.recurrent_channels = recurrent_channels
        self.conv = nn.Conv2d(input_channels, 10 + recurrent_channels, kernel_size=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, features, sr_color, history_color):
        filters, recurrent = self.conv(features).split([10, self.recurrent_channels], dim=1)

        filters = F.softmax(filters, dim=1)
        recurrent = torch.sigmoid(recurrent) # We use sigmoid for recurrent

        def apply_filter(img, filter):
            N,C,H,W = img.shape
            KS = int(filter.size(1)**0.5)
            assert KS**2 == filter.size(1) and KS%2 == 1, f"Wrong filtering kernel shape: {filter.shape}"

            # unfold (N,C,H,W) tensor into (N,C*KS*KS,H*W) patches
            img_unfolded = F.unfold(img, (KS, KS), padding=KS//2)
            img_patches = img_unfolded.reshape(N,C,KS*KS,H,W)

            img_filtered = (img_patches * filter.unsqueeze(1)).sum(dim=2)
            return img_filtered

        current_filter, history_filter = filters.split([9,1], dim=1)

        sr_color_filtered = apply_filter(sr_color, current_filter)
        history_color_filtered = history_color * history_filter

        return sr_color_filtered + history_color_filtered, recurrent

class ColorPredictionNetwork(nn.Module):
    def __init__(self, input_channels, recurrent_channels):
        super().__init__()
        self.recurrent_channels = recurrent_channels
        self.conv = nn.Conv2d(input_channels, 3 + recurrent_channels, kernel_size=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, features, sr_color, history_color):
        # sr_color and history_color are not used but they're kept for FilterNetwork compatibility
        color, recurrent = self.conv(features).split([3, self.recurrent_channels], dim=1)

        color = torch.relu(color)
        recurrent = torch.relu(recurrent) # We use sigmoid for recurrent

        return color, recurrent

class AMDNetTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_network = UNetTiny(16)
        self.prediction_network = FilterNetwork(8, recurrent_channels=1)

        # INCR init beacuase of pixel shuffle
        self.feature_network.decoder1[1][0].weight.data.copy_(icnr_init(self.feature_network.decoder1[1][0].weight.data)) 

    def forward(self, x, sr_color, prev_color):
        input_unshuffled = F.pixel_unshuffle(x, downscale_factor=2)
        features = self.feature_network(input_unshuffled)
        features_shuffled = F.pixel_shuffle(features, upscale_factor=2)
        
        final_color, recurrent_state = self.prediction_network(features_shuffled, sr_color, prev_color)

        return final_color, recurrent_state

class KPNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_network = SESR(in_channels=11, return_outs=16)
        self.prediction_network = FilterNetwork(16, recurrent_channels=8)

    def forward(self, x, sr_color, prev_color):
        features = self.feature_network(x)
        
        final_color, recurrent_state = self.prediction_network(features, sr_color, prev_color)

        return final_color, recurrent_state

class CPNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_network = SESR(in_channels=11, return_outs=16)
        self.prediction_network = ColorPredictionNetwork(16, recurrent_channels=12)

    def forward(self, x, sr_color, prev_color):
        features = self.feature_network(x)
        
        final_color, recurrent_state = self.prediction_network(features, sr_color, prev_color)

        return final_color, recurrent_state
