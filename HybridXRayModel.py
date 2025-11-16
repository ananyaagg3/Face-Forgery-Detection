import torch
import torch.nn as nn
import torch.nn.functional as F

from fft_features import fft_features
from FaceXRayHRNet import FaceXRayHRNet  # make sure the filename/class match exactly

class HybridXRayModel(nn.Module):
    def __init__(self):
        super(HybridXRayModel, self).__init__()
        # FaceXRayHRNet should return shape (B, 1, 1, 1) per your current implementation
        self.face_xray = FaceXRayHRNet(pretrained=True)

        # Fused feature size:
        # - FaceXRayHRNet: 1 value (1*1*1)
        # - FFT magnitude (collapsed to 1 channel): 256*256 = 65536
        # Total = 65536 + 1 = 65537
        self.fc = nn.Linear(65537, 2)

    def forward(self, x):
        spatial_xray = self.face_xray(x)          # expected (B, 1, 1, 1)
        fft_mag     = fft_features(x)             # (B, 1, 256, 256)

        fused = torch.cat(
            [spatial_xray.view(x.size(0), -1),
             fft_mag.view(x.size(0), -1)],
            dim=1
        )  # (B, 65537)

        out = self.fc(fused)                      # (B, 2)
        return out
