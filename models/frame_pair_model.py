import torch
from torch import nn
import os


class Net1(nn.Module):
    """
    (Frame_i, Processed_frame_center) -> Processed_frame_i
    """

    def name(self):
        return "PairFrameModel"

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.net(x)

    def save(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), path + filename)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()


class FramePairModel(object):
    def __init__(self, process_center, process_second):
        self.center = process_center
        self.second = process_second

    def __call__(self, x):
        """x: sequence of frames"""
        pass
