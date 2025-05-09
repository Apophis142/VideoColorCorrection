import torch
from torch import nn
import os
from enlighten_inference import EnlightenOnnxModel
from torchvision import transforms


class FrameModel(nn.Module):
    """
    (Frame_i, Processed_frame_center) -> Processed_frame_i
    """

    def name(self):
        return "PairFrameModel"

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=9, padding=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def save(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), path + filename + ".pth" if not filename.endswith(".pth") else "")

    def load(self, path, device="cuda"):
        if device == "cpu":
            self.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device("cpu")))
        else:
            self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()


class FramePairModel(object):
    def __init__(self, path_to_weights):
        self.model = EnlightenOnnxModel()
        self.net = FrameModel()
        if torch.cuda.is_available():
            self.net.load(path_to_weights)
        else:
            self.net.load(path_to_weights, "cpu")
        self.process_center = transforms.Compose([
            lambda t: t.transpose(2, 0).detach().cpu().numpy() * 255,
            self.model.predict,
            torch.tensor,
            lambda t: t.transpose(2, 0) / 255
        ])

    def __call__(self, x_c, x_i):
        """(x_c, x_i): a pair of frames"""
        return self.net(torch.cat([self.process_center(x_c), x_i], dim=0))

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)


if __name__ == "__main__":
    net = FrameModel()
    net.eval()
    x = torch.rand([960, 512, 6]).transpose(0, 2)
    print(x.element_size() * x.nelement() / 1024**2)

    print(net(x), net(x).shape)
