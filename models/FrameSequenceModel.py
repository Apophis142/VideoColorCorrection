import torch
from torch import nn
import os
from enlighten_inference import EnlightenOnnxModel
from models.RetinexNet import RetinexNet
from torchvision import transforms


class SequenceModel(nn.Module):
    """
    (Processed_frame_center, Frame_i, ..., Frame_i+k-1) -> (Processed_frame_i, ..., Processed_frame_i+k-1)
    """

    def name(self):
        return "SequenceFrameModel"

    def __init__(self, frame_sequence_length):
        super().__init__()
        self.frame_sequence_length = frame_sequence_length
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3*(frame_sequence_length+1), out_channels=12*frame_sequence_length+3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=12*frame_sequence_length+3, out_channels=6*frame_sequence_length, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=6*frame_sequence_length, out_channels=3*frame_sequence_length, kernel_size=9, padding=4),
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
            self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        else:
            self.load_state_dict(torch.load(path))
        self.eval()


class SequenceFrameModel(object):
    def __init__(self, frame_sequence_length, path_to_weights, center_model):
        self.frame_sequence_length = frame_sequence_length
        if center_model == "EnlightenGAN":
            self.model = EnlightenOnnxModel()
            self.process_center = transforms.Compose([
                lambda t: t.transpose(2, 0).detach().cpu().numpy() * 255,
                self.model.predict,
                torch.tensor,
                lambda t: t.transpose(2, 0) / 255
            ])
        elif center_model == "RetinexNet":
            self.model = RetinexNet()
            self.predict_center = transforms.Compose([
                lambda t: self.model.predict("models/weights/RetinexNet/", t.numpy()),
            ])
        self.net = SequenceModel(frame_sequence_length)
        if torch.cuda.is_available():
            self.net.load(path_to_weights)
        else:
            self.net.load(path_to_weights, "cpu")

    def __call__(self, xs):
        """xs: sequence of frames"""
        return self.net(torch.cat([
            self.process_center(xs[:3, :, :]), xs[3:]
        ], dim=0))

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)
