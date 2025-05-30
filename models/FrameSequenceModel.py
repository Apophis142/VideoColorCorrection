import torch
from torch import nn
import os
import numpy as np
from onnxruntime import InferenceSession
from models.RetinexNet import RetinexNet
from torchvision import transforms


class EnlightenOnnxModel:
    def __init__(self):
        self.graph = InferenceSession('./models/weights/EnlightenGAN/enlighten.onnx',
                                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def __repr__(self):
        return f'<EnlightenGAN OnnxModel {id(self)}>'

    def predict(self, batch):
        image_numpy, = self.graph.run(['output'], {'input': batch.numpy()})
        image_numpy = (image_numpy + 1) / 2.0
        image_numpy = np.clip(image_numpy, 0., 1.)
        return torch.tensor(image_numpy)



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
            self.load_state_dict(torch.load(path, map_location=device))
        self.eval()


class SequenceFrameModel(object):
    def __init__(self, frame_sequence_length, path_to_weights, center_model, device, dtype=torch.float32):
        self.frame_sequence_length = frame_sequence_length
        self.net = SequenceModel(frame_sequence_length)
        if center_model == "EnlightenGAN":
            self.center_model = EnlightenOnnxModel()
            self.process_center = transforms.Compose([
                self.center_model.predict,
            ])
        elif center_model == "RetinexNet":
            self.center_model = RetinexNet("models/weights/RetinexNet/").to(dtype)
            self.process_center = transforms.Compose([
                self.center_model.predict,
            ])
        self.net.load(path_to_weights, "cpu")
        self.net = self.net.to(dtype)
        self.dtype = dtype
        self.device = device

    # def __call__(self, xs):
    #     """xs: sequence of frames"""
    #     if len(xs.shape) == 3:
    #         xs = xs.unsqueeze(0)
    #     return self.net(torch.cat([
    #         self.process_center(xs[:, :3, :, :]), xs[:, 3:, :, :]
    #     ], dim=1)).squeeze()

    def __call__(self, xs, xs_center):
        model = self.center_model.to(self.device)
        xs_center = xs_center.to(self.dtype).to(self.device)
        processed_center = model.predict(xs_center).clip(0., 1.).cpu()
        del model, xs_center

        net = self.net.to(self.device)
        xs = xs.to(self.dtype)
        res = net(torch.cat([processed_center, xs], dim=1).to(self.device)).detach()
        del net

        torch.cuda.empty_cache()

        return res

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)
