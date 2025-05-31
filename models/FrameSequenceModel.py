import torch
from torch import nn
import os
from models.EnligthenGAN import EnlightenOnnxModel
from models.RetinexNet import RetinexNet


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
        self.net.load(path_to_weights, "cpu")
        self.net = self.net.to(dtype)
        self.dtype = dtype
        self.device = device

        if center_model == "EnlightenGAN":
            self.center_model = EnlightenOnnxModel()
            self.net = self.net.to(self.device)
            def process_center(_self, _model, _tensor):
                if len(_tensor.shape) == 3:
                    _tensor = _tensor.unsqueeze(0)
                _tensor = _tensor.to(torch.float).numpy()
                res = []
                for k in range(_tensor.shape[0]):
                    res.append(_model.predict(_tensor[k:k+1, :, :, :]))
                return torch.cat(res, dim=0)

            def process_all_frames(_self, _model, _tensor):
                _tensor = _tensor.to(_self.dtype).to(_self.device)

                return _model(_tensor).detach().cpu()

        elif center_model == "RetinexNet":
            self.center_model = RetinexNet("models/weights/RetinexNet/").to(dtype)
            def process_center(_self, _model, _tensor):
                _tensor = _tensor.to(_self.dtype).to(_self.device)
                _model = _model.to(_self.device)
                return _model(_tensor).clip(0., 1.).cpu()

            def process_all_frames(_self, _model, _tensor):
                _model = _model.to(_self.device)
                _tensor = _tensor.to(_self.dtype).to(_self.device)

                return _model(_tensor).detach().cpu()

        else:
            raise ValueError("Unknown center model: %s" % center_model)

        self.process_center = process_center
        self.process_frames = process_all_frames


    # def __call__(self, xs):
    #     """xs: sequence of frames"""
    #     if len(xs.shape) == 3:
    #         xs = xs.unsqueeze(0)
    #     return self.net(torch.cat([
    #         self.process_center(xs[:, :3, :, :]), xs[:, 3:, :, :]
    #     ], dim=1)).squeeze()

    def __call__(self, xs, xs_center):
        processed_center = self.process_center(self, self.center_model, xs_center)
        res = self.process_frames(self, self.net, torch.cat([processed_center, xs], dim=1))

        torch.cuda.empty_cache()
        return res
