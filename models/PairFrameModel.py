import torch
from torch import nn
import os
from models.EnligthenGAN import EnlightenOnnxModel
from models.RetinexNet import RetinexNet
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
            self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        else:
            self.load_state_dict(torch.load(path, map_location=device))
        self.eval()


class FramePairModel(object):
    def __init__(self, path_to_weights, center_model, device="cpu", dtype=torch.float32):
        self.net = FrameModel()
        self.net.load(path_to_weights, device)
        self.net = self.net.to(dtype)
        self.device = device
        self.dtype = dtype

        if center_model == "EnlightenGAN":
            self.center_model = EnlightenOnnxModel()
            self.net = self.net.to(self.device)

            def process_center(_self, _model, _tensor):
                if len(_tensor.shape) == 3:
                    _tensor = _tensor.unsqueeze(0)
                _tensor = _tensor.to(torch.float).numpy()
                res = []
                for k in range(_tensor.shape[0]):
                    res.append(_model.predict(_tensor[k:k + 1, :, :, :]))
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

    # def __call__(self, x_c, x_i, is_preprocessed=False):
    #     """(x_c, x_i): a pair of frames"""
    #     if len(x_c.shape) == 3:
    #         x_c = x_c.unsqueeze(0)
    #         x_i = x_i.unsqueeze(0)
    #     if not is_preprocessed:
    #         x_c = self.process_center(x_c)
    #     return self.net(torch.cat([x_c, x_i], dim=1)).squeeze()

    def __call__(self, xs, xs_center):
        processed_center = self.process_center(self, self.center_model, xs_center)

        xs = xs.view(xs.shape[0], -1, 3, *xs.shape[-2:]).to(self.dtype)
        res = self.process_frames(self, self.net, torch.stack([
            torch.cat([processed_center[i, :, :, :], xs[i, j, :, :, :]], dim=0)
            for i in range(xs.shape[0])
            for j in range(xs.shape[1])
        ], dim=0))

        torch.cuda.empty_cache()

        return res
