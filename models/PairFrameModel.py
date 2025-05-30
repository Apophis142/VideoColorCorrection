import torch
from torch import nn
import os
from enlighten_inference import EnlightenOnnxModel
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
        if center_model == "EnlightenGAN":
            self.center_model = EnlightenOnnxModel()
            self.process_center = transforms.Compose([
                lambda t: t.transpose(2, 0).detach().cpu().numpy() * 255,
                self.center_model.predict,
                torch.tensor,
                lambda t: t.transpose(2, 0) / 255
            ])
        elif center_model == "RetinexNet":
            self.center_model = RetinexNet("models/weights/RetinexNet/").to(dtype)
            self.process_center = transforms.Compose([
                lambda t: self.center_model.predict,
            ])
        self.net.load(path_to_weights, "cpu")
        self.net = self.net.to(dtype)
        self.device = device
        self.dtype = dtype

    # def __call__(self, x_c, x_i, is_preprocessed=False):
    #     """(x_c, x_i): a pair of frames"""
    #     if len(x_c.shape) == 3:
    #         x_c = x_c.unsqueeze(0)
    #         x_i = x_i.unsqueeze(0)
    #     if not is_preprocessed:
    #         x_c = self.process_center(x_c)
    #     return self.net(torch.cat([x_c, x_i], dim=1)).squeeze()

    def __call__(self, xs, xs_center):
        model = self.center_model.to(self.device)
        xs_center = xs_center.to(self.dtype).to(self.device)
        processed_center = model.predict(xs_center).clip(0., 1.).cpu()
        del model, xs_center

        xs = xs.view(xs.shape[0], -1, 3, *xs.shape[-2:]).to(self.dtype)
        net = self.net.to(self.device)
        res = net(torch.stack([
            torch.cat([processed_center[i, :, :, :], xs[i, j, :, :, :]], dim=0)
            for i in range(xs.shape[0])
            for j in range(xs.shape[1])
        ], dim=0).to(self.device)).detach().cpu()
        del net

        torch.cuda.empty_cache()

        return res

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)
