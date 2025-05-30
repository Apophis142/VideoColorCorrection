from onnxruntime import InferenceSession
import numpy as np
import torch


class EnlightenOnnxModel:
    def __init__(self):
        self.graph = InferenceSession('./models/weights/EnlightenGAN/enlighten.onnx',
                                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def __repr__(self):
        return f'<EnlightenGAN OnnxModel {id(self)}>'

    def predict(self, batch):
        image_numpy, = self.graph.run(['output'], {'input': batch})
        image_numpy = (image_numpy + 1) / 2.0
        image_numpy = np.clip(image_numpy, 0., 1.)
        return torch.tensor(image_numpy)
