import os
import torch
import cv2
from torchvision import transforms


class FramesSequence(object):
    def __init__(self, frames, predicted_frame):
        self.frames = [torch.tensor(frame) for frame in frames]
        self.length = len(frames)
        self.center_frame = predicted_frame

    def __getitem__(self, item):
        return self.frames[item]

    def to_tensor(self):
        return torch.cat([self.center_frame, *self.frames], dim=2)


class FramesPair(object):
    def __init__(self, second_frame, predicted_frame):
        self.frames = [predicted_frame, second_frame]
        self.length = 2
        self.center_frame = predicted_frame

    def __getitem__(self, item):
        return self.frames[item]

    def to_tensor(self):
        return torch.cat(self.frames, dim=2)


def create_frames_sequence_dataset(low_light_path, processed_path, frame_sequence_length, transform=None):
    if transform is None:
        transform = transforms.Compose([
            cv2.imread,
            torch.tensor,
            transforms.Resize([600, 400]),
            lambda t: (t / 255).to(torch.float16).transpose(0, 2)
        ])
    x_dataset = []
    y_dataset = []
    skip = 0
    xs = sorted( os.listdir(low_light_path), key=lambda x: int(x.replace(".jpg", "")))
    ys = os.listdir(processed_path)
    for x_path in xs:
        if skip:
            skip -= 1
            continue
        first_frame = int(x_path.replace(".jpg", ""))
        if all(f"{j}.jpg" in xs and f"{j}.jpg" in ys for j in range(first_frame, first_frame + frame_sequence_length)):
            x_dataset += [
                torch.cat(
                    [transform(processed_path + f"{first_frame + frame_sequence_length // 2}.jpg"),
                    *[transform(low_light_path + f"{j}.jpg")
                     for j in range(first_frame, first_frame + frame_sequence_length)]], dim=0
                )
            ]
            y_dataset += [
                torch.cat([transform(processed_path + f"{j}.jpg")
                for j in range(first_frame, first_frame + frame_sequence_length)], dim=0)
            ]
            skip = frame_sequence_length

    return x_dataset, y_dataset


def create_frames_pair_dataset(low_light_path, processed_path, frame_sequence_length, resize=None):
    if resize is None:
        resize = [960, 512]
    transform = transforms.Compose([
        cv2.imread,
        torch.tensor,
        lambda t: t.transpose(2, 0),
        transforms.Resize(resize),
        lambda t: (t / 255).to(torch.float16)
    ])
    x_dataset = []
    y_dataset = []
    skip = 0
    xs = sorted(os.listdir(low_light_path), key=lambda x: int(x.replace(".jpg", "")))
    ys = os.listdir(processed_path)
    for x_path in xs:
        if skip:
            skip -= 1
            continue
        center = int(x_path.replace(".jpg", ""))
        if all(f"{j}.jpg" in xs and f"{j}.jpg" in ys for j in range(center - frame_sequence_length // 2,center + frame_sequence_length // 2 + 1)):
            center_y_frame = transform(processed_path + x_path)
            x_dataset += [
                torch.cat((center_y_frame, transform(low_light_path + f"{j}.jpg")), dim=0)
                for j in range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
            ]
            y_dataset += [
                transform(processed_path + f"{j}.jpg")
                for j in range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
            ]
            skip = frame_sequence_length

    return x_dataset, y_dataset

def create_global_pair_dataset(low_light_path, processed_path, frame_sequence_length, transform_size: list[int]=None):
    x_dataset = []
    y_dataset = []

    for path_x, path_y in zip(os.listdir(low_light_path), os.listdir(processed_path)):
        x, y = create_frames_pair_dataset(
            f"{low_light_path}/{path_x}/",
            f"{processed_path}/{path_y}/",
            frame_sequence_length,
            resize=transform_size
        )
        x_dataset += x
        y_dataset += y

    return x_dataset, y_dataset


if __name__ == "__main__":
    x, y = create_frames_pair_dataset("frames/low/0/", "frames/processed_enlightenGAN/0/", 5)
    x = torch.stack(x)
    y = torch.stack(y)

    print(x.shape, y.shape)
    print(x.element_size() * x.nelement() / 1024**2, y.element_size() * y.nelement() / 1024**2)
