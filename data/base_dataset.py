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
        transform = transforms.Compose([torch.tensor])
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
                    [transform(cv2.imread(processed_path + f"{first_frame + frame_sequence_length // 2}.jpg")),
                    *[transform(cv2.imread(low_light_path + f"{j}.jpg"))
                     for j in range(first_frame, first_frame + frame_sequence_length)]], dim=2
                ).transpose(2, 0) / 255
            ]
            y_dataset += [
                torch.cat([transform(cv2.imread(processed_path + f"{j}.jpg"))
                for j in range(first_frame, first_frame + frame_sequence_length)], dim=2).transpose(2, 0) / 255
            ]
            skip = frame_sequence_length

    return x_dataset, y_dataset


def create_frames_pair_dataset(low_light_path, processed_path, frame_sequence_length, transform=None):
    if transform is None:
        transform = transforms.Compose([torch.tensor])
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
            center_y_frame = transform(cv2.imread(processed_path + x_path))
            x_dataset += [
                torch.cat((center_y_frame, transform(cv2.imread(low_light_path + f"{j}.jpg"))), dim=2).transpose(2, 0) / 255
                for j in range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
            ]
            y_dataset += [
                torch.tensor(transform(cv2.imread(processed_path + f"{j}.jpg"))).transpose(2, 0) / 255
                for j in range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
            ]
            skip = frame_sequence_length

    return x_dataset, y_dataset

def create_global_pair_dataset(low_light_path, processed_path, frame_sequence_length, transform_size: list[int]=None):
    x_dataset = []
    y_dataset = []

    if transform_size is None:
        transform = transforms.Compose([torch.tensor])
    else:
        transform = transforms.Compose([torch.tensor, transforms.Resize(transform_size)])
    for path_x, path_y in zip(os.listdir(low_light_path), os.listdir(processed_path)):
        x, y = create_frames_pair_dataset(
            f"{low_light_path}/{path_x}/",
            f"{processed_path}/{path_y}/",
            frame_sequence_length,
            transform=transform
        )
        x_dataset += x
        y_dataset += y

    return x_dataset, y_dataset


if __name__ == "__main__":
    print(create_frames_pair_dataset("frames/low/1/", "frames/processed_enlightenGAN/1/", 5))
