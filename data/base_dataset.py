import os
import torch
import cv2
from torchvision import transforms


def create_frames_sequence_dataset(
        low_light_path,
        processed_path,
        frame_sequence_length,
        resize=None,
        dtype=None
):
    if resize is None:
        resize = [960, 512]
    if dtype is None:
        dtype = "float32"
    read_and_preprocess_img = transforms.Compose([
        cv2.imread,
        torch.tensor,
        lambda t: t.transpose(2, 0),
        transforms.Resize(resize),
        lambda t: (t / 255).to(dtype)
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
                    [read_and_preprocess_img(processed_path + f"{first_frame + frame_sequence_length // 2}.jpg"),
                    *[read_and_preprocess_img(low_light_path + f"{j}.jpg")
                     for j in range(first_frame, first_frame + frame_sequence_length)]], dim=0
                )
            ]
            y_dataset += [
                torch.cat([read_and_preprocess_img(processed_path + f"{j}.jpg")
                for j in range(first_frame, first_frame + frame_sequence_length)], dim=0)
            ]
            skip = frame_sequence_length

    return x_dataset, y_dataset


def create_frames_pair_dataset(
        low_light_path,
        processed_path,
        frame_sequence_length,
        resize=None,
        dtype: torch.dtype=None
):
    if resize is None:
        resize = [960, 512]
    if dtype is None:
        dtype = "float32"
    read_and_preprocess_img = transforms.Compose([
        cv2.imread,
        torch.tensor,
        lambda t: t.transpose(2, 0),
        transforms.Resize(resize),
        lambda t: (t / 255).to(dtype)
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
            center_y_frame = read_and_preprocess_img(processed_path + x_path)
            x_dataset += [
                torch.cat((center_y_frame, read_and_preprocess_img(low_light_path + f"{j}.jpg")), dim=0)
                for j in range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
            ]
            y_dataset += [
                read_and_preprocess_img(processed_path + f"{j}.jpg")
                for j in range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
            ]
            skip = frame_sequence_length

    return x_dataset, y_dataset


def create_global_pair_dataset(
        low_light_path,
        processed_path,
        frame_sequence_length,
        transform_size: list[int]=None,
        dtype: torch.dtype=None
):
    x_dataset = []
    y_dataset = []

    if dtype is None:
        dtype = torch.float16

    total_memory = 0
    for path_x, path_y in zip(os.listdir(low_light_path), os.listdir(processed_path)):
        x, y = create_frames_pair_dataset(
            f"{low_light_path}/{path_x}/",
            f"{processed_path}/{path_y}/",
            frame_sequence_length,
            resize=transform_size,
            dtype=dtype,
        )
        x_dataset += x
        y_dataset += y
        print("Processed folder %s: %d frame pairs. Used memory: %.2f Mb" %
              (path_x, len(x), memory := ((x[0].element_size() + y[0].element_size()) * x[0].nelement() * len(x)) / 1024**2))
        total_memory += memory
    print("Processed %d folders: %d frame pairs. Total memory: %.2f Mb" %
          (len(os.listdir(low_light_path)), len(x_dataset), total_memory))
    return x_dataset, y_dataset


def create_global_sequence_dataset(
        low_light_path,
        processed_path,
        frame_sequence_length,
        transform_size: list[int]=None,
        dtype: torch.dtype=None
):
    x_dataset = []
    y_dataset = []

    if dtype is None:
        dtype = torch.float16

    total_memory = 0
    for path_x, path_y in zip(os.listdir(low_light_path), os.listdir(processed_path)):
        x, y = create_frames_sequence_dataset(
            f"{low_light_path}/{path_x}/",
            f"{processed_path}/{path_y}/",
            frame_sequence_length,
            resize=transform_size,
            dtype=dtype,
        )
        x_dataset += x
        y_dataset += y
        print("Processed folder %s: %d frame sequences. Used memory: %.2f Mb" %
              (path_x, len(x), memory := ((x[0].element_size() + y[0].element_size()) * x[0].nelement() * len(x)) / 1024**2))
        total_memory += memory
    print("Processed %d folders: %d frame sequences. Total memory: %.2f Mb" %
          (len(os.listdir(low_light_path)), len(x_dataset), total_memory))
    return x_dataset, y_dataset


if __name__ == "__main__":
    x, y = create_frames_pair_dataset("frames/low/0/", "frames/processed_enlightenGAN/0/", 5)
    x = torch.stack(x)
    y = torch.stack(y)

    print(x.shape, y.shape)
    print(x.element_size() * x.nelement() / 1024**2, y.element_size() * y.nelement() / 1024**2)
