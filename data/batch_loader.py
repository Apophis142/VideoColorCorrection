import torch
import os


def load_batch_pair_frames(
        batch_paths: list[str],
        img_load: callable(str),
):
    x_batch, y_batch = [], []
    for paths in batch_paths:
        y_batch += [img_load(paths[2])]
        x_batch += [torch.cat([img_load(paths[1]), img_load(paths[0])], dim=0)]

    return torch.stack(x_batch), torch.stack(y_batch)


def preload_all_pair_frames_paths(
        path_low: str,
        path_normal: str,
        frame_sequence_length: int,
) -> list[tuple[str]]:
    res = []
    for (directory_x, _, files_x), (directory_y, _, files_y)  in zip(list(os.walk(path_low)),
                                                                     list(os.walk(path_normal))):
        files_x = sorted(files_x, key=lambda s: int(s.replace(".jpg", "")))

        skip = 0
        for k, filename in enumerate(files_x):
            if skip:
                skip -= 1
                continue

            center = int(filename.replace(".jpg", ""))
            if all(
                    f"{j}.jpg" in files_x and f"{j}.jpg" in files_y
                    for j in range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
            ):
                res += [
                    (directory_x + '/' + f"{j}.jpg", directory_y + '/' + f"{center}.jpg", directory_y + '/' + f"{j}.jpg")
                    for j in range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
                ]
                skip = frame_sequence_length

    return res


def load_batch_sequence_frames(
        batch_paths: list[str],
        img_load: callable(str),
):
    x_batch, y_batch = [], []
    for paths in batch_paths:
        y_batch += [torch.cat([img_load(path) for path in paths[1]])]
        x_batch += [torch.cat([img_load(path)for path in paths[0]], dim=0)]

    return torch.stack(x_batch), torch.stack(y_batch)


def preload_all_sequence_frames_paths(
        path_low: str,
        path_normal: str,
        frame_sequence_length: int,
) -> list[tuple[tuple[str]]]:
    res = []
    for (directory_x, _, files_x), (directory_y, _, files_y) in zip(list(os.walk(path_low)),
                                                                    list(os.walk(path_normal))):
        files_x = sorted(files_x, key=lambda s: int(s.replace(".jpg", "")))

        skip = 0
        for k, filename in enumerate(files_x):
            if skip:
                skip -= 1
                continue

            center = int(filename.replace(".jpg", ""))
            if all(
                    f"{j}.jpg" in files_x and f"{j}.jpg" in files_y
                    for j in range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
            ):
                res += [
                    (
                        (directory_y + '/' + f"{center}.jpg", *[
                            directory_x + '/' + f"{j}.jpg" for j in
                            range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
                        ]),
                        tuple(
                            directory_y + '/' + f"{j}.jpg" for j in
                            range(center - frame_sequence_length // 2, center + frame_sequence_length // 2 + 1)
                        )
                    )
                ]
                skip = frame_sequence_length

    return res
