import threading
from random import sample
import time

import cv2
import torch
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
import os

from data.batch_loader import load_batch_sequence_frames, load_batch_pair_frames

if not os.path.exists("models/training"):
    os.makedirs("models/training")
if not os.path.exists("models/pair_frame"):
    os.makedirs("models/pair_frame")


flag_preloaded_next_batch = False
trained_flag = False
next_batch_paths: list = None
preloaded_next_batch: torch.tensor
global_break: bool = False


def train_nn(
        net: nn.Module,
        train_paths: list[str],
        test_paths: list[str],
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        device: torch.device,
        filename_to_save: str,
        loss_function: nn.Module=None,
        epoch_frequency_save: int=10,
        resize: list[int]=None,
        dtype: str=None,
):
    global global_break

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if loss_function is None:
        loss_function = nn.MSELoss()
    lock = threading.Lock()
    if resize is None:
        resize = [600, 400]
    if dtype is None:
        dtype = torch.float
    read_and_preprocess_img = transforms.Compose([
        cv2.imread,
        torch.tensor,
        lambda t: t.transpose(2, 0),
        transforms.Resize(resize),
        lambda t: (t / 255).to(dtype)
    ])

    batch_loader = {
        "SequenceFrameModel": load_batch_sequence_frames,
        "PairFrameModel": load_batch_pair_frames,
    }[net.name()]
    data_loading_thread = threading.Thread(target=load_batch_thread, args=(
        lambda paths: batch_loader(paths, read_and_preprocess_img),
        lock
    ))
    data_loading_thread.start()
    model_training_thread = threading.Thread(target=training_thread, args=(
        net,
        optimizer,
        loss_function,
        train_paths,
        test_paths,
        num_epochs,
        epoch_frequency_save,
        batch_size,
        filename_to_save,
        device,
        lock,
    ))

    model_training_thread.start()

    while True:
        with open("emergency_stop.txt", 'r+') as f:
            if f.readline().strip() == "stop":
                global_break = True
                f.truncate()
                break
        time.sleep(10)
    # data_loading_thread.join()
    # model_training_thread.join()



def load_batch_thread(batch_loader, lock):
    global flag_preloaded_next_batch, next_batch_paths, preloaded_next_batch
    global trained_flag, global_break
    while True:
        if not flag_preloaded_next_batch and next_batch_paths is not None:
            lock.acquire()
            if next_batch_paths:
                preloaded_next_batch = batch_loader(next_batch_paths)
            flag_preloaded_next_batch = True
            next_batch_paths = None
            lock.release()
        if trained_flag or global_break:
            break


def training_thread(
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.modules.loss,
        train_data_paths: list,
        test_data_paths: list,
        num_epochs: int,
        epoch_frequency_save: int,
        batch_size: int,
        filename_to_save: str,
        device,
        lock
):
    global next_batch_paths, flag_preloaded_next_batch, trained_flag, preloaded_next_batch, global_break

    num_train_batches = (len(train_data_paths) + batch_size - 1) // batch_size
    num_test_batches = (len(test_data_paths) + batch_size - 1) // batch_size

    train_hist, eval_hist = [], []
    with tqdm(total=(num_train_batches + num_test_batches) * num_epochs, position=0, leave=True) as pbar:

        for epoch in range(1, num_epochs + 1):
            running_loss, n = 0., 0

            batches = sample(train_data_paths, k=len(train_data_paths))
            net.train()
            lock.acquire()
            next_batch_paths = batches[:batch_size]
            flag_preloaded_next_batch = False
            lock.release()
            for batch_num in range(1, num_train_batches + 1):
                optimizer.zero_grad()

                while not flag_preloaded_next_batch:
                    continue

                lock.acquire()
                inputs = preloaded_next_batch[0].to(device)
                targets = preloaded_next_batch[1].to(device)

                next_batch_paths = batches[batch_size * batch_num:batch_size * (batch_num + 1)]
                flag_preloaded_next_batch = False

                lock.release()

                outputs = net(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                pbar.set_description("Epoch: %d (training), Batch: %d, Loss: train %.4f, eval %.4f" %
                                     (
                                         epoch, batch_num, running_loss / batch_num,
                                         eval_hist[-1] if eval_hist else torch.nan
                                     ))
                pbar.update()
                torch.cuda.empty_cache()
            train_hist += [running_loss / num_train_batches]

            net.eval()
            eval_loss = 0.
            lock.acquire()
            next_batch_paths = test_data_paths[:batch_size]
            flag_preloaded_next_batch = False
            lock.release()
            for batch_num in range(1, num_test_batches + 1):

                while not flag_preloaded_next_batch:
                    continue

                lock.acquire()
                inputs = preloaded_next_batch[0].to(device)
                targets = preloaded_next_batch[1].to(device)

                next_batch_paths = batches[batch_size * batch_num:batch_size * (batch_num + 1)]
                flag_preloaded_next_batch = False
                lock.release()

                outputs = net(inputs)
                eval_loss += loss_function(outputs, targets).item()

                pbar.set_description("Epoch: %d (validation), Batch: %d, Loss: train %.4f, eval %.4f" %
                                     (epoch, batch_num, train_hist[-1], eval_loss / batch_num))
                pbar.update()
                torch.cuda.empty_cache()

            eval_hist += [eval_loss / num_test_batches]

            if epoch % epoch_frequency_save == 0 and epoch:
                net.save("weights/training/", f"{filename_to_save}_epoch_{epoch}")
                with open(f"models/training/{filename_to_save}_loss_history.pkl", 'wb+') as f:
                    pickle.dump((tuple(train_hist), tuple(eval_hist)), f)

            if global_break:
                break


    trained_flag = True
    net.save("weights/", f"{filename_to_save}_trained")
    with open(f"{filename_to_save}_loss_history_epoch_{epoch}.pkl", 'wb+') as f:
        pickle.dump((tuple(train_hist), tuple(eval_hist)), f)
    plt.plot(train_hist, label="training")
    plt.xlabel("epoch")
    plt.plot(eval_hist, label="validation")
    plt.xlabel("epoch")
    plt.title("Loss")
    plt.legend()
    plt.savefig(fname=f"models/pair_frame/{filename_to_save}_losses.png")

    return train_hist, eval_hist
