import torch
from torch import nn, optim
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
import os


if not os.path.exists("models/training"):
    os.makedirs("models/training")
if not os.path.exists("models/pair_frame"):
    os.makedirs("models/pair_frame")


def train_nn(
        net: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        learning_rate: float,
        num_epochs: int,
        loss_function: nn.Module=None,
        epoch_frequency_save: int=10,
):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_hist = []
    eval_hist = []
    if loss_function is None:
        loss_function = nn.MSELoss()

    with tqdm(total=len(train_loader) * num_epochs, position=0, leave=True) as pbar:

        for epoch in range(1, num_epochs + 1):
            running_loss, n = 0.0, 0

            net.train()
            for batch_num, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                outputs = net(inputs)
                loss = loss_function(outputs, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n += 1

                if batch_num % 100 == 0:
                    pbar.set_description("Epoch: %d, Batch: %d, Loss: %.2f" % (epoch, batch_num, running_loss / n))
                pbar.update()
            loss_hist.append(running_loss / len(train_loader))
            net.eval()
            eval_loss = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    if torch.cuda.is_available():
                        x_eval = inputs.cuda()
                        y_eval = targets.cuda()
                    eval_loss += loss_function(net(x_eval), y_eval).item()
                eval_hist.append(eval_loss / len(test_loader))

            if epoch % epoch_frequency_save == 0 and epoch:
                net.save("C:\\Users\\User\\PycharmProjects\\vkr1\\training\\weights\\", f"{net.name()}_epoch_{epoch}")
                with open(f"models\\training\\{net.name()}_loss_history_epoch_{epoch}", 'wb+') as f:
                    pickle.dump((tuple(loss_hist), tuple(eval_hist)), f)
                plt.plot(loss_hist)
                plt.xlabel("epoch")
                plt.title("training loss")
                plt.savefig(fname=f"models/training/{net.name()}training_loss.png")
                plt.plot(eval_hist)
                plt.xlabel("epoch")
                plt.title("validation loss")
                plt.savefig(fname=f"models/pair_frame/{net.name()}validation_loss.png")
            torch.cuda.empty_cache()
        pbar.close()

        net.save("C:\\Users\\User\\PycharmProjects\\vkr1\\weights\\", f"{net.name()}_trained")
        with open(f"{net.name()}_loss_history_epoch_{epoch}", 'wb+') as f:
            pickle.dump((tuple(loss_hist), tuple(eval_hist)), f)
        plt.plot(loss_hist)
        plt.xlabel("epoch")
        plt.title("training loss")
        plt.savefig(fname=f"models/pair_frame/{net.name()}training_loss.png")
        plt.plot(eval_hist)
        plt.xlabel("epoch")
        plt.title("validation loss")
        plt.savefig(fname=f"models/pair_frame/{net.name()}validation_loss.png")

    return loss_hist, eval_hist
