import torch
from torch import nn, optim
from tqdm import tqdm
import pickle


def train_nn(
        net: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        x_eval: torch.tensor,
        y_eval: torch.tensor,
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
            with torch.no_grad():
                x_eval, y_eval = torch.tensor(x_eval, dtype=torch.float32), torch.tensor(y_eval)
                if torch.cuda.is_available():
                    x_eval = x_eval.cuda()
                    y_eval = y_eval.cuda()
                eval_hist.append(loss_function(net(x_eval), y_eval).item())

            if epoch % epoch_frequency_save == 0 and epoch:
                net.save("C:\\Users\\User\\PycharmProjects\\vkr1\\weights\\", f"{net.name()}_epoch_{epoch}")
                with open(f"{net.name()}_loss_history_epoch_{epoch}", 'wb+') as f:
                    pickle.dump((tuple(loss_hist), tuple(eval_hist)), f)
            torch.cuda.empty_cache()
        pbar.close()

        net.save("C:\\Users\\User\\PycharmProjects\\vkr1\\weights\\", f"{net.name()}_trained")
        with open(f"{net.name()}_loss_history_epoch_{epoch}", 'wb+') as f:
            pickle.dump((tuple(loss_hist), tuple(eval_hist)), f)

    return loss_hist, eval_hist
