import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from data.base_dataset import create_global_pair_dataset
from train_nn import train_nn
from models.frame_pair_model import Net1

parser = argparse.ArgumentParser(description="Training model to color correcting videos")
parser.add_argument("low_light_frames_path", type=str)
parser.add_argument("processed_low_light_frames_path", type=str)
parser.add_argument("frames_sequence_length", type=int)
parser.add_argument("--normal_light_frames-path", default="", type=str)
parser.add_argument("-batch", "--batch_size", default=8, type=int)
parser.add_argument("-epoch", "--num_epochs", default=100, type=int)
parser.add_argument("-sf", "--epoch_save_frequency", default=10, type=int)
parser.add_argument("-save", "--filename_to_save", required=True, type=str)
parser.add_argument("-lr", "--learning_rate", default=.001, type=float)
parser.add_argument("-m", "--model", default="pairframe", type=str)
parser.add_argument("-resize", "--resize_shape", nargs=2, default=[600, 400], type=int)
parser.add_argument("-dtype", "--tensor_dtype", default="float16", type=str)
args = parser.parse_args()

print(args)
dtypes = {
    "float32": torch.float32,
    "float16": torch.float16,
    "double": torch.double,
    "float": torch.float,
    "half": torch.float16,
}
x, y = create_global_pair_dataset(
    args.low_light_frames_path,
    args.processed_low_light_frames_path,
    args.frames_sequence_length,
    args.resize_shape,
    dtypes[args.tensor_dtype]
)

dataset = TensorDataset(torch.stack(x), torch.stack(y))
train_dataset, test_dataset = random_split(dataset, [(train_length := int(len(dataset) * .8)), len(dataset) - train_length])
data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
net = Net1()
if torch.cuda.is_available():
    net = net.cuda()
net = net.to(dtypes[args.tensor_dtype])
hist = train_nn(
    net,
    data_loader,
    test_data_loader,
    args.learning_rate,
    args.num_epochs,
    filename_to_save=args.filename_to_save,
    epoch_frequency_save=args.epoch_save_frequency
)
