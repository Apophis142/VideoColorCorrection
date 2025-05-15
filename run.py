import argparse

import torch

from random import sample

parser = argparse.ArgumentParser(description="Training model to color correcting videos")
parser.add_argument("-m", "--model", default="pairframe", type=str)
parser.add_argument("low_light_frames_path", type=str)
parser.add_argument("processed_low_light_frames_path", type=str)
parser.add_argument("frames_sequence_length", type=int)
parser.add_argument("--normal_light_frames-path", default="", type=str)
parser.add_argument("-batch", "--batch_size", default=8, type=int)
parser.add_argument("-epoch", "--num_epochs", default=100, type=int)
parser.add_argument("-gpu", "--gpu_id", default=0, type=int)
parser.add_argument("-sf", "--epoch_save_frequency", default=10, type=int)
parser.add_argument("-save", "--filename_to_save", required=True, type=str)
parser.add_argument("-lr", "--learning_rate", default=.001, type=float)
parser.add_argument("-loss", "--loss_function", default="mae", type=str)
parser.add_argument("-resize", "--resize_shape", nargs=2, default=[600, 400], type=int)
parser.add_argument("-dtype", "--tensor_dtype", default="float16", type=str)
parser.add_argument("-multi", "--multi_threading_training", default="y", type=str)
args = parser.parse_args()

if args.multi_threading_training == "y":
    from train_nn import train_nn
elif args.multi_threading_training == "n":
    from one_thread_training import train_nn

print(args)
dtypes = {
    "float32": torch.float32,
    "float16": torch.float16,
    "double": torch.double,
    "float": torch.float,
    "half": torch.float16,
}
loss_functions = {
    "mse": torch.nn.MSELoss(),
    "mae": torch.nn.L1Loss(),
}
if args.model == "pairframe":
    from models.PairFrameModel import FrameModel
    from data.batch_loader import preload_all_pair_frames_paths as load_paths

    net = FrameModel()
elif args.model == "sequenceframe":
    from models.FrameSequenceModel import SequenceModel
    from data.batch_loader import preload_all_sequence_frames_paths as load_paths

    net = SequenceModel(frame_sequence_length=args.frames_sequence_length)
else:
    raise ValueError


dataset_paths = load_paths(
    args.low_light_frames_path,
    args.processed_low_light_frames_path,
    args.frames_sequence_length,
)
train_test_splitter = sample(dataset_paths, k=(ds_size := len(dataset_paths)))
train_dataset, test_dataset = train_test_splitter[:int(.8*ds_size)], train_test_splitter[int(.8*ds_size):]
if args.gpu_id == -1:
    device = torch.device("cpu")
elif args.gpu_id >= 0:
    device = torch.device("cuda:%d" % args.gpu_id)
else:
    raise Exception
net = net.to(dtypes[args.tensor_dtype])
net = net.to(device)
hist = train_nn(
    net,
    train_dataset,
    test_dataset,
    lr=args.learning_rate,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    device=device,
    loss_function=loss_functions[args.loss_function],
    filename_to_save=args.filename_to_save,
    epoch_frequency_save=args.epoch_save_frequency,
    resize=args.resize_shape,
    dtype=dtypes[args.tensor_dtype]
)
