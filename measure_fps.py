import cv2
import argparse
import torch
from torchvision import transforms
import numpy as np

import time
import threading


parser = argparse.ArgumentParser(description="Measuring FPS")

parser.add_argument('-m', '--model', type=str, required=True, choices=["pairframe", "sequenceframe"],
                    help="Type of model to use")
parser.add_argument('-fl', '--frames_sequence_length', type=int, required=True,
                    help="Length of frames sequence for model")
parser.add_argument('-w', '--weights', type=str, required=True,
                    help="Path to model's weights")
parser.add_argument('-cw', '-center', '--center-model', type=str, required=True,
                    choices=["EnlightenGAN", "RetinexNet"],
                    help="Model used to process center frame in sequence")
parser.add_argument('-vid', '--video-path', type=str, required=True,
                    help="Path to video for processing")
parser.add_argument('-skip', '-skip-frames', '--number-of-frames-to-skip', type=int, default=300,
                    help="Number of first video's frames to skip (default: 300)")
parser.add_argument('-check-frames', '--number-of-frames-to-check', type=int, default=600,
                    help="Number of frames to process (default: 600)")
parser.add_argument('--path-to-file', type=str, default='./metrics/FPS/',
                    help="Path to file to collect results")
parser.add_argument('-gpu','--gpu-id', type=int, default=0,
                    help="GPU's id to run the script on. Use -1 for CPU")
parser.add_argument('-b', '--batch-size', type=int, default=1,
                    help="Number of sequences the model processing at once (default: 1)")
parser.add_argument('-show', '--show-results', action=argparse.BooleanOptionalAction,
                    help="Use this argument to see the results after processing")
parser.add_argument('--verbose', action=argparse.BooleanOptionalAction,
                    help="Use this argument to see the progress during processing frames")
parser.add_argument('-save', '--save-results', type=str, default="",
                    help="Path to save processed video. Don't use this parameter to not save the result (default: '')")
parser.add_argument("-dtype", "--tensor-dtype", default="float32", type=str,
                    help="Frames and model's data type")

args = parser.parse_args()


print(args)
dtypes = {
    "float32": torch.float32,
    "float16": torch.float16,
    "double": torch.double,
    "float": torch.float,
    "half": torch.float16,
}


def batch_processing_thread(lock_preprocessing, lock_processing):
    global args, start_arr

    global batch_preprocessed_flag, batch_processed_flag
    global frame_counter
    global tensor_batch, tensor_batch_center, processed_batch

    while True:
        if batch_preprocessed_flag:
            lock_preprocessing.acquire()
            inp = (tensor_batch, tensor_batch_center)
            batch_preprocessed_flag = False
            print(end='')
            lock_preprocessing.release()

            lock_processing.acquire()
            start_arr.append(time.time())
            processed_batch = model(*inp).view(-1, 3, *tensor_batch.shape[-2:])
            batch_processed_flag = True
            print(end='')
            lock_processing.release()
        if frame_counter >= args.number_of_frames_to_check:
            break


device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cpu")
dtype = dtypes[args.tensor_dtype]

batch_preprocess = transforms.Compose([
    lambda batch_arr: torch.stack([torch.cat([
        torch.tensor(_frame) for _frame in frames
    ], dim=2) for frames in batch_arr], dim=0),
    lambda t: t.transpose(1, 3) / 255,
])
batch_center_preprocess = transforms.Compose([
    lambda batch_arr: torch.stack([torch.tensor(_frame) for _frame in batch_arr], dim=0),
    lambda t: t.transpose(1, 3) / 255,
])


if args.model == "sequenceframe":
    from models.FrameSequenceModel import SequenceFrameModel

    model = SequenceFrameModel(args.frames_sequence_length, args.weights, args.center_model, device, dtype)
elif args.model == "pairframe":
    from models.PairFrameModel import FramePairModel

    model = FramePairModel(args.weights, args.center_model, device, dtype)
else:
    raise ValueError("Unknown model: %s" % args.model)


vid = cv2.VideoCapture(args.video_path)
FPS = vid.get(cv2.CAP_PROP_FPS)

batch = [[] for _ in range(args.batch_size)]
batch_center = []
batch_id = 0
batch_frame_id = 0
frame_counter = 0

vid.set(cv2.CAP_PROP_POS_FRAMES, args.number_of_frames_to_skip)

out_frames = []

batch_preprocessed_flag = False
batch_processed_flag = False
processed_batch: torch.tensor

lock_pre = threading.Lock()
lock_post = threading.Lock()

processing_thread = threading.Thread(target=batch_processing_thread, args=(lock_pre, lock_post))
processing_thread.start()

global_start = time.time()
start_arr, finish_arr = [], []
timers = []

while True:
    if not batch_preprocessed_flag:
        ret, frame = vid.read()
        if not ret:
            break
        if batch_id == 0 and batch_frame_id == 0:
            lock_pre.acquire()
        if batch_id < args.batch_size:
            batch[batch_id].append(frame)
            batch_frame_id += 1
            if batch_frame_id == args.frames_sequence_length // 2 + 1:
                batch_center.append(frame)
                # print(frame_counter, batch_id, batch_frame_id)
            if batch_frame_id == args.frames_sequence_length:
                batch_frame_id = 0
                batch_id += 1
        else:
            tensor_batch = batch_preprocess(batch)
            tensor_batch_center = batch_center_preprocess(batch_center)
            batch_id = 0
            batch = [[] for _ in range(args.batch_size)]
            batch_center = []
            batch_preprocessed_flag = True
            print(end='')
            lock_pre.release()

    if batch_processed_flag:
        lock_post.acquire()
        print(end='')
        frames_array = np.clip(processed_batch.transpose(1, 3).numpy() * 255, 0., 255.).astype(np.uint8)
        batch_processed_flag = False
        lock_post.release()
        out_frames.extend([frames_array[i, :, :, :] for i in range(args.batch_size * args.frames_sequence_length)])
        frame_counter += args.batch_size * args.frames_sequence_length
        finish_arr.append(time.time())
        if args.verbose:
            print("Batch processed: %.4fs. Processed %d/%d frames" %
                  (finish_arr[-1] - start_arr[len(finish_arr) - 1], frame_counter, args.number_of_frames_to_check))

        timers.append(finish_arr[-1] - start_arr[len(finish_arr) - 1])

    if frame_counter >= args.number_of_frames_to_check:
        break

global_timer = time.time() - global_start
vid.release()

import pickle

filename = f"{args.model}x{args.frames_sequence_length}x{args.batch_size}_{args.center_model}.pkl"
with open(args.path_to_file + filename, 'wb+') as file:
    pickle.dump((timers, global_timer), file)
with open(args.path_to_file + 'FPS.txt', 'a') as file:
    print("%s: %.4f" % (f"{args.model}x{args.frames_sequence_length}x{args.batch_size}_{args.center_model}",
                        frame_counter / global_timer), file=file)

if args.save_results:
    print("Saving results into %s" % args.save_results + '.avi')
    out = cv2.VideoWriter(args.save_results + '.avi', -1, FPS,
                          (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    for frame in out_frames:
        out.write(frame)
    out.release()

if args.show_results:
    print("Showing results...")
    for frame in out_frames:
        cv2.imshow("processed", frame)
        time.sleep(1 / FPS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
