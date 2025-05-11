import os
import time

import cv2
from models.PairFrameModel import FramePairModel

import torch
from torchvision import transforms
import numpy as np


HEIGHT, WIDTH = 512, 960
PATH = "data/video/low/"
VIDEOS = os.listdir(PATH)
print(VIDEOS)
ID = 0

vid = cv2.VideoCapture(PATH + VIDEOS[ID])
FPS = vid.get(cv2.CAP_PROP_FPS)

FRAMES_NUMBER = 15
model = FramePairModel(f"models/weights/pair_x_{FRAMES_NUMBER}frames_mse_trained.pth")
transform_to_tensor = transforms.Compose([
    torch.tensor,
    lambda t: t.transpose(2, 0) / 255
])
transform_to_numpy = transforms.Compose([
    lambda t: model(*t),
    lambda t: (t.detach().transpose(2, 0).cpu().numpy() * 255).astype(np.uint8)
])


frames_queque = []
tensor_frames = []
predicted_frames = []
skip_frames = 255
total = 0
while True:
    rep, frame =vid.read()
    if skip_frames:
        skip_frames -= 1
        continue
    if total == 180:
        break
    if len(tensor_frames) < FRAMES_NUMBER:
        if not tensor_frames:
            if not frames_queque:
                global_start = time.time()
            start = time.time()
        frames_queque.append(frame)
        tensor_frames.append(transform_to_tensor(frame))
        continue
    predicted_center = model.process_center(tensor_frames[FRAMES_NUMBER // 2])
    predicted_frames += [transform_to_numpy((predicted_center, frame, True)) for frame in tensor_frames]

    tensor_frames = []
    print("Time to process %d frames: %.2f s" % (FRAMES_NUMBER, time.time() - start))

    total += FRAMES_NUMBER

print("Time to process %d frames: %.2f s" % (len(predicted_frames), time.time() - global_start))

time.sleep(5)
for source, predicted in zip(frames_queque, predicted_frames):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("source", source)
    cv2.imshow("predicted", predicted)
    time.sleep(1 / FPS)

vid.release()
cv2.destroyAllWindows()
