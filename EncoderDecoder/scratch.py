from autoencoder import SimpleModel
from dataset import GE_Dataset
import train

import cv2
import math
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as f
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def convert_label(label):
    label = list(label)
    label[0] = str(round(float(label[0]) * 1000))
    label[1] = str(round(float(label[1]) * 1000))
    label[2] = str(round(math.degrees(label[2])))
    label[3] = str(round(math.degrees(label[3])))
    label[4] = str(round(math.degrees(label[4])))
    return label

model = SimpleModel().cuda()
# Only train the encoder for now
trainable_params = []
for name, param in model.named_parameters():
    if 'enc' in name:
        trainable_params.append(param)
optimizer = optim.Adam(trainable_params, lr=0.0005, betas=(0.9, 0.999))
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
epoch = -1

from utils.torch_utils import load

location = 'Goldengate Bridge'
# rel_path = 'ReplicateView1/2021-04-26[21_33_09]'
rel_path = '2021-04-26[22_41_49]'
path = f'E:/Research/code/EncoderDecoder/results/{rel_path}'
# path = 'E:/Research/code/EncoderDecoder/results/ModelsFromMeeting/Patience5/2021-04-13[06_14_04]' # brooklyn, goldengate
# path = 'E:/Research/code/EncoderDecoder/results/Patience6/2021-04-13[04_08_28]' # sydney harbour
# path = 'E:/Research/code/EncoderDecoder/results/Patience6/2021-04-13[03_34_27]' # tower
# path = 'E:/Research/code/EncoderDecoder/results/2021-04-21[23_57_28]'
# E:\Research\code\EncoderDecoder\results\ReplicateView1\
# E:/Research/code/EncoderDecoder/results/2021-04-21[23_57_28]

model.load_state_dict(torch.load(f'{path}/{location}/best.pth'))
# epoch, model, optimizer, scheduler = load(
#     'E:/Research/code/EncoderDecoder/results/2021-04-12[11_31_21]/Brooklyn Bridge/epoch_299.pth', model, optimizer, epoch, scheduler
# )

use_train = True
train_generator, val_generator, overhead_view = train.get_generators(Path(f'E:/Research/code/data/{location}'), batch_size=1)
if use_train:
    data_generator = train_generator
    dataset_type = 'train'
else:
    data_generator = val_generator
    dataset_type = 'val'

# val_split = 0.2
# seed = 42
# dataset = GE_Dataset(Path(f'E:/Research/code/data/{location}'))
# dataset_size = len(dataset)
# val_size = int(np.floor(val_split * dataset_size))
# train_dataset, val_dataset = torch.utils.data.random_split(
#     dataset, [dataset_size - val_size, val_size], generator=torch.Generator().manual_seed(seed)
# )

# data_generator = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
# data_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
# data_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
# data_generator = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

# open file to erase
# with open('scratch.log', 'w') as file:
#     pass

scratch_dir = Path(f'scratch/{location}/{rel_path}')
img_out_dir = scratch_dir / f'{dataset_type}'
if not img_out_dir.is_dir():
    img_out_dir.mkdir(parents=True)


tensors = []
losses = []
model.eval()
with torch.no_grad():
    running_loss = 0
    count = 0
    for x, y in data_generator:
        count += 1
        out, encoding = model(x)
        y_pred = encoding[:, :5].to(torch.float64)

        img_orig = x[0].cpu().numpy().transpose(1, 2, 0) * 255.0
        img_pred = out[0].cpu().numpy().transpose(1, 2, 0) * 255.0
        label = list(convert_label(y[0].cpu().numpy()))
        label = '_'.join(label)
        cv2.imwrite(f'{img_out_dir}/{label}_orig.png', img_orig)
        cv2.imwrite(f'{img_out_dir}/{label}_pred.png', img_pred)

        loss = f.mse_loss(y_pred, y)
        item_loss = loss.item()
        losses.append(item_loss)
        tensors.append({
            'Actual': y,
            'Pred': y_pred
        })

        running_loss += loss

h = sorted(losses)
mean = np.mean(h)
median = np.median(h)
std_dev = np.std(h)

print(f'Avg loss: {mean}')
print(f'Median loss: {median}')
print(f'Std Dev loss: {std_dev}')

args = np.argsort(losses)
with open (f'{scratch_dir}/predictions_{dataset_type}.log', 'w') as file:
    for i, loss in zip(args, h):
        actual = convert_label(tensors[i]["Actual"][0])
        pred = convert_label(tensors[i]["Pred"][0])
        file.write(f'Actual: {actual}\n')
        file.write(f'Pred: {pred}\n')
        file.write(f'MSE loss: {loss}\n')
        file.write('\n')
        # print(loss)
        # print(losses[i])



import scipy.stats as stats
import matplotlib.pyplot as plt

h = sorted(losses)
fit = stats.norm.pdf(h, mean, std_dev)
# print(fit)
plt.plot(h, fit, '-|')
plt.hist(h, density=True)
plt.title(f'Loss PDF - {location} - {dataset_type}')
plt.xlim([0, 8.5])
plt.ylim([0, 3])
plt.text(6.5, 2.85, f'Mean:      {round(mean, 3)}')
plt.text(6.5, 2.7, f'Median:   {round(median, 3)}')
plt.text(6.5, 2.55, f'Std Dev:  {round(std_dev, 3)}')

plt.savefig(f'{scratch_dir}/lossDistribution_{dataset_type}.png')

# with open('scratch2.log', 'a') as file:
#     for loss in losses:
#         file.write(f'{loss}\n')
#         # if loss == median_loss:
#         #     print("here")
# print(running_loss.item() / count + 1)

# model.eval()
# with torch.no_grad():
#     running_loss = 0
#     count = 0
#     for x, y in data_generator:
#         count += 1
#         out, encoding = model(x)
#         y_pred = encoding[:, :5].to(torch.float64)
#         loss = f.mse_loss(y_pred, y)
#         # loss = f.l1_loss(y_pred, y)
#         # loss = f.smooth_l1_loss(y_pred, y)
#         running_loss += loss

# print(running_loss.item() / count + 1)


# print(train.validate(model, data_generator))

# print()
# print(f'Avg loss: {running_loss.item() / (count + 1)}')
# import pandas as pd

# test = []

# test.append((1, 10000))
# test.append((2, 10500))
# test.append((3, 20000))

# df = pd.DataFrame(test)
