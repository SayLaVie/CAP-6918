from autoencoder import SimpleModel
from dataset import GE_Dataset
import utils.torch_utils as torch_utils
import utils.logger as Logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchinfo import summary


LEARNING_RATE = 0.0005
EPSILON = 1e-8
BETA_1 = 0.9
BETA_2 = 0.999
BATCH_SIZE = 1
DATASET_LIST = ['Brooklyn Bridge', 'Goldengate Bridge', 'Sydney Harbour Bridge', 'Tower Bridge']
# DATASET_LIST = ['Goldengate Bridge', 'Sydney Harbour Bridge', 'Tower Bridge']
NUM_EPOCHS = 300
DESCRIPTIVE_NOTES = (
    "...Loss1 type: MSE Loss\n"
    "...Loss2 type: MSE Loss\n"
    "...Activation function: Leaky Relu"
)
MISC_NOTES = (
    "...These runs have moved the activation functions to wrap only the convolutions\n"
    "...Initializing weights the same way as VGG\n"
    "...Using patience value of 10\n"
    "...Using the decoder to match input view"
)


def train(model, optimizer, generator, overhead_view=None):
    model.train()
    running_loss = 0
    count = 0
    for x, y in generator:
        count += 1
        out, encoding = model(x)
        y_pred = encoding[:, :5].to(torch.float64)
        loss1 = f.mse_loss(y_pred, y)
        loss2 = f.mse_loss(x, out)
        # loss2 = f.mse_loss(overhead_view, out)
        loss = loss1 + loss2
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss.item() / (count + 1)

def validate(model, generator, overhead_view=None):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        count = 0
        for x, y in generator:
            count += 1
            out, encoding = model(x)
            y_pred = encoding[:, :5].to(torch.float64)
            loss1 = f.mse_loss(y_pred, y)
            loss2 = f.mse_loss(x, out)
            # loss2 = f.mse_loss(overhead_view, out)
            loss = loss1 + loss2
            running_loss += loss
    
    return running_loss.item() / (count + 1)

def setup_dirs(location, timestamp):
    results_dir = Path(f'results/{timestamp}/{location}')
    if not results_dir.is_dir():
        results_dir.mkdir(parents=True)
    return results_dir

def get_generators(path, val_split=0.2, seed=42, batch_size=BATCH_SIZE):
    dataset = GE_Dataset(path)
    dataset_size = len(dataset)
    val_size = int(np.floor(val_split * dataset_size))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [dataset_size - val_size, val_size], generator = torch.Generator().manual_seed(seed)
    )
    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Create tensor of batched overhead views
    overhead_view = torch.empty((BATCH_SIZE, 3, 224, 224)) # (3, 224, 224) is hardcoded image shape
    for i in range(BATCH_SIZE):
        overhead_view[0] = dataset.overhead_view
    print(overhead_view.shape)

    return train_generator, val_generator, overhead_view.cuda()

def print_history(history, filename):
    df = pd.DataFrame(history)
    df.to_csv(f'{filename}.csv')
    plt.plot(df)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validate'], loc='upper right')
    plt.savefig(f'{filename}.png')
    plt.close()

def write_predicted_overhead(img):
    pass


"""
Main
"""
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    from utils.timer import Timer

    for i in range(3): # do each location multiple times for comparison
        timestamp = Timer.timeFilenameString()
        timer = Timer()
        for location in DATASET_LIST:
            timer.start()
            model = SimpleModel().cuda()

            # print(str(model))
            # model.apply(init_weights)
            train_generator, val_generator, overhead_view = get_generators(Path(f'E:/Research/code/data/{location}'))

            # Only train the encoder for now
            # trainable_params = []
            # for name, param in model.named_parameters():
            #     if 'enc' in name:
            #         trainable_params.append(param)
            trainable_params = model.parameters()
            optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON)
            # optimizer = optim.SGD(trainable_params, lr=LEARNING_RATE)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
            # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
            
            results_dir = setup_dirs(location, timestamp)
            log_filename = results_dir / f'{timestamp}.log'
            log = Logger.Logger(str(log_filename), stdout=True)
            log.write(f'Training on location: {location}')
            log.write(f'Learning rate: {LEARNING_RATE}')
            log.write(f'Beta 1: {BETA_1}')
            log.write(f'Beta 2: {BETA_2}')
            log.write(f'Epsilon: {EPSILON}')
            log.write(f'Batch size: {BATCH_SIZE}')
            log.write(f'Num Epochs: {NUM_EPOCHS}')
            log.write(f'Descriptive Notes:\n{DESCRIPTIVE_NOTES}')
            log.write(f'Misc Notes:\n{MISC_NOTES}')
            log.write()

            # TODO: capture this in log output
            summary(model, (BATCH_SIZE, 3, 224, 224))

            start_epoch = 0

            # Manually load resumes here for now
            # start_epoch, model, optimizer, scheduler = torch_utils.load(
            #     'E:/Research/code/EncoderDecoder/results/2021-04-07[21_13_24]/Brooklyn Bridge/epoch_299.pth', 
            #     model, optimizer, start_epoch, scheduler)
            # log.write(f'Resuming from E:/Research/code/EncoderDecoder/results/2021-04-07[21_13_24]/Brooklyn Bridge/epoch_299.pth at epoch {start_epoch}')

            log.write()

            history = { "train": [], "validate": [] }
            best_val_loss = np.inf
            best_epoch = -1

            for epoch in range(start_epoch, start_epoch+NUM_EPOCHS):
                # train_loss = train(model, optimizer, train_generator, overhead_view)
                train_loss = train(model, optimizer, train_generator)
                # val_loss = validate(model, val_generator, overhead_view)
                val_loss = validate(model, val_generator)
                scheduler.step(val_loss)
                # scheduler.step()

                history["train"].append(train_loss)
                history["validate"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_filename = results_dir / 'best.pth'
                    torch.save(model.state_dict(), str(best_model_filename))
                
                # Save progress
                if (epoch + 1) % 100 == 0:
                    log.write(f'Saving interim progress at epoch {epoch}')
                    iterim_filename = results_dir / f'epoch_{epoch}'
                    recent_history = { 'train': [], 'validate': [] }
                    recent_history['train'] = history['train'][-100:]
                    recent_history['validate'] = history['validate'][-100:]
                    print_history(recent_history, iterim_filename) # Print only last 100 results
                    # torch_utils.save(f'{iterim_filename}.pth', epoch+1, model, optimizer, scheduler=scheduler)

            timer.stop()
            log.write(f'Training {location} finished in {timer.elapsed() / 60} minutes')
