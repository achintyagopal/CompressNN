from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import cv2 as cv
from main import VAE
from visualize import make_dot

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--eval-images', type=int, default=100, metavar='N',
                    help='number of samples to generate (should be perfect square)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--load-model", type=str, default='vae_10',
        help="The file containing already trained model.")
parser.add_argument("--save-model", default="vae", type=str,
        help="The file containing already trained model.")

parser.add_argument("--save-image", default="vae", type=str,
        help="The file containing already trained model.")
parser.add_argument("--mode", type=str, default="train-eval", choices=["train", "eval", "train-eval"],
                        help="Operating mode: train and/or test.")

parser.add_argument("--num-samples", default=1, type=int,
        help="The number of samples to draw from distribution")
args = parser.parse_args()


torch.manual_seed(args.seed)

kwargs = {}

data_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)

model = torch.load(args.load_model)
reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False

def stack(ra):
    num_per_row = int(np.sqrt(len(ra)))
    rows = [np.concatenate(tuple(ra[i* num_per_row : i*num_per_row + num_per_row]), axis=1) 
            for i in range(num_per_row)]
    img = np.concatenate(tuple(rows), axis=0)
    return img

# model.eval()

total_loss = 0.
compression = 0.
# print(len(data_loader))
for batch_idx, (data, _) in enumerate(data_loader):
    x = Variable(data)
    mu, logvar, mask = model.encode(x)
    
    mu = mu.detach()
    mask = mask.detach()
    mu.requires_grad = True
    optimizer = optim.Adam([mu], lr=3e-1)

    for _ in range(20):
        optimizer.zero_grad()
        
        recon_x = model.decode(mu * mask)
        loss = torch.sum(reconstruction_function(recon_x, x))
        loss.backward()
        
        optimizer.step()

    results = (recon_x[:100,:]).view(100, 28, 28)
    imgs = []
    for result in results:
        print(result.size())
        imgs.append(result.data.cpu().numpy())
    imgFile = stack(imgs)
    imgFile = imgFile * 255 / np.max(imgFile)
    cv.imwrite("Eval_" + str(batch_idx) + ".png", imgFile)

    # # print(recon_x.size())

    total_loss += torch.sum(reconstruction_function(recon_x, x))

    print(float(torch.sum(mask == 0.).data.numpy()[0]))
    compression += float(torch.sum(mask == 0.).data.numpy()[0])
    # compression += (float(compress)/((batch_idx + 1) * args.batch_size))
    if batch_idx % 10 == 0:
        print(batch_idx * args.batch_size)

print(total_loss / 10000)
print(compression / 10000)






