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
parser.add_argument("--load-model", type=str,
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

if "train" in args.mode:
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if "eval" in args.mode:
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if args.mode == "eval":
    if not args.load_model:
        raise ValueError("Need which model to evaluate")
    args.epoch = 1
    args.eval_interval = 1


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.z_size = 20
        self.seq = nn.Sequential(
            # input is 1 x 28 x 28
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 14 x 14
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64)
        )

        # state size. 64 x 7 x 7
        # need to resize to batch, 6272
        self.fc1 = nn.Linear(64*7*7, 1024)

        self.fc21 = nn.Linear(1024, self.z_size)
        self.fc22 = nn.Linear(1024, self.z_size)
        self.fc23 = nn.Linear(1024, self.z_size)

        # have z vector
        self.fc3 = nn.Linear(self.encoder_size(), 1024)

        self.fc4 = nn.Linear(1024, 64*7*7)

        self.seq2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            # 64,14,14
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder_size(self):
        return self.z_size

    def encode(self, x):
        h1 = self.seq(x)
        h2 = h1.view(-1,64*7*7)
        h3 = self.relu(self.fc1(h2))
        return self.fc21(h3), self.fc22(h3), torch.round(self.sigmoid(self.fc23(h3)))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.fc4(h3)
        h5 = h4.view(-1, 64, 7, 7)
        return self.seq2(h5)

    def sampleAndDecode(self, mu, logvar, mask):
        z = self.reparametrize(mu, logvar)

        z = z * mask
        # max_val, _ = torch.max(x, x.dim() - 1, keepdim=True)
        # x_hard = x == max_val.expand_as(x)
        # tmp  = x_hard.float() - x
        # tmp2 = tmp.clone()
        # tmp2.detach_()
        # x = tmp2 + x


        return self.decode(z), mu, logvar, mask

    def forward(self, x):
        mu, logvar, mask = self.encode(x)
        return self.sampleAndDecode(mu, logvar, mask)


model = VAE()

reconstruction_function = nn.MSELoss()
bce_loss = nn.BCELoss()
bce_loss.size_average = False
reconstruction_function.size_average = False

def loss_function(recon_xs, x, mu, logvar, mask):
    BCE = 0
    for recon_x in recon_xs:
        BCE += reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    KLD += bce_loss(mask, Variable(torch.zeros(mask.size())))
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    # i = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        # i += data.size()[0]
        # repeat model(data) multiple times, mu and logvar won't change, recon_batch will, it's like batch 
        total_batch = []
        recon_batch, mu, logvar, mask = model(data)
        # print(recon_batch.size())
        total_batch.append(recon_batch)
        for _ in range(args.num_samples - 1):
            recon_batch, _, _, _ = model.sampleAndDecode(mu, logvar, mask)
            total_batch.append(recon_batch)

        loss = loss_function(total_batch, data, mu, logvar, mask)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

epses = []
for _ in range(args.eval_images):
    z = torch.FloatTensor(1,model.z_size).normal_()
    z = Variable(z)
    epses.append(z)

def test(epoch):
    model.eval()
    test_loss = 0
    i = 0
    for batch_idx, (data, _) in enumerate(test_loader):
        data = Variable(data, volatile=True)
        # recon_batch, mu, logvar, mask = model(data)
        mu, logvar, mask = model.encode(data)
        recon_batch = model.decode(mu * mask)
        test_loss += loss_function([recon_batch], data, mu, logvar, mask).data[0]
        if i == 0:
            results = (recon_batch[:100,:]).view(100, 28, 28)
            imgs = []
            for result in results:
                # print(result.size())
                imgs.append(result.data.cpu().numpy())
            imgFile = stack(imgs)
            imgFile = imgFile * 255 / np.max(imgFile)
            imgFileName = args.save_image + "_" + str(epoch) + ".png"
            cv.imwrite(imgFileName, imgFile)

        i += 1

    test_loss /= len(test_loader.dataset)
    print(test_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    # if epoch % args.eval_interval == 0:
    #     imgs = []
    #     for eps in epses:
    #         model.eval()
    #         x = model.decode(eps)
    #         imgFile = np.resize((x.data).cpu().numpy(), (28,28))
    #         imgs.append(imgFile)

    #     imgFile = stack(imgs)
    #     imgFile = imgFile * 255 / np.max(imgFile)
    #     imgFileName = args.save_image + "_" + str(epoch) + ".png"
    #     cv.imwrite(imgFileName, imgFile)

def stack(ra):
    num_per_row = int(np.sqrt(len(ra)))
    rows = [np.concatenate(tuple(ra[i* num_per_row : i*num_per_row + num_per_row]), axis=1) 
            for i in range(num_per_row)]
    img = np.concatenate(tuple(rows), axis=0)
    return img

if args.load_model:
    model = torch.load(args.load_model)

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        if "train" in args.mode:
            train(epoch)
        if "eval" in args.mode:
            test(epoch)

        if epoch % args.save_interval == 0:
            torch.save(model, args.save_model + "_" + str(epoch))

    torch.save(model, args.save_model + "_" + str(epoch))
