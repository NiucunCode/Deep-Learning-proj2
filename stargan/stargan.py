import argparse
import os
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from models import weights_init_normal
from models import LambdaLR
from models import GeneratorResNet
from models import Discriminator
from dataloader import ImageDataset


os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=2, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="data", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.000001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=25, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")
parser.add_argument("--img_width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--lambda_cls", type=float, default=1.0, help="domain classification loss weight")
parser.add_argument("--lambda_rec", type=float, default=10.0, help="reconstruction loss weight")
parser.add_argument("--lambda_gp", type=float, default=10.0, help="gradient loss weight")
parser.add_argument("--is_print", type=bool, default=False, help="whether to print the network or not")
parser.add_argument(
    "--selected_attrs",
    "--list",
    nargs="+",
    help="selected attributes for translation",
    default=["Monet", "Vangogh", "Ukiyoe", "Cezanne"],
    # default=["Monet", "Vangogh", "Ukiyoe", "Cezanne", "Photo"],
)
parser.add_argument("--n_critic", type=int, default=5, help="number of training iterations for WGAN discriminator")
opt = parser.parse_args()
print(opt)

c_dim = len(opt.selected_attrs)
img_shape = (opt.channels, opt.img_height, opt.img_width)

cuda = torch.cuda.is_available()

# Loss functions
criterion_cycle = nn.L1Loss()


def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)


def print_network(model, name):
    """
    Print out the network information
    https://github.com/yunjey/stargan/blob/master/solver.py
    """
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


# Initialize generator and discriminator
generator = GeneratorResNet(input_shape=img_shape, residual_blocks=opt.residual_blocks, c_dim=c_dim)
discriminator = Discriminator(input_shape=img_shape, c_dim=c_dim)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_cycle.cuda()

if opt.is_print:
    print_network(generator, 'Generator')
    print_network(discriminator, 'Discriminator')

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if opt.epoch != 0:
    # Load pre-trained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
'''
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
'''
# Configure transforms
train_transforms = [
    transforms.Resize(int(1.12 * opt.img_height), Image.BICUBIC),
    transforms.CenterCrop(opt.img_height),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
test_transforms = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
# Configure train data_loader and test data_loader
dataloader = DataLoader(
    ImageDataset(root=opt.dataset_name, transforms_=train_transforms, mode="train", attributes=opt.selected_attrs),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

test_dataloader = DataLoader(
    ImageDataset(root=opt.dataset_name, transforms_=test_transforms, mode="test", attributes=opt.selected_attrs),
    batch_size=10,
    shuffle=True,
    num_workers=1,
)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """
    Calculates the gradient penalty loss for WGAN GP:(L2_norm(dy/dx) - 1)**2
    https://github.com/yunjey/stargan/blob/master/solver.py
    """
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# fixed images to test
fixed_test_imgs, fixed_test_labels = next(iter(test_dataloader)) # TODO:change


def sample_images(batches_done, test_imgs, test_labels):
    """
    Saves a generated sample of domain translations
    """

    test_imgs = Variable(test_imgs.type(Tensor))
    test_labels = Variable(test_labels.type(Tensor))

    img_samples = None
    for i in range(10):
        img, label = test_imgs[i], test_labels[i]

        # Repeat for number of label changes
        imgs = img.repeat(c_dim, 1, 1, 1)
        # Make changes to labels
        labels = Variable(Tensor(np.eye(c_dim)))
        # Generate translations
        gen_imgs = generator(imgs, labels)
        # Concatenate images by width
        gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
        img_sample = torch.cat((img.data, gen_imgs), -1)
        # Add as row to generated samples
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)

    save_image(img_samples.view(1, *img_samples.shape), "images/%s.png" % batches_done, normalize=True)


# Train
print("Start Training...")

for epoch in range(opt.epoch, opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        # Model inputs
        imgs = Variable(imgs.type(Tensor))
        labels = Variable(labels.type(Tensor))

        # Sample labels as generator inputs
        fake_label = np.zeros((imgs.size(0), c_dim))
        # index
        label_index = np.random.randint(0, c_dim, (imgs.size(0), 1))
        img_index = np.arange(imgs.size(0)).reshape(-1, 1)
        fake_label[img_index, label_index] = 1
        sample_l = Variable(Tensor(fake_label))
        # Generate fake batch of images
        fake_imgs = generator(imgs, sample_l)
        # Train Discriminator

        # Real images
        real_validity, pred_cls = discriminator(imgs)
        # Fake images
        fake_validity, _ = discriminator(fake_imgs.detach())
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data)
        # Adversarial loss
        loss_D_adv = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty
        # Classification loss
        loss_D_cls = criterion_cls(pred_cls, labels)
        # Total loss
        loss_D = loss_D_adv + opt.lambda_cls * loss_D_cls
        # Backward and optimize
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Every n_critic times update generator
        if (i+1) % opt.n_critic == 0:
            # Train Generator

            # Translate and reconstruct image
            gen_imgs = generator(imgs, sample_l)
            recov_imgs = generator(gen_imgs, labels)
            # Discriminator evaluates translated image
            fake_validity, pred_cls = discriminator(gen_imgs)
            # Adversarial loss
            loss_G_adv = -torch.mean(fake_validity)
            # Classification loss
            loss_G_cls = criterion_cls(pred_cls, sample_l)
            # Reconstruction loss
            loss_G_rec = criterion_cycle(recov_imgs, imgs)
            # Total loss
            loss_G = loss_G_adv + opt.lambda_cls * loss_G_cls + opt.lambda_rec * loss_G_rec
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f, adv: %f, aux: %f] [G loss: %f, adv: %f, aux: %f, cycle: %f]"
                % (
                    epoch+1,
                    opt.n_epochs,
                    i+1,
                    len(dataloader),
                    loss_D.item(),
                    loss_D_adv.item(),
                    loss_D_cls.item(),
                    loss_G.item(),
                    loss_G_adv.item(),
                    loss_G_cls.item(),
                    loss_G_rec.item(),
                )
            )
        batches_done = epoch * len(dataloader) + i
        # If at sample interval sample and save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, fixed_test_imgs, fixed_test_labels)

    # Update learning rates
    # lr_scheduler_G.step()
    # lr_scheduler_D.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % (epoch+1))
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % (epoch+1))
