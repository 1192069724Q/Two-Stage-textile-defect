"""
@Author: Alex
@Date: 2022.July.4th
Wasserstein GANs with gradient penalty
"""
import argparse
import os
import numpy as np
import math
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch


from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from models import DTB_GAN

def create_DTBGAN(num_class, pretrain=False):
    return  DTB_GAN(in_channels=4,model_channels=32,out_channels=num_class,num_res_blocks=1)




DTB=create_DTBGAN(num_class=3)

def load_data(
        *, data_dir, batch_size, image_size, class_cond=False, deterministic=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files, all_masks = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        all_masks,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    results_mask = []
    for entry in sorted(bf.listdir(data_dir + "/img")):
        full_path = bf.join(data_dir + "/img", entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    for entry in sorted(bf.listdir(data_dir + "/mask")):
        full_path = bf.join(data_dir + "/mask", entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp"]:
            results_mask.append(full_path)
        elif bf.isdir(full_path):
            results_mask.extend(_list_image_files_recursively(full_path))
    return results, results_mask


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, mask_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_masks = mask_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        p_mask = self.local_masks[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        with bf.BlobFile(p_mask, "rb") as f:
            pil_mask = Image.open(f)
            pil_mask.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )
        while min(*pil_mask.size) >= 2 * self.resolution:
            pil_mask = pil_mask.resize(
                tuple(x // 2 for x in pil_mask.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
        pil_mask = pil_mask.resize(
            tuple(round(x * scale) for x in pil_mask.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y: crop_y + self.resolution, crop_x: crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        arr_mask = np.array(pil_mask.convert("L"))
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), np.expand_dims(arr_mask/255 ,0),out_dict


os.makedirs("images", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=25, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)


img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
#generator = Generator()

generator =DTB
discriminator = Discriminator()

trained_GA=True
## freeze G_B(model_B)

if trained_GA:
    for name,param in generator.named_parameters():
        if name.split('.')[0]=='model_B':
            param.requires_grad=False

# freeze G_A (input_block,middle_block,output_block,out) or
else:
    for name,param in generator.named_parameters():
        if name.split('.')[0]!='model_B':
            param.requires_grad=False



if cuda:
    generator.cuda()
    discriminator.cuda()

data_dir="d3"
data=load_data(data_dir=data_dir,batch_size=2,image_size=256,class_cond=True,deterministic=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
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

## when we train G_B,we define a move list
batches_done = 0
key =0          #  control move interval
xy_idx=0
def t_array():
    x=-1.0
    y=-1.0
    d=0.2
    t_list=[]
    for i in range(11):
        for j in range(11):
            #print((x+d*i,y+d*i))
            t_list.append((x+d*i,y+d*j))
    return t_list

move= t_array()  #move_list

for epoch in range(opt.n_epochs):
    for i, (imgs, mask,c) in enumerate(data):

        mask=mask.float().cuda()
        imgs=imgs.cuda()
        if key % 500 ==0:
            print('move!')
            an, bn = move[xy_idx]
            xy_idx=xy_idx+1
        theta = torch.tensor([
                [1, 0, an],
                [0, 1, bn]
        ], dtype=torch.float).cuda()
        grid = F.affine_grid(theta.unsqueeze(0).repeat(2, 1, 1), imgs.size()).cuda()
        imgs = F.grid_sample(imgs, grid)
        mask = F.grid_sample(mask, grid)

        # saved moved imgs and mask
     #   if key % 500 ==0:
       #     save_image(mask.data[:25], "mask/mask_%d.png" % key, nrow=5, normalize=True)
        #    save_image(imgs.data[:25], "mask/imgs_%d.png" % key, nrow=5, normalize=True)



        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for k in range(5):

            optimizer_D.zero_grad()

        # Sample noise as generator input
        #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            z1 = Variable(torch.randn((imgs.shape[0],100))).cuda()
            z2 = Variable(torch.randn((imgs.shape[0], 3, 256, 256))).cuda()
        # Generate a batch of images
            fake_imgs = generator(z1,z2,mask)



        # Real images
            real_validity = discriminator(real_imgs)
        # Fake images
            fake_validity = discriminator(fake_imgs)
        # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)
        # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z1,z2,mask)

            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, 2, d_loss.item(), g_loss.item())
            )

            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic
        key=key+1
        if i==100:
            break
    if epoch % 20 == 0:
        torch.save(discriminator, "a_first_stage_models/" + str(epoch) + ".pt")
        torch.save(generator.state_dict(), "a_second_stage_models/" + str(epoch) + ".pt")