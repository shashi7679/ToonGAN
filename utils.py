import torch
import os
import config
import torch
from torchvision.utils import save_image
from Face_parsing_model import BiSeNet
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os.path as osp
import cv2
import time
import sys
import logging
import torch.distributed as dist
import skimage


def check_cords(xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1, x_shape=480, y_shape=480):
    if xmin_mod_1<0:
        xmin_mod_1 = 0
    
    else:
        pass
    
    if ymin_mod_1<0:
        ymin_mod_1 = 0

    else:
        pass
    
    if xmax_mod_1>x_shape:
        xmax_mod_1 = x_shape
    
    else:
        pass
    
    if ymax_mod_1>y_shape:
        ymax_mod_1 = y_shape
    
    else:
        pass
    
    return xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        temp = torch.cat((x, y), dim=0)
        output_image = torch.cat((temp, y_fake), dim=0)
        save_image(output_image, folder +  f"/output_{epoch}.png")
        # save_image(y_fake, folder + f"/toon_gen_{epoch}.png")
        # save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        # save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(model.state_dict(), filename)


def load_checkpoint(checkpoint_file, model):
    print("=> Loading checkpoint")
    new_checkpoint={}
    checkpoint = torch.load(checkpoint_file)
    # print(checkpoint["state_dict"])
    for k in checkpoint.keys():
        new_key = k.replace("module.","")
        new_checkpoint[new_key] = checkpoint[k]
    # model.load_state_dict(new_checkpoint["state_dict"])
    model.load_state_dict(new_checkpoint)
    return model
    # optimizer.load_state_dict(new_checkpoint["optimizer"])

    # # If we don't do this then it will just have learning rate of old checkpoint
    # # and it will lead to many hours of debugging \:
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr
    
    # return model

def setup_logger(logpth):
    logfile = 'BiSeNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank()==0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())

def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [255, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [255, 255, 255],
                   [0, 0, 0], [255, 255, 255], [0, 0, 0],
                   [255, 255, 255], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0]]
#0 - 
#1 - Frontal Face
#2 - Right Eye brow
#3 - Left Eye Brow
#4 - Left Eye
#5 - Right Eye
#6 - 

#12 - Upper Lip
#13 - Lower Lip
#14 - Neck
#15 - 
#16 - Shoulder
#17 - Hair
#18 - Hat
#19
#20 
    real_image = np.asarray(im)
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    # vis_im = cv2.cvtColor(vis_im, cv2.COLOR_BGR2RGB)

    mask = np.zeros(vis_im.shape, np.uint8)
    # mask = mask * 255
    # print("Mask : ",mask)
    output = np.where(vis_im>=np.array([150, 150, 150]), mask, real_image)
    skimage.io.imsave('im.png',output)
    return output

def face_parsing(image, model_name='79999_iter.pth'):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = os.path.join('saved_models', model_name)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        image = Image.fromarray(image)
        image = image.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        # print(np.unique(parsing))

        return vis_parsing_maps(image, parsing, stride=1)