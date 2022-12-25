import torch
import config
from torchvision.utils import save_image

def check_cords(xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1):
    if xmin_mod_1<0:
        xmin_mod_1 = 0
    
    else:
        pass
    
    if ymin_mod_1<0:
        ymin_mod_1 = 0

    else:
        pass
    
    if xmax_mod_1>480:
        xmax_mod_1 = 480
    
    else:
        pass
    
    if ymax_mod_1>480:
        ymax_mod_1 = 480
    
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
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr