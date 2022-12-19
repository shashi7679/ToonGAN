import torch
import os
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import Face_Toon_Dataset,Test_Faces
from models import Generator,Discriminator
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import face_detection

import warnings
warnings.filterwarnings("ignore")



torch.backends.cudnn.benchmarks = True

torch.backends.cudnn.deterministic = True

wandb.login(key='46f2d6a5ffcc458fed2cca6cf446900f97e396e1')


Gen_loss = []
Dis_loss = []

def test(test_dl, save_dir):
    gen = Generator(in_channels=3).to(config.DEVICE)
    optm = torch.optim.Adam(gen.parameters(),lr=config.LEARNING_RATE,betas=(config.BETA1,0.999))
    ################# Loading least training loss Model #################
    file_path = f"./models/minloss_"+config.CHECKPOINT_GEN
    load_checkpoint(
        file_path,gen,optm,config.LEARNING_RATE
    )
    gen.eval()
    loop = tqdm(test_dl)
    with torch.no_grad():
        for idx, x in enumerate(loop):
            x = x.to(config.DEVICE)
            gen_img = gen(x)
            x = x * 0.5 + 0.5
            gen_img = gen_img * 0.5 + 0.5
            image = torch.cat((x, gen_img), dim=0)
            save_image(image, save_dir +  f"/best_train_model_{idx}.png")
    print("Results saved in ",save_dir," !!")

    ############### Loading least validation loss Model #################
    file_path = f"./models/min_valloss_"+config.CHECKPOINT_GEN
    load_checkpoint(
        file_path, gen, optm, config.LEARNING_RATE
    )
    gen.eval()
    loop = tqdm(test_dl)
    with torch.no_grad():
        for idx, x in enumerate(loop):
            x = x.to(config.DEVICE)
            gen_img = gen(x)
            x = x * 0.5 + 0.5
            gen_img = gen_img * 0.5 + 0.5
            image = torch.cat((x, gen_img), dim=0)
            save_image(image, save_dir +  f"/best_val_model_{idx}.png")
    print("Results saved in ",save_dir," !!")


def validation(netG, netD, val_dl, L2_Loss, BCE_Loss, Identity_loss):
    netG.eval()
    netD.eval()
    loop = tqdm(val_dl)
    val_loss = []
    with torch.no_grad():
        for idx, (x,y) in enumerate(loop):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            y_fake = netG(x)
            D_real = netD(x,y)
            D_fake = netD(x, y_fake)
            D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
            D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
            Dis_loss = (D_real_loss + D_fake_loss)/2

            G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
            L2 = L2_Loss(y_fake, y)
            Identity = Identity_loss(y_fake,y)
            Gen_loss = G_fake_loss*config.CE_LAMBDA + L2*config.L2_LAMBDA - Identity*config.IDENTITY_LAMBDA

            wandb.log({
                'val_disc_loss':Dis_loss.item(), 
                'val_gen_loss':Gen_loss.item(), 
                'val_BCELoss':G_fake_loss.item(),
                'val_L2Loss':L2.item(),
                'val_Identity':Identity.item()
                })
            val_loss.append(Gen_loss)
    loss = torch.stack(val_loss).mean().item()
    print("[INFO] Validation loss :- ",loss)
    netD.train()
    netG.train()
    return loss



def train(netG, netD, train_dl, OptimizerG, OptimizerD, L2_Loss, BCE_Loss, Identity_loss):
    loop = tqdm(train_dl)
    train_loss=[]
    for idx, (x,y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        ############## Train Discriminator ##############
        y_fake = netG(x)
        D_real = netD(x,y)
        D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
        D_fake = netD(x,y_fake.detach())
        D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss)/2
        #D_loss = D_real_loss + D_fake_loss
        
        netD.zero_grad()
        Dis_loss.append(D_loss.item())
        D_loss.backward()
        OptimizerD.step()
        
        ############## Train Generator ##############
        D_fake = netD(x, y_fake)
        G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
        L2 = L2_Loss(y_fake,y)
        L_identity = Identity_loss(x, y_fake)
        G_loss = G_fake_loss*config.CE_LAMBDA + L2*config.L2_LAMBDA - L_identity*config.IDENTITY_LAMBDA
        current_gen_loss = G_loss.item()

        
        OptimizerG.zero_grad()
        Gen_loss.append(G_loss.item())
        G_loss.backward()
        OptimizerG.step()
        train_loss.append(G_loss)

        
        wandb.log({
            "train_Gen_loss": G_loss.item(), 
            "train_Disc_loss":D_loss.item(),
            "train_BCELoss":G_fake_loss.item(),
            "train_L2Loss":L2.item(),
            "train_Identity":L_identity.item()
            })
        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
    loss = torch.stack(train_loss).mean().item()
    return loss


def main():
    best_gen_loss = 100.0
    best_val_loss = 100.0 
    netD = Discriminator(in_channels=3).to(config.DEVICE)
    netG = Generator(in_channels=3).to(config.DEVICE)
    OptimizerD = torch.optim.Adam(netD.parameters(),lr=config.LEARNING_RATE,betas=(config.BETA1,0.999))
    OptimizerG = torch.optim.Adam(netG.parameters(),lr=config.LEARNING_RATE,betas=(config.BETA1,0.999))
    BCE_Loss = nn.BCEWithLogitsLoss()
    Identity_loss = nn.L1Loss()
    L2_Loss = nn.MSELoss()
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,netG,OptimizerG,config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,netD,OptimizerD,config.LEARNING_RATE
        )
        
    train_dataset = Face_Toon_Dataset(root=config.TRAIN_DIR)
    train_dl = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=True)
    val_dataset = Face_Toon_Dataset(root=config.VAL_DIR)
    val_dl = DataLoader(val_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=True)
    test_dataset = Test_Faces(root=config.TEST_DIR)
    test_dl = DataLoader(test_dataset,batch_size=1,shuffle=False)
    
    
    for epoch in range(config.NUM_EPOCHS):
        print("\n")
        print("[INFO] EPOCH :- ",epoch+1)
        print("\n")
        current_gen_loss = train(
            netG, netD, train_dl, OptimizerG, OptimizerD, L2_Loss, BCE_Loss, Identity_loss
        )
        #+++++++++++++++++++++++++++++++++++++++++#
        ########## Train Epoch Analysis ###########
        #+++++++++++++++++++++++++++++++++++++++++#
        if current_gen_loss<best_gen_loss:
            ##### Min Gen Loss #####
            best_gen_loss = current_gen_loss
            if config.SAVE_MODEL:
                file_name = f"./models/minloss_"+config.CHECKPOINT_GEN
                save_checkpoint(netG, OptimizerG, filename=file_name)
                print(f"Saved Generator with min loss after {epoch+1} epochs!!!!")
        else:
            print("Not the best one!!!!")

        #++++++++++++++++++++++++++++++++++++++++++#
        ########### Val Epoch Analysis #############
        #++++++++++++++++++++++++++++++++++++++++++#
        val_loss = validation(netG, netD, val_dl, L2_Loss, BCE_Loss, Identity_loss)
        if val_loss<best_val_loss:
            #### Min Val Gen Loss ####
            best_val_loss = val_loss
            if config.SAVE_MODEL:
                file_name = f"./models/min_valloss_"+config.CHECKPOINT_GEN
                save_checkpoint(netG, OptimizerG, filename=file_name)
                print("Saved best Val performing generator!!!!")
        
        ############### Training and Validation Loss after each epoch ############
        wandb.log({
            "train_loss":current_gen_loss,
            "val_loss":val_loss
        })

        if config.SAVE_MODEL and (epoch+1)%50==0:
            save_checkpoint(netG, OptimizerG, filename=config.CHECKPOINT_GEN)
            save_checkpoint(netD, OptimizerD, filename=config.CHECKPOINT_DISC)
        
        ############# Test Images ###########
        if (epoch+1)%5==0:
            print("\n")
            print("Testing on Real Faces...")
            save_dir = f"./test_results/{epoch+1}_results"
            Exists = os.path.exists(save_dir)
            if not Exists:
                os.makedirs(save_dir)
            test(test_dl, save_dir)

        
        if (epoch+1)%5==0:
            save_some_examples(netG, val_dl, epoch+1, folder="evaluation")

    
if __name__ == "__main__":
    
    hyper_parameters = dict(
        DEVICE = config.DEVICE,
        LEARNING_RATE = config.LEARNING_RATE,
        BETA1 = config.BETA1,
        BATCH_SIZE = config.BATCH_SIZE,
        NUM_WORKERS = config.NUM_WORKERS,
        IMAGE_SIZE = config.IMAGE_SIZE,
        CHANNELS_IMG = config.CHANNELS_IMG,
        L2_LAMBDA = config.L2_LAMBDA,
        CE_LAMBDA = config.CE_LAMBDA,
        IDENTITY_LAMBDA = config.IDENTITY_LAMBDA,
        LAMBDA_GP = config.LAMBDA_GP,
        NUM_EPOCHS = config.NUM_EPOCHS
    )
    wandb.init(project='Toon-GAN-2',config=hyper_parameters)
    main()
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(Gen_loss,label="Generator")
    plt.plot(Dis_loss,label="Discriminator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss.png')
