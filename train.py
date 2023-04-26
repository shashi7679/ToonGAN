import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import Face_Toon_Dataset, Test_Faces
from models import Generator,Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import os
import cv2
import numpy as np
from torchvision.utils import save_image



torch.backends.cudnn.benchmarks = True

torch.backends.cudnn.deterministic = True

wandb.login(key='46f2d6a5ffcc458fed2cca6cf446900f97e396e1')


Gen_loss = []
Dis_loss = []

def test(test_dl, save_dir):
    gen = Generator(in_channels=3)
    ################# Loading least training loss Model #################
    file_path = f"./saved_models/min_train_loss_"+config.CHECKPOINT_GEN
    # file_path = f"./saved_models/min_train_loss_gen.pt"
    gen = load_checkpoint(
        file_path,gen
    )
    gen = gen.to(config.DEVICE)
    # gen = torch.jit.load(file_path).to(config.DEVICE)
    gen.eval()
    loop = tqdm(test_dl)
    with torch.no_grad():
        for idx, x in enumerate(loop):
            image, (x_shape, y_shape), (xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1), original_image = x[0], x[1], x[2], x[3]
            x_shape, y_shape, xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = x_shape.item(), y_shape.item(), xmin_mod_1.item(), ymin_mod_1.item(), xmax_mod_1.item(), ymax_mod_1.item()
            image = image.to(config.DEVICE)
            toon_face = gen(image)
            image = image * 0.5 + 0.5
            toon_face = toon_face * 0.5 + 0.5
            actual_x_dim, actual_y_dim = (xmax_mod_1 - xmin_mod_1), (ymax_mod_1 - ymin_mod_1)
            toon_face = toon_face.permute(0,2,3,1)
            toon_face = toon_face.cpu().detach().numpy()
            toon_face = toon_face[0]
            # toon_face = cv2.resize(toon_face, (actual_x_dim, actual_y_dim))

            image = image.permute(0,2,3,1)
            image = image.cpu().detach().numpy()
            image = image[0]
            # image = cv2.resize(image, (actual_x_dim, actual_y_dim))

            concat_array = np.concatenate((image,toon_face),axis=0)
            concat_array = concat_array * 255.0
            concat_array = cv2.normalize(concat_array, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            concat_array = cv2.cvtColor(concat_array, cv2.COLOR_BGR2RGB)
            saved_file = str(save_dir) + f"/best_train_model_{idx}.png"
            cv2.imwrite(saved_file, concat_array)
            # save_image(image, save_dir +  saved_file)
    print("Test results using best train model saved in ",save_dir," !!")

    ############### Loading least validation loss Model #################
    file_path = f"./saved_models/min_val_loss_"+config.CHECKPOINT_GEN
    gen2 = Generator(in_channels=3)
    gen2 = load_checkpoint(
        file_path, gen2
    )
    gen2 = gen2.to(config.DEVICE)
    gen2.eval()
    loop = tqdm(test_dl)
    with torch.no_grad():
        for idx, x in enumerate(loop):
            image, (x_shape, y_shape), (xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1), original_image = x[0], x[1], x[2], x[3]
            x_shape, y_shape, xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = x_shape.item(), y_shape.item(), xmin_mod_1.item(), ymin_mod_1.item(), xmax_mod_1.item(), ymax_mod_1.item()
            image = image.to(config.DEVICE)
            toon_face = gen2(image)
            image = image * 0.5 + 0.5
            toon_face = toon_face * 0.5 + 0.5
            actual_x_dim, actual_y_dim = (xmax_mod_1 - xmin_mod_1), (ymax_mod_1 - ymin_mod_1)
            toon_face = toon_face.permute(0,2,3,1)
            toon_face = toon_face.cpu().detach().numpy()
            toon_face = toon_face[0]
            # toon_face = cv2.resize(toon_face, (actual_x_dim, actual_y_dim))

            image = image.permute(0,2,3,1)
            image = image.cpu().detach().numpy()
            image = image[0]
            # image = cv2.resize(image, (actual_x_dim, actual_y_dim))

            concat_array = np.concatenate((image,toon_face),axis=0)
            concat_array = concat_array * 255.0
            concat_array = cv2.normalize(concat_array, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            concat_array = cv2.cvtColor(concat_array, cv2.COLOR_BGR2RGB)
            saved_file = str(save_dir) + f"/best_val_model_{idx}.png"
            cv2.imwrite(saved_file, concat_array)
    print("Test results using best val model saved in ",save_dir," !!")

    # ############### Loading least loss SWA Model #################
    # file_path = f"./saved_models/min_swa_loss_"+config.CHECKPOINT_GEN
    # gen2 = Generator(in_channels=3)
    # gen2 = load_checkpoint(
    #     file_path, gen2
    # )
    # gen2 = gen2.to(config.DEVICE)
    # gen2.eval()
    # loop = tqdm(test_dl)
    # with torch.no_grad():
    #     for idx, x in enumerate(loop):
    #         image, (x_shape, y_shape), (xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1), original_image = x[0], x[1], x[2], x[3]
    #         x_shape, y_shape, xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = x_shape.item(), y_shape.item(), xmin_mod_1.item(), ymin_mod_1.item(), xmax_mod_1.item(), ymax_mod_1.item()
    #         image = image.to(config.DEVICE)
    #         toon_face = gen2(image)
    #         image = image * 0.5 + 0.5
    #         toon_face = toon_face * 0.5 + 0.5
    #         actual_x_dim, actual_y_dim = (xmax_mod_1 - xmin_mod_1), (ymax_mod_1 - ymin_mod_1)
    #         toon_face = toon_face.permute(0,2,3,1)
    #         toon_face = toon_face.cpu().detach().numpy()
    #         toon_face = toon_face[0]
    #         # toon_face = cv2.resize(toon_face, (actual_x_dim, actual_y_dim))

    #         image = image.permute(0,2,3,1)
    #         image = image.cpu().detach().numpy()
    #         image = image[0]
    #         # image = cv2.resize(image, (actual_x_dim, actual_y_dim))

    #         concat_array = np.concatenate((image,toon_face),axis=0)
    #         concat_array = concat_array * 255.0
    #         concat_array = cv2.normalize(concat_array, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    #         concat_array = cv2.cvtColor(concat_array, cv2.COLOR_BGR2RGB)
    #         saved_file = str(save_dir) + f"/best_swa_model_{idx}.png"
    #         cv2.imwrite(saved_file, concat_array)
    # print("Test results using best val model saved in ",save_dir," !!")

def SWA_Model(swa_model, netD, val_dl, L2_Loss, BCE_Loss, Identity_loss):
    swa_model.eval()
    netD.eval()
    loop = tqdm(val_dl)
    swa_loss = []
    with torch.no_grad():
        for idx, (x,y) in enumerate(loop):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            y_fake = swa_model(x)
            D_real = netD(x,y)
            D_fake = netD(x, y_fake)
            D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
            D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
            Dis_loss = (D_real_loss + D_fake_loss)/2

            G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
            L2 = L2_Loss(y_fake, y)
            Identity = Identity_loss(y_fake,y)
            Gen_loss = G_fake_loss*config.CE_LAMBDA + L2*config.L2_LAMBDA - Identity*config.IDENTITY_LAMBDA
            swa_loss.append(Gen_loss)

            #++++++++++++++++++++++++ WANDB ++++++++++++++++++++++++#
            wandb.log({
                'swa_disc_loss':Dis_loss.item(), 
                'swa_gen_loss':Gen_loss.item(), 
                'swa_BCELoss':G_fake_loss.item(),
                'swa_L2Loss':L2.item(),
                'swa_IdentityLoss':Identity.item()
                })
    loss = torch.stack(swa_loss).mean().item()
    print("[INFO] Validation loss :- ",loss)
    netD.train()
    swa_model.train()
    return loss


def Validation(netG, netD, val_dl, L2_Loss, BCE_Loss, Identity_loss):
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
            val_loss.append(Gen_loss)

            #++++++++++++++++++++++++ WANDB ++++++++++++++++++++++++#
            wandb.log({
                'val_disc_loss':Dis_loss.item(), 
                'val_gen_loss':Gen_loss.item(), 
                'val_BCELoss':G_fake_loss.item(),
                'val_L2Loss':L2.item(),
                'val_IdentityLoss':Identity.item()
                })
    loss = torch.stack(val_loss).mean().item()
    print("[INFO] Validation loss :- ",loss)
    netD.train()
    netG.train()
    return loss

def train(netG, netD, train_dl, OptimizerG, OptimizerD, L2_Loss, BCE_Loss, Identity_loss, scheduler, epoch):
    loop = tqdm(train_dl)
    train_loss = []
    iters = len(train_dl)
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
        # D_loss = D_real_loss + D_fake_loss
        
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
        
        OptimizerG.zero_grad()
        Gen_loss.append(G_loss.item())
        G_loss.backward()
        OptimizerG.step()
        train_loss.append(G_loss)

        scheduler.step(epoch + idx/ iters)
        
        #++++++++++++++++++++++++++ WANDB ++++++++++++++++++++++#
        wandb.log({
            "train_Gen_loss": G_loss.item(), 
            "train_Disc_loss":D_loss.item(),
            "train_BCELoss":G_fake_loss.item(),
            "train_L2Loss":L2.item(),
            "train_IdentityLoss":L_identity.item()
            })
        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
    loss = torch.stack(train_loss).mean().item()
    print("[INFO] Training loss :- ",loss)
    return loss


def main():
    best_train_loss = 100.0
    best_val_loss = 100.0
    best_swa_loss = 100.0

    netD = Discriminator(in_channels=3)
    netG = Generator(in_channels=3)

    #+++++++++++++++++++++++++ Data Parallelism +++++++++++++++++#
    netG = nn.DataParallel(netG, device_ids=[0,1,2,3])
    netD = nn.DataParallel(netD, device_ids=[0,1,2,3])

    netD = netD.to(config.DEVICE)
    netG = netG.to(config.DEVICE)

    OptimizerD = torch.optim.Adam(netD.parameters(),lr=config.LEARNING_RATE,betas=(config.BETA1,0.999))
    OptimizerG = torch.optim.Adam(netG.parameters(),lr=config.LEARNING_RATE,betas=(config.BETA1,0.999))
    
    #-------------------------- learning rate schedule----------------------#
    # swa_model = torch.optim.swa_utils.AveragedModel(netG)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(OptimizerG, T_max=config.NUM_EPOCHS)
    # swa_scheduler = torch.optim.swa_utils.SWALR(OptimizerG, swa_lr= config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(OptimizerG, T_0=config.SWA_START, T_mult=1, eta_min=5e-3, verbose=True)

    # print(swa_model)
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
    eval_dl = DataLoader(val_dataset,batch_size=8,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=True)
    test_dataset = Test_Faces(root=config.TEST_DIR)
    test_dl = DataLoader(test_dataset,batch_size=1,shuffle=False)
    
    
    for epoch in range(config.NUM_EPOCHS):
        print("[INFO] EPOCH :- ",epoch+1)

        print("++++++++++++++++++++++++++++ Training ++++++++++++++++++++++")
        current_train_loss = train(
            netG, netD, train_dl, OptimizerG, OptimizerD, L2_Loss, BCE_Loss, Identity_loss, scheduler, epoch
        )

        #-------------------------------------------------#
        #-------------- Train Epoch Analysis--------------#
        #-------------------------------------------------#

        if current_train_loss<best_train_loss:
            best_train_loss = current_train_loss  #Minimum train loss
            if config.SAVE_MODEL:
                file_name = f"./saved_models/min_train_loss_"+config.CHECKPOINT_GEN
                temp_gen = netG
                temp_gen = temp_gen.to(config.DEVICE)
                netG = netG.to(config.DEVICE)
                # model_scripted = torch.jit.script(temp_gen)
                # model_scripted.save(file_name_pt)
                save_checkpoint(netG, OptimizerG, filename=file_name)
                print(f"[INFO] Saved Generator with min training loss after {epoch+1} epochs!!!!")
            
        else:
            print("[INFO] Not the min training loss!!!")
        

        #-------------------------------------------------#
        #-------------- Val Epoch Analysis----------------#
        #-------------------------------------------------#
        print("++++++++++++++++++++++++++++ Validation ++++++++++++++++++++++")
        current_val_loss = Validation(netG, netD, val_dl, L2_Loss, BCE_Loss, Identity_loss)
        if current_val_loss<best_val_loss:
            best_val_loss = current_val_loss #Minimum validation loss
            if config.SAVE_MODEL:
                file_name = f"./saved_models/min_val_loss_"+config.CHECKPOINT_GEN
                temp_gen = netG
                temp_gen = temp_gen.to(config.DEVICE)
                netG = netG.to(config.DEVICE)
                # model_scripted = torch.jit.script(temp_gen)
                # model_scripted.save(file_name_pt)
                save_checkpoint(netG, OptimizerG, filename=file_name)
                print("[INFO] Saved best Val performing generator!!!!")
            
        else:
            print("[INFO] Not the min validation loss")

        
        #+++++++++++++++++++++++++++++++ WANDB +++++++++++++++++++++++++++#
        wandb.log({
            "train_loss":current_train_loss,
            "val_loss":current_val_loss
        })

        if config.SAVE_MODEL and (epoch+1)%50==0:
            save_checkpoint(netG, OptimizerG, filename=config.CHECKPOINT_GEN)
            save_checkpoint(netD, OptimizerD, filename=config.CHECKPOINT_DISC)
        
        
        #-------------------------------------------------#
        #------------ SWA Model Evaluation ---------------#
        #-------------------------------------------------#
        # if (epoch+1)%5==0:
        #     swa_loss = SWA_Model(swa_model, netD, val_dl, L2_Loss, BCE_Loss, Identity_loss)

        #     if swa_loss < best_swa_loss:
        #         best_swa_loss = swa_loss
        #         if config.SAVE_MODEL:
        #             file_name = f"./saved_models/min_swa_loss_"+config.CHECKPOINT_GEN
        #             temp_gen = netG
        #             temp_gen = temp_gen.to(config.DEVICE)
        #             netG = netG.to(config.DEVICE)
        #             # model_scripted = torch.jit.script(temp_gen)
        #             # model_scripted.save(file_name_pt)
        #             save_checkpoint(netG, OptimizerG, filename=file_name)
        #             print(f"[INFO] Saved SWA Generator with min loss after {epoch+1} epochs!!!!")

        
        #-------------------------------------------------#
        #-------------------- Testing --------------------#
        #-------------------------------------------------#

        if (epoch+1)%10==0:
            print("\n")
            print("Testing on Real Faces...")
            save_dir = f"./test_results/{epoch+1}_results"
            Exists = os.path.exists(save_dir)
            if not Exists:
                os.makedirs(save_dir)
            test(test_dl, save_dir)
            

        if (epoch+1)%5==0:
            save_some_examples(netG, eval_dl, epoch+1, folder="evaluation")
            # save_some_examples(swa_model, eval_dl, epoch+1, folder="evaluation/swa")
        
        # if (epoch+1) > config.SWA_START:
        #     swa_model.update_parameters(netG)
        #     swa_scheduler.step()
        # else:
        #     scheduler.step()
        


    
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
        CE_LAMBDA = 1,
        IDENTITY_LAMBDA = 0,
        LAMBDA_GP = config.LAMBDA_GP,
        NUM_EPOCHS = config.NUM_EPOCHS,
        SWA_START = config.SWA_START
    )
    wandb.init(project='Toonify-GAN',config=hyper_parameters)
    main()
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(Gen_loss,label="Generator")
    plt.plot(Dis_loss,label="Discriminator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss.png')
