# from utils import load_checkpoint
# import torch
# import config
# from models import Generator
# from dataset import Test_Faces
# from torch.utils.data import DataLoader

# # gen = Generator(in_channels=3)
# # optm = torch.optim.Adam(gen.parameters(),lr=config.LEARNING_RATE,betas=(config.BETA1,0.999))
# # file_path = f"./saved_models/min_train_loss_"+config.CHECKPOINT_GEN
# # gen = load_checkpoint(
# #     file_path,gen,optm,config.LEARNING_RATE
# # )
# # gen.eval()
# # temp = torch.randn((1,3,256,256))
# # y = gen(temp)
# # print(y.shape)

# dataset = Test_Faces("./toon_pix_data/test/")
# loader = DataLoader(dataset, batch_size=1)

# for x,_,_,_ in loader:
#     print(x.shape)



import wandb

wandb.login(key='46f2d6a5ffcc458fed2cca6cf446900f97e396e1')
wandb.init(project='Temp_Login')