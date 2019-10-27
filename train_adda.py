import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

from train_utils import Trainer
from addadataset import ADDADataset
from discriminator import Discriminator

from config import CONFIG


def main():
    if CONFIG["CUDA"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    weight_name = CONFIG["model"]["pretrained_weight"]
    model_dict = torch.load(weight_name)
    
    source_net = PoseEstimationWithMobileNet()
    target_net = PoseEstimationWithMobileNet()

    load_state(source_net, model_dict)
    load_state(target_net, model_dict)

    discriminator = Discriminator()
    criterion = nn.BCELoss()

    source_net = source_net.cuda(CONFIG["GPU"]["source_net"])
    target_net = target_net.cuda(CONFIG["GPU"]["target_net"])
    discriminator = discriminator.to(device)
    criterion = criterion.to(device)

    optimizer_tg = torch.optim.Adam(target_net.parameters(),
                                   lr=CONFIG["training_setting"]["t_lr"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(),
                                  lr=CONFIG["training_setting"]["d_lr"])

    dataset = ADDADataset()
    dataloader = DataLoader(dataset, CONFIG["dataset"]["batch_size"], shuffle=True, num_workers=0)

    trainer = Trainer(source_net, target_net, discriminator, 
                     dataloader, optimizer_tg, optimizer_d, criterion, device)
    trainer.train()


if __name__ == "__main__":
    main()
