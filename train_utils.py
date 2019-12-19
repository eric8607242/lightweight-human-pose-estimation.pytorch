import sys
import json
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from val import evaluate
from datasets.coco import CocoTrainDataset
from datasets.transformations import ConvertKeypoints
from modules.loss import l2_loss

from config import CONFIG
Tensor = torch.cuda.FloatTensor

class Trainer:
    def __init__(self, source_net, target_net, discriminator, 
            dataloader, optimizer_tg, optimizer_d, criterion, device):
        self.source_net = source_net
        self.target_net = target_net

        self.discriminator = discriminator

        self.dataloader = dataloader

        self.optimizer_tg = optimizer_tg
        self.optimizer_d = optimizer_d

        self.criterion = criterion

        self.val_labels = CONFIG["dataset"]["val_labels"]
        self.val_images_folder = CONFIG["dataset"]["val_images_folder"]
        self.val_output_name = CONFIG["dataset"]["val_output_name"]

        val_datasets = CocoTrainDataset(self.val_labels, self.val_images_folder, 8, 7, 1, 
                                               transform=transforms.Compose([
                                                   ConvertKeypoints()
                                                   ]))
        self.val_dataloader = DataLoader(val_datasets, batch_size=CONFIG["dataset"]["batch_size"], shuffle=True, num_workers=0)

        self.device = device
        self.pafs_loss_info = []
        self.heatmaps_loss_info = []

        self.lowest_loss = 10000

    def train(self):
        self.source_net.eval()
        self.target_net.train()
        self.discriminator.train()
        loss = self._evaluate(self.target_net)
        loss_f = nn.BCELoss()
        g_loss = []
        d_loss = []

        for epoch in range(CONFIG["training_setting"]["epochs"]):
            #self._evaluate(self.target_net)
            self.target_net.train()
            self.discriminator.train()
            self.source_net.set_mode(True)
            self.target_net.set_mode(True)
            for step, X in enumerate(self.dataloader):
                source_data = X["source"].cuda(CONFIG["GPU"]["source_net"])
                target_data = X["target"].cuda(CONFIG["GPU"]["target_net"])

                self.criterion.zero_grad()
                self.target_net.zero_grad()
                self.source_net.zero_grad()
                self.discriminator.zero_grad()

                target_output = self.target_net(target_data)
                predicted_output = self.discriminator(target_output)
                
                loss_f = nn.BCELoss()
                source_label = Variable(Tensor(source_data.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
                generator_loss = loss_f(predicted_output, source_label)
                generator_loss.backward()

                self.optimizer_tg.step()
                
                self.criterion.zero_grad()
                self.target_net.zero_grad()
                self.source_net.zero_grad()
                self.discriminator.zero_grad()

                target_output = self.target_net(target_data)
                with torch.no_grad():
                    source_output = self.source_net(source_data)

                source_predict = self.discriminator(source_output)
                target_predict = self.discriminator(target_output)

                source_label = Variable(Tensor(source_data.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
                target_label = Variable(Tensor(source_data.shape[0], 1).fill_(0.0), requires_grad=False).to(self.device)
                discriminator_loss = (loss_f(source_predict, source_label) + loss_f(target_predict, target_label))/2
                discriminator_loss.backward()
                
                g_loss.append(generator_loss.item())
                d_loss.append(discriminator_loss.item())
                
                self.optimizer_d.step()
                
                if((step + 1) % 5 == 0):
                    print("Epoch [{}/{}] Step {}:"
                          " d_loss={:.5f} g_loss={:.5f}"
                          .format(epoch+1, 
                                  CONFIG["training_setting"]["epochs"], 
                                  step+1, 
                                  discriminator_loss.item(),
                                  generator_loss.item(),
                                  ))
            self.criterion.zero_grad()
            self.target_net.zero_grad()
            self.source_net.zero_grad()
            self.discriminator.zero_grad()
            loss = self._evaluate(self.target_net)
            if loss <= self.lowest_loss:
                print("Save!")
                torch.save({"state_dict":self.target_net.state_dict()}, CONFIG["training_setting"]["save_target_model"])
                self.lowest_loss = loss

            self.target_net.set_mode(True)


    def _evaluate(self, model):
        num_refinement_stages = 1
        batches_per_iter = 1
        model.eval()
        model.set_mode(False)

        total_losses = [0, 0] * (2)
        total_iter = 0

        for i, batch_data in enumerate(self.val_dataloader):
            images = batch_data["image"].cuda()
            keypoint_masks = batch_data["keypoint_mask"].cuda()
            paf_masks = batch_data["paf_mask"].cuda()
            keypoint_maps = batch_data["keypoint_maps"].cuda()
            paf_maps = batch_data["paf_maps"].cuda()

            stages_output = model(images)

            losses = []
            for loss_idx in range(len(total_losses) // 2):
                losses.append(l2_loss(stages_output[loss_idx*2], keypoint_maps, keypoint_masks, images.shape[0]))
                losses.append(l2_loss(stages_output[loss_idx*2 + 1], paf_maps, paf_masks, images.shape[0]))

                total_losses[loss_idx*2] += losses[-2].item()
                total_losses[loss_idx*2+1] += losses[-1].item()

            loss = losses[0]
            for loss_idx in range(1, len(losses)):
                loss += losses[loss_idx]
                
            total_iter += 1


        for loss_idx in range(len(total_losses) // 2):
            print('\n'.join(['stage{}_pafs_loss:     {}', 'stage{}_heatmaps_loss: {}']).format(
                loss_idx + 1, total_losses[loss_idx * 2 + 1] / total_iter,
                loss_idx + 1, total_losses[loss_idx * 2] / total_iter))

        self.pafs_loss_info.append(total_losses[3] / total_iter)
        self.heatmaps_loss_info.append(total_losses[2] / total_iter)

        print("Valdating Step"
                " Loss = {}".format(total_losses[3] / total_iter))
        return total_losses[3] / total_iter

        

