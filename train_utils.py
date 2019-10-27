import sys
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms

from val import evaluate
from datasets.coco import CocoTrainDataset
from datasets.transformations import ConvertKeypoints

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
        self.val_dataloader = DataLoader(val_datasets, batch_size=CONFIG["training"]["batch_size"], shuffle=True, num_workers=0)

        self.device = device

    def train(self):
        self.source_net.eval()
        self.target_net.train()
        self.discriminator.train()

        for epoch in range(CONFIG["training_setting"]["epochs"]):
            #self._evaluate(self.target_net)
            self.source_net.set_mode(True)
            self.target_net.set_mode(True)
            for step, X in enumerate(self.dataloader):

                self.criterion.zero_grad()
                self.target_net.zero_grad()
                self.source_net.zero_grad()
                self.discriminator.zero_grad()

                source_data = X["source"].cuda(CONFIG["GPU"]["source_net"])
                target_data = X["target"].cuda(CONFIG["GPU"]["target_net"])

                target_output = self.target_net(target_data)
                with torch.no_grad():
                    source_output = self.source_net(source_data)

                source_predict = self.discriminator(source_output)
                target_predict = self.discriminator(target_output)

                source_label = Variable(Tensor(source_data.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
                target_label = Variable(Tensor(source_data.shape[0], 1).fill_(0.0), requires_grad=False).to(self.device)


                real_loss = self.criterion(source_predict, source_label)
                fake_loss = self.criterion(target_predict, target_label)

                discriminator_loss = (real_loss + fake_loss) / 2
                discriminator_loss.backward()

                self.optimizer_d.step()


                self.criterion.zero_grad()
                self.target_net.zero_grad()
                self.source_net.zero_grad()
                self.discriminator.zero_grad()

                target_output = self.target_net(target_data)
                predicted_output = self.discriminator(target_output)
                
                source_label = Variable(Tensor(source_data.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)

                generator_loss = self.criterion(predicted_output, source_label)
                generator_loss.backward()

                self.optimizer_tg.step()

                if((step + 1) % 5 == 0):
                    torch.save({"state_dict":self.target_net.state_dict()}, CONFIG["training_setting"]["save_target_model"])
                    torch.save(self.discriminator, CONFIG["training_setting"]["save_discriminator"])
                    print("Epoch [{}/{}] Step {}:"
                          " d_loss={:.5f} g_loss={:.5f}"
                          .format(epoch+1, 
                                  CONFIG["training_setting"]["epochs"], 
                                  step+1, 
                                  discriminator_loss.item(),
                                  generator_loss.item(),
                                  ))
            self.target_net.zero_grad()
            #self.target_net.set_mode(False)
            #self._evaluate(self.target_net)
            self.target_net.set_mode(True)


    def _evaluate(self, model):
        num_refinement_stages = 1
        batches_per_iter = 1
        model.eval()
        model.set_mode(False)

        total_losses = [0, 0] * (2)

        for batch_data in self.val_dataloader:
            images = batch_data["image"]
            keypoint_masks = batch_data["keypoint_mask"]
            paf_masks = batch_data["paf_masks"]
            keypoint_maps = batch_data["keypoint_maps"]
            paf_maps = batch_data["paf_maps"]

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

        print("Valdating Step"
                " Loss = {}".format(loss))

        #evaluate(self.val_labels, self.val_output_name, self.val_images_folder, model)
        

