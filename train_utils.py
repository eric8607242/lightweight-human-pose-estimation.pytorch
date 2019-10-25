import sys
import cv2
import torch

from val import evaluate

from config import CONFIG

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
                concate_output = torch.cat((target_output, source_output), 0)
                concate_output = concate_output.to(self.device)

                predicted_output = self.discriminator(concate_output)

                source_label = torch.ones(source_output.size(0))
                target_label = torch.zeros(target_output.size(0))
                concate_label = torch.cat((source_label, target_label), 0).to(self.device)
                concate_label = concate_label.long()

                discriminator_loss = self.criterion(predicted_output, concate_label)
                discriminator_loss.backward()

                self.optimizer_d.step()

                predict_class = predicted_output.argmax(1).long()
                acc = predict_class.eq(concate_label)
                acc = acc.float().mean()


                self.criterion.zero_grad()
                self.target_net.zero_grad()
                self.source_net.zero_grad()
                self.discriminator.zero_grad()

                target_output = self.target_net(target_data)
                predicted_output = self.discriminator(target_output)
                
                source_label = torch.ones(source_output.size(0)).to(self.device)

                generator_loss = self.criterion(predicted_output, source_label.long())
                generator_loss.backward()

                self.optimizer_tg.step()

                if((step + 1) % 5 == 0):
                    torch.save({"state_dict":self.target_net.state_dict()}, CONFIG["training_setting"]["save_target_model"])
                    torch.save(self.discriminator, CONFIG["training_setting"]["save_discriminator"])
                    print("Epoch [{}/{}] Step {}:"
                          " d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                          .format(epoch+1, 
                                  CONFIG["training_setting"]["epochs"], 
                                  step+1, 
                                  discriminator_loss.item(),
                                  generator_loss.item(),
                                  acc.item()))
            self.target_net.zero_grad()
            #self.target_net.set_mode(False)
            #self._evaluate(self.target_net)
            self.target_net.set_mode(True)


    def _evaluate(self, model):
        num_refinement_stages = 1
        batches_per_iter = 1
        model.eval()
        model.set_mode(False)

        evaluate(self.val_labels, self.val_output_name, self.val_images_folder, model)
        

