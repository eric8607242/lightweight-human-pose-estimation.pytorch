import numpy as np

CONFIG = {
    "CUDA" : True,
    "GPU":{
        "source_net" : 0,
        "target_net" : 0
    },
    "model" : {
        "pretrained_weight" : "./models/checkpoint_iter_370000.pth",
    },
    "dataset" : {
        "input_size" : 256,
        "batch_size" : 64,
        "source_img_folder" : "./data/rgb/",
        "target_img_folder" : "./data/ther/",
        "val_labels" : "./data/annotation/training.json",
        "val_output_name" : "./d.json",
        "val_images_folder" : "./data/images/"
    },
    "training_setting" : {
        "epochs" : 1000,
        "t_lr" : 2e-4,
        "d_lr" : 1e-4,
        "save_target_model" : "./models/target_model.pth",
        "save_discriminator" : "./models/discriminator_model.pth"
    }



}
