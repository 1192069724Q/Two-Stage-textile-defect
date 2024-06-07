import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from models import UNetModel
from models import CBAM_Unet

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
def create_unet(num_class, pretrain=False):
    return  UNetModel(in_channels=3,model_channels=32,out_channels=num_class,num_res_blocks=1)

def create_CBAMunet(num_class, pretrain=False):
    return  CBAM_Unet(in_channels=3,model_channels=32,out_channels=num_class,num_res_blocks=1)


def main():
    classes = 1

    # path of detection model
    weights_path = "./save_weights/12-7_cm/Z_model_440.pth"
    import os

    # dir of detected image
    img_path = "data\star/"
    ip=os.listdir(img_path)
    palette_path = "palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_unet(num_class=2,pretrain=False)
    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    for i in range(len(ip)):
        original_img = Image.open(img_path+ip[i])

    # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
        img = data_transform(original_img)
    # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
     # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))

            t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        #prediction = output['out'].argmax(1).squeeze(0)
        prediction = output.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)*255
        mask = Image.fromarray(prediction)
        print(ip[i])
        mask.save('data/Vis/'+ip[i])
    print()


if __name__ == '__main__':
    main()
