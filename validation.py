import os
import torch

from train_utils import evaluate
from my_dataset import VOCSegmentation
import transforms as T
from models import  UNetModel
from models import CBAM_Unet

def create_unet(num_class, pretrain=False):
    return  UNetModel(in_channels=3,model_channels=32,out_channels=num_class,num_res_blocks=1)

def create_CBAMunet(num_class, pretrain=False):
    return  CBAM_Unet(in_channels=3,model_channels=32,out_channels=num_class,num_res_blocks=1)

class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=SegmentationPresetEval(256),
                                  txt_name="val.txt")

    num_workers = 2
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)



    model= create_unet(num_class=2,pretrain=False)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    model.to(device)

    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
    print(confmat)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 validation")

    parser.add_argument("--data-path", default="data/", help="VOCdevkit root")
    parser.add_argument("--weights", default='')
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
