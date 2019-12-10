import os
import sys
import torch
import torchvision
import model
import numpy as np
from PIL import Image


VOC_PATH = "./VOCdevkit/VOC2012"

if len(sys.argv) < 2:
    exit(0)

ckpt_path = sys.argv[1]


def class2rgb(idx, rgb):
    def bit1(x, bit):
        return int(x & (1 << bit) != 0)
    if rgb == 'r':
        return 0x80*bit1(idx, 0) + 0x40*bit1(idx, 3)
    elif rgb == 'g':
        return 0x80*bit1(idx, 1) + 0x40*bit1(idx, 4)
    elif rgb == 'b':
        return 0x80*bit1(idx, 2) + 0x40*bit1(idx, 5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = model.UNet()
net.load_state_dict(torch.load(ckpt_path))
net = net.to(device)
net.eval()

toTensor = torchvision.transforms.ToTensor()
class2rgb_np = np.vectorize(class2rgb)

sum_accuracy = 0.0

list_f = open("./ImageSets/test.txt").readlines()
for line in list_f:
    img_path = os.path.join(VOC_PATH, "JPEGImages", line.replace('\n', '.jpg'))
    input_img = Image.open(img_path)
    input_img = toTensor(input_img).to(device)
    input_img = input_img.unsqueeze(0)

    img_path = os.path.join(VOC_PATH, "SegmentationClass", line.replace('\n', '.png'))
    gt_img = Image.open(img_path)
    gt_img = np.array(gt_img)   # [H, W]
    gt_img[gt_img==255] = 0

    out_seg = net(input_img)
    out_seg = out_seg.argmax(1)[0].to('cpu').numpy()    # [H, W]

    out_seg_r = np.expand_dims(class2rgb_np(out_seg, 'r'), -1)
    out_seg_g = np.expand_dims(class2rgb_np(out_seg, 'g'), -1)
    out_seg_b = np.expand_dims(class2rgb_np(out_seg, 'b'), -1)
    out_seg_rgb = np.concatenate([out_seg_r, out_seg_g, out_seg_b], axis=-1)
    out_img = Image.fromarray(out_seg_rgb.astype(np.uint8))
    out_img.save(os.path.join("./SegImages", line.replace('\n', '.png')))

    accuracy = np.sum(gt_img == out_seg) / gt_img.size
    sum_accuracy += accuracy
    print("%s: %f" % (line.replace('\n', ''), accuracy))

print("Average Accuracy: %f" % (sum_accuracy / len(list_f)))
