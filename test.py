import sys
import torch
import torchvision
import model
import numpy as np
from PIL import Image

if len(sys.argv) < 3:
    exit(0)

ckpt_path = sys.argv[1]
img_path = sys.argv[2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = model.UNet()
net.load_state_dict(torch.load(ckpt_path))

input_img = Image.open(img_path)
toTensor = torchvision.transforms.ToTensor()
input_img = toTensor(input_img)
#normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#input_img = normalize(toTensor(input_img))
input_img = input_img.unsqueeze(0)

out_seg = net(input_img)
out_seg = out_seg.argmax(1)[0]
print(out_seg)

def class2rgb(idx, rgb):
    if rgb == 'r':
        return 0x80*(idx&0x01) + 0x40*(idx&0x08)
    elif rgb == 'g':
        return 0x80*(idx&0x02) + 0x40*(idx&0x10)
    elif rgb == 'b':
        return 0x80*(idx&0x04) + 0x40*(idx&0x20)

class2rgb_np = np.vectorize(class2rgb)

out_seg = out_seg.numpy()
out_seg_r = np.expand_dims(class2rgb_np(out_seg, 'r'), -1)
out_seg_g = np.expand_dims(class2rgb_np(out_seg, 'g'), -1)
out_seg_b = np.expand_dims(class2rgb_np(out_seg, 'b'), -1)
out_seg = np.concatenate([out_seg_r, out_seg_g, out_seg_b], axis=-1)

out_seg = out_seg.astype(np.uint8)
out_img = Image.fromarray(out_seg)
out_img.save('test.png')

