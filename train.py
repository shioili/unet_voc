import sys
import torch
import torch.nn as nn
import torchsummary
import torchvision
import voc_loader
import model
import ckpt_saver

if len(sys.argv) < 3:
    print("./train.py tag num_epochs")
    exit(0)

tag = sys.argv[1]
num_epochs = int(sys.argv[2])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

crop_size = (128, 128)

dataloader = voc_loader.get_dataloader(30, crop_size=crop_size, shuffle=True)

net = model.UNet()
net = net.to(device)
torchsummary.summary(net, (3, crop_size[0], crop_size[1]))

csaver = ckpt_saver.CkptSaver(net, tag=tag)

ce_loss = nn.CrossEntropyLoss()
ce_loss = ce_loss.to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        img, seg = data
        img, seg = img.to(device), seg.to(device)

        outputs = net(img)
        loss = ce_loss(outputs, seg)

        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss

    print('[%d] loss: %.3f' % (epoch+1, running_loss / len(dataloader)))
    if epoch % 10 == 0:
        csaver.save(epoch)

csaver.save(num_epochs)

