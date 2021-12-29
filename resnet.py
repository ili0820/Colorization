import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision.transforms import ToTensor
import torch
from torch import nn
import numpy as np
import cv2
from skimage import color
from tqdm import tqdm





class ColoringDataset(Dataset):
    def __init__(self,root_dir,data_type='train',transform=None):
        data_paths=os.path.join(root_dir+data_type)
        filenames=os.listdir(data_paths)

        self.full_filenames=[os.path.join(data_paths,f) for f in filenames]
        self.transform=transform
        self.data_type=data_type


    def __len__(self):
        return len(self.full_filenames)
    def __getitem__(self,idx):
        image = io.imread(self.full_filenames[idx])
        if len(image.shape)==3:
            lchannel, achannel, bchannel = cv2.split(color.rgb2lab(image))
            label=cv2.merge((achannel,bchannel))
        else:
            label = cv2.merge((image, image, image))
            lchannel, _, _ = cv2.split(color.rgb2lab(label))
            label=image
        if self.transform:
            lchannel=self.transform(lchannel)
            label=self.transform(label)

        return lchannel,label

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
        super(Conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
      if not self.activation:
        return self.batchnorm(self.conv(x))
      return self.relu(self.batchnorm(self.conv(x)))

class ConvTrans_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
        super(ConvTrans_block, self).__init__()
        self.relu = nn.ReLU()
        self.convtrans = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.activation = activation

    def forward(self, x):
      return self.relu(self.convtrans(x))


class Res_block(nn.Module):
    def __init__(self, in_channels, red_channels, out_channels, is_plain=False):
        super(Res_block, self).__init__()
        self.relu = nn.ReLU()
        self.is_plain = is_plain

        if in_channels == 64:
            self.convseq = nn.Sequential(
                Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        elif in_channels == out_channels:
            self.convseq = nn.Sequential(
                Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Identity()
        else:
            self.convseq = nn.Sequential(
                Conv_block(in_channels, red_channels, kernel_size=1, padding=0, stride=2),
                Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)

            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        y = self.convseq(x)
        if self.is_plain:
            x = y
        else:
            x = y + self.iden(x)
        x = self.relu(x)  # relu(skip connection)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels=3,  is_plain=False):
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.
        super(ResNet, self).__init__()
        self.conv1 = Conv_block(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            Res_block(64, 64, 256, is_plain),
            Res_block(256, 64, 256, is_plain),
            Res_block(256, 64, 256, is_plain)
        )

        self.conv3_x = nn.Sequential(
            Res_block(256, 128, 512, is_plain),
            Res_block(512, 128, 512, is_plain),
            Res_block(512, 128, 512, is_plain),
            Res_block(512, 128, 512, is_plain)
        )

        self.conv4_x = nn.Sequential(
            Res_block(512, 256, 1024, is_plain),
            Res_block(1024, 256, 1024, is_plain),
            Res_block(1024, 256, 1024, is_plain),
            Res_block(1024, 256, 1024, is_plain),
            Res_block(1024, 256, 1024, is_plain),
            Res_block(1024, 256, 1024, is_plain)
        )

        self.conv5_x = nn.Sequential(
            Res_block(1024, 512, 2048, is_plain),
            Res_block(2048, 512, 2048, is_plain),
            Res_block(2048, 512, 2048, is_plain),
        )
        self.conv6_x = nn.Sequential(
            ConvTrans_block(2048, 1024, kernel_size=4, stride=2, padding=1),
            ConvTrans_block(1024, 512, kernel_size=4, stride=2, padding=1),
            ConvTrans_block(512, 256, kernel_size=4, stride=2, padding=1),
            ConvTrans_block(256, 128, kernel_size=4, stride=2, padding=1),
            ConvTrans_block(128, 64, kernel_size=4, stride=2, padding=1)

        )
        self.final = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(self.normalize_l(x))
        x = self.maxpool1(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.conv6_x(x)
        x = self.final(self.unnormalize_ab(x))

        return x

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm


    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm







def train(config):
    model=ResNet(in_channels=1).cuda()
    train_dataset = ColoringDataset(root, 'train', transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    mseloss=nn.MSELoss()
    l1loss=nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learn_rate"])
    num_epochs=config["epoch"]
    for epoch in range(num_epochs+1):
    # 학습 모드 셋팅
        model.train()
        costs = []
        loop= tqdm(enumerate(train_dataloader),total=len(train_dataloader))
        for (idx,batch) in loop:
            batch = (t.float().cuda() for t in batch)

            x_train,y_train=batch

            prediction = model(x_train)


            optimizer.zero_grad()
            cost = mseloss(prediction,y_train)

            cost.backward()
            optimizer.step()
            costs.append(cost.data.item())
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=np.mean(costs))

        if epoch%10==0:
            torch.save(model.state_dict(), os.path.join(config["output_dir"], "epoch_{0:d}.pt".format(epoch)))




def test(config):
    test_dataset = ColoringDataset(root, 'test', transform=ToTensor())
    test_dataloader = DataLoader(test_dataset,batch_size=config["batch_size"], shuffle=False)
    model=ResNet(in_channels=1).cuda()
    model.load_state_dict(torch.load("output/model/epoch_200.pt"))
    # 테스트 모드 셋팅
    model.eval()
    result_dir = "output/results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    cnt=0


    with torch.no_grad():
        for (step, batch) in enumerate(test_dataloader):

            batch = (t.float().cuda() for t in batch)

            input_features, labels = batch

            output = model(input_features).cpu()


            output=np.array(output)

            for idx in range(output.shape[0]):

                lchannel=np.array(input_features[idx].cpu())


                result=output[idx]

                label = cv2.merge([lchannel[0], result[0], result[1]])
                label=color.lab2rgb(label)
                filename=os.path.join(result_dir, "{}.jpg".format(cnt))
                plt.imsave(filename,label)
                cnt += 1




if(__name__=="__main__"):
    root = "srdata/"
    output_dir ="output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = {"mode": "test",
              "model_name":"epoch_{0:d}.pt".format(10),
              "output_dir":output_dir,
              "learn_rate":0.0001,
              "batch_size":8,
              "epoch":50,
              }

    if(config["mode"] == "train"):
        train(config)
    else:
        test(config)