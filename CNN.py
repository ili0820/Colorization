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



class CNN(nn.Module):
    def __init__(self):

        super().__init__()

        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1_1", nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv1.add_module("relu1_1", nn.ReLU(True))
        self.conv1.add_module("conv1_2", nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True))
        self.conv1.add_module("relu1_2", nn.ReLU(True))
        self.conv1.add_module("BatcNorm1",nn.BatchNorm2d(64))

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2_1", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv2.add_module("relu2_1", nn.ReLU(True))
        self.conv2.add_module("conv2_2", nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True))
        self.conv2.add_module("relu2_2", nn.ReLU(True))
        self.conv2.add_module("BatcNorm2",nn.BatchNorm2d(128))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3_1", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv3.add_module("relu3_1", nn.ReLU(True))
        self.conv3.add_module("conv3_2", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv3.add_module("relu3_2", nn.ReLU(True))
        self.conv3.add_module("conv3_3", nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True))
        self.conv3.add_module("relu3_3", nn.ReLU(True))
        self.conv3.add_module("BatcNorm3",nn.BatchNorm2d(256))

        self.conv4 = nn.Sequential()
        self.conv4.add_module("conv4_1", nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv4.add_module("relu4_1", nn.ReLU(True))
        self.conv4.add_module("conv4_2", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv4.add_module("relu4_2", nn.ReLU(True))
        self.conv4.add_module("conv4_3", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv4.add_module("relu4_3", nn.ReLU(True))
        self.conv4.add_module("BatcNorm4",nn.BatchNorm2d(512))

        self.conv5 = nn.Sequential()
        self.conv5.add_module("conv5_1", nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True))
        self.conv5.add_module("relu5_1", nn.ReLU(True))
        self.conv5.add_module("conv5_2", nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True))
        self.conv5.add_module("relu5_2", nn.ReLU(True))
        self.conv5.add_module("conv5_3", nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True))
        self.conv5.add_module("relu5_3", nn.ReLU(True))
        self.conv5.add_module("BatcNorm5",nn.BatchNorm2d(512))

        self.conv6 = nn.Sequential()
        self.conv6.add_module("conv6_1", nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True))
        self.conv6.add_module("relu6_1", nn.ReLU(True))
        self.conv6.add_module("conv6_2", nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True))
        self.conv6.add_module("relu6_2", nn.ReLU(True))
        self.conv6.add_module("conv6_3", nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True))
        self.conv6.add_module("relu6_3", nn.ReLU(True))
        self.conv6.add_module("BatcNorm6",nn.BatchNorm2d(512))

        self.conv7 = nn.Sequential()
        self.conv7.add_module("conv7_1", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv7.add_module("relu7_1", nn.ReLU(True))
        self.conv7.add_module("conv7_2", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv7.add_module("relu7_2", nn.ReLU(True))
        self.conv7.add_module("conv7_3", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv7.add_module("relu7_3", nn.ReLU(True))
        self.conv7.add_module("BatcNorm7",nn.BatchNorm2d(512))

        self.conv8 = nn.Sequential()
        self.conv8.add_module("deconv8_1", nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True))
        self.conv8.add_module("relu8_1", nn.ReLU(True))
        self.conv8.add_module("conv8_2", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv8.add_module("relu8_2", nn.ReLU(True))
        self.conv8.add_module("conv8_3", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv8.add_module("relu8_3", nn.ReLU(True))
        self.conv8.add_module("conv8_3", nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True))

        self.conv9 = nn.Sequential()
        self.conv9.add_module("deconv9_1", nn.ConvTranspose2d(313, 256, kernel_size=4, stride=2, padding=1, bias=True))
        self.conv9.add_module("relu9_1", nn.ReLU(True))
        self.conv9.add_module("deconv9_2", nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True))
        self.conv9.add_module("relu9_2", nn.ReLU(True))


        self.model_out = nn.Conv2d(256, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)


    def forward(self,x):
        conv1_2 = self.conv1(x)
        conv2_2 = self.conv2(conv1_2)
        conv3_3 = self.conv3(conv2_2)
        conv4_3 = self.conv4(conv3_3)
        conv5_3 = self.conv5(conv4_3)
        conv6_3 = self.conv6(conv5_3)
        conv7_3 = self.conv7(conv6_3)
        conv8_3 = self.conv8(conv7_3)
        conv9_3 = self.conv9(conv8_3)
        out_reg = self.model_out(conv9_3)

        return out_reg








def train(config):
    model=CNN().cuda()
    train_dataset = ColoringDataset(root, 'train', transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    loss_func=nn.MSELoss()
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
            cost = loss_func(prediction,y_train)
            cost.backward()
            optimizer.step()
            costs.append(cost.data.item())
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=np.mean(costs))

        if epoch%10==0:
            torch.save(model.state_dict(), os.path.join(config["output_dir"], "base_epoch_{0:d}.pt".format(epoch)))




def test(config):
    val_dataset = ColoringDataset(root, 'test', transform=ToTensor())
    val_dataloader = DataLoader(val_dataset,batch_size=config["batch_size"], shuffle=False)
    model=CNN().cuda()
    model.load_state_dict(torch.load("output/base_epoch_0.pt"))

    model.eval()
    result_dir = "output/results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    cnt=0


    with torch.no_grad():
        for (step, batch) in enumerate(val_dataloader):
            # .cuda()를 통해 메모리에 업로드

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
                #
                # plt.imshow(label)
                # plt.show()



if(__name__=="__main__"):
    root = "srdata/"
    output_dir ="output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = {"mode": "test",
              "model_name":"epoch_{0:d}.pt".format(10),
              "output_dir":output_dir,
              "learn_rate":0.001,
              "batch_size":16,
              "epoch":50,
              }

    if(config["mode"] == "train"):
        train(config)
    else:
        test(config)