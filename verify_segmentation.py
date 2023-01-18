import cv2
import os
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, utils

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

class Segmnetation(nn.Module):
    def __init__(self, model, Trans):
        super(Segmnetation, self).__init__()
        self.model = model#.backbone
        self.trans = Trans

    def forward(self, x):
        map = x
        map = TF.resize(map, (520, 520), interpolation=TF.InterpolationMode.BILINEAR)
        map = self.trans(map)
        map = self.model(map)
        map = map['out']
        return map


weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
model = lraspp_mobilenet_v3_large( weights=weights)
Trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = model.eval()

for param in model.parameters():
    param.requires_grad = False

operation_func = Segmnetation(model, Trans)
img_transform = transforms.Compose([transforms.ToTensor()])


img_ind = 0
folder = './seg_expt/text_type_2'

file_name = folder + '/res.txt'

if not os.path.exists(file_name):
    open(file_name, "w").close()

class node:
    def __init__(self, sim, img):
        self.sim = sim
        self.img = img

with open(file_name, "w") as file:
    for img_ind in range(8):
        print("Analysis for img_ind : ", img_ind)

        file.write(f"Analysis for img_ind : {img_ind}")
        file.write("\n")

        All = []
        for i in range(1, 16):

            for j in range(2):
                evaluator = Evaluator(21)

                path_1 = folder + f'/check_{i}/new_img_{img_ind}_{j}.png'
                path_2 = folder + f'/check_{i}/og_img_{img_ind}.png'

                try:
                    img_1 = Image.open(path_1)
                    img_2 = Image.open(path_2)

                    img_tensor_1 = img_transform(img_1)
                    img_tensor_2 = img_transform(img_2)

                    img_tensor_1 = torch.unsqueeze(img_tensor_1, 0)
                    img_tensor_2 = torch.unsqueeze(img_tensor_2, 0)

                    seg_1 = operation_func(img_tensor_1)
                    seg_1 = seg_1.data.cpu().numpy()
                    seg_1 = np.argmax(seg_1, axis=1)

                    seg_2 = operation_func(img_tensor_2)
                    seg_2 = seg_2.data.cpu().numpy()
                    seg_2 = np.argmax(seg_2, axis=1)

                    evaluator.add_batch(seg_1, seg_2)
                    mIoU = evaluator.Mean_Intersection_over_Union()
                    print(mIoU)

                    All.append(node(sim=mIoU, img=path_1))

                except:
                    pass


                # print(path_1)
                #

            # exit()

        All = sorted(All, key=lambda x: 1 - x.sim)

        og_img = folder + f'/check_1/og_img_{img_ind}.png'
        og_img = cv2.imread(og_img)
        og_img = cv2.resize(og_img, (512, 512))
        best_imgs = [og_img]

        for i in range(min(len(All), 10)):
            print(All[i].img)
            print(All[i].sim)

            file.write(All[i].img)
            file.write("\n")

            file.write(str(All[i].sim))
            file.write("\n")

            img = cv2.imread(All[i].img)
            best_imgs.append(img)

        best_imgs = cv2.hconcat(best_imgs)
        uq = folder.split("/")[1]
        cv2.imwrite(f'{folder}/all_{uq}_{img_ind}.png', best_imgs)
