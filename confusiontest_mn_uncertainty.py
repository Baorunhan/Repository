# -*- coding: utf-8 -*-
from encodings import utf_8, utf_8_sig
import pandas as pd
from tabnanny import check
from unittest import result
from kornia import feature
import torch
import numpy as np
import torchvision
from torchvision import transforms as transforms
import torchvision.transforms.functional as TF
from collections import OrderedDict
import torch.nn as nn
import time as ti
import argparse
from torch.autograd import Variable
from net0809 import ASP0809
from mobilenetv3 import MNV3_large2_uncertainty
import MyImgfuns as Mif
import datetime
import cv2
import glob
import itertools
import matplotlib.pyplot as plt
import scipy.io as scio
import os
from shutil import copyfile
import MylossF

args = None
from shutil import move
from math import exp
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# 绘制混淆矩阵
def random_str(slen=10):
    seed = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 在这些种子里面随机算slen个字符
    sa = []
    for i in range(slen):
        sa.append(random.choice(seed))
    return ''.join(sa)


def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument('-m', '--modelpath', default="", help='train dir')
    parser.add_argument('-v', '--validdir', default="", help='valid dir')
    parser.add_argument('-t', '--thr', default=32768, type=int, help='valid dir')  # 
    parser.add_argument('-f', '--flag', default="", help='special version')  # 
    parser.add_argument('-p', '--paral', default=0, type=int, help='special version')  # 
    parser.add_argument('-re', '--select_retry', default=False, type=bool, help='special version') 

    args = parser.parse_args(['-m', "./Models/model__2022_03_17_06_uncertainty/",#11
                             '-v', "../dataset/paoku/use/",
                             '-f',"uncertainty_paoku",'-t','32768','-p','1'])
	 
    # args = parser.parse_args()
    return args


class Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, transform_iaa=None, target_transform=None,
                 loader=None, is_valid_file=None, select_retry=False, randomselect=False, gt = None):
        super(Dataset, self).__init__(root, transform=transform, target_transform=target_transform,
                                      is_valid_file=is_valid_file)
        self.transform_iaa = transform_iaa
        self.select_retry = select_retry
        self.randomselect = randomselect
        self.selectnum = 150000
        self.gt = gt
        #筛选retry-X
        if self.select_retry:
            sequence_set = []
            for single in self.samples:
                if ('retry' in single[0]) and ('retry-0' in single[0]):
                    if self.gt is not None:
                        single[1] = self.gt
                    sequence_set.append(single)                           
                elif ('retry' in single[0]) == False:
                    sequence_set.append(single)
                else:
                    sequence_set.append(single)                      
            self.samples = sequence_set
            
        if self.randomselect:
            Toalindex = range(len(self.samples))
            indexvector = random.sample(Toalindex,self.selectnum)
            self.samples = self.samples[indexvector]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


if __name__ == '__main__':

    args = parse_args()
    log = []
    dis_vecrtor = []
    softmax = nn.Softmax(dim=1)
    time = "{}".format(datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M'))#不要随机编码
    print(time)

	#标志位#
    thr_mode = 1
    solo_mode = 1
    thr = args.thr
    paraflag = args.paral
    print("threshold:{}".format(thr))
    log.append("threshold:{}\n".format(thr))
    delFlag = 0#很关键
    teacher_flag = 0
    copyEmptyDir = 0
	#标志位#

    if delFlag:
        print("original picture has been moved!")		
	
	#预处理参数#
    cutside = 5
    Smallestscore = 65536
    Fakegreatestscore = 0
    # 测试集的路径#
    test_dir = args.validdir
    print("fp test dir:{}".format(test_dir))
    #'''
    test_transform = transforms.Compose([ 
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    # test_set = torchvision.datasets.ImageFolder(test_dir, test_transform)
    test_set = Dataset(test_dir, test_transform,select_retry=args.select_retry)
    print(" retry-x number for: %d"%(len(test_set.samples)))

    classnum = 2#图像种类个数
    appear_times = Variable(torch.zeros(classnum , 1))
    for label in test_set.targets:
        appear_times[label] += 1
    print(appear_times)#测试样本总数
    #数据集#

    device = torch.device('cuda')

    G = MNV3_large2_uncertainty(classnum).to(device)  # 初始化网络,算不确定度的话网络结构变化
    
    #这里写一个遍历函数
    def Loopcheck(arg,findnum,G,test_set,teacher_flag,thr_mode,resultlist):

        logprint = 0#是否写log
        savemat = 1
        savecsv = 1

        mpath = args.modelpath + 'ckpt_' +findnum+'.pth'
        checkpoint = torch.load(mpath)


        print("model path:{}".format(mpath))
        log.append("model path:{}\n".format(mpath))
            		
        new_state_dict = OrderedDict()

        print("Parallel Model Flag: %d" %(paraflag))
        for k, v in checkpoint['net'].items():  # 之前是“G”
            if args.paral == 1:
                name = k[7:]  # remove `module.`
            else:
                name = k[:]
            new_state_dict[name] = v
            #print(v.size())
			
        G.load_state_dict(new_state_dict)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)  # , pin_memory=True
        confusionmap = Variable(torch.zeros(classnum, classnum))
        G.eval()#固定网络，进行测试
        print('example')
        to_pil_image = transforms.ToPILImage()
        uncertainty_re_fp = []
        uncertainty_re_spoof = []
        uncertainty_error_fp = []
        uncertainty_error_spoof = []
        pathall_fp = []
        pathall_spoof = []
        with torch.no_grad():
            for batch_num, (data, target, path) in enumerate(val_loader):
                print("\r" + "{}/{}={:.2f}%".format(batch_num, len(val_loader), batch_num / len(val_loader) * 100), end="",
                        flush=True)

                data, target = data.to(device), target.to(device)
                # print(data[0] * 255)
                # print(target)
                output, feature, log_sigma2 = G(data)#这是v2的方式
                feature = feature.detach().cpu().numpy()#特征（均值）
                sigma_sq = torch.exp(log_sigma2).detach().cpu().numpy()#方差
                prediction = torch.max(output, 1)  # 这里输出的是预测
                finger_score = nn.Softmax(dim=1)(output)
                for i in range(len(target)):
                    confusionmap[target[i]][prediction.indices[i]] += 1  
                    uc = MylossF.aggregate_PFE(feature[i][:],sigma_sq[i][:])
                    if (prediction.indices[i]!=target[i]):
                        if(target[i]==0):
                            uncertainty_error_fp.append(uc)
                        else:
                            uncertainty_error_spoof.append(uc)                        
                        print("\nScore:%d uncertainty:%.10f path: %s \n"%(finger_score[i][0] * 65536, uc,path[i].encode('UTF-8', 'ignore').decode('UTF-8')))
                    else:
                        if(target[i]==0):
                            uncertainty_re_fp.append(uc)
                            pathall_fp.append(path[i])
                        else:
                            uncertainty_re_spoof.append(uc)
                            pathall_spoof.append(path[i])
        cm = confusionmap.numpy().astype('float') / confusionmap.sum(axis=1)[:, np.newaxis]
        log.append("TF: %1.5f FF: %1.5f " % (cm[0][0],cm[1][1]))
        print(cm)
        print(confusionmap)
        print("FP Avg True Uncertainty:%.10f"%(np.mean(np.array(uncertainty_re_fp))))
        print("Spoof Avg True Uncertainty:%.10f"%(np.mean(np.array(uncertainty_re_spoof))))

        if logprint:
            #f = open("../PictureOut/log.txt", 'a')
            spflag = "_"+args.flag
            confusion_out_path = "../testresult/confusion_test%s" % time + spflag
            if not os.path.isdir(confusion_out_path):#创建测试结果文件夹
                os.mkdir(confusion_out_path)
            f = open(confusion_out_path + "/log.txt", 'a')
            f.writelines(log)
            f.close()
        if savemat:
            mat_out_path = "./matfile/"
            if not os.path.isdir(mat_out_path):#创建测试结果文件夹
                os.mkdir(mat_out_path)
            scio.savemat(mat_out_path+'Uncertainty_max_multiid_paoku.mat',{'fp':np.array(uncertainty_re_fp),'spoof':np.array(uncertainty_re_spoof),
                                                            'efp':np.array(uncertainty_error_fp),'espoof':np.array(uncertainty_error_spoof)})
        if savecsv:
            csv_out_path = "./csvfile/"
            if not os.path.isdir(csv_out_path):#创建测试结果文件夹
                os.mkdir(csv_out_path)
            dict = {'path':pathall_fp,'score':uncertainty_re_fp}
            df = pd.DataFrame(dict)
            df.to_csv(csv_out_path+'0fp.csv',index=False,encoding="utf_8_sig")
            dict = {'path':pathall_spoof,'score':uncertainty_re_spoof}
            df = pd.DataFrame(dict)
            df.to_csv(csv_out_path+'spoof.csv',index=False,encoding="utf_8_sig")        
    Findlist = ['11']
    resultlist = []
    for i in Findlist:
        Loopcheck(args,i,G,test_set,teacher_flag,thr_mode,resultlist)



