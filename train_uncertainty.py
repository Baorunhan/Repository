'''
pytorch官方文档:https://pytorch.org/docs/stable/index.html
'''
#from msilib import sequence
from ast import Not
from cv2 import split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.utils.data as tud
from torch.autograd import  Function
import warnings
warnings.filterwarnings("ignore")
import torchvision
from torchvision import transforms as transforms
from Myphotometric import ImgAugTransform
from torch import Tensor, bilinear
from typing import Callable,Optional
import os
import numpy as np
import argparse
from torch.autograd import Variable
#导入网络模型
from mobilenetv3 import MNV3_large2_uncertainty, Enablecertainlayer_fortrain
import datetime 
import re
import random,string
import cv2
import PIL.Image as Image
import MylossF
from itertools import cycle#读取两个长度不一的数据集
from shutil import copyfile
import PIL
args = None

class Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, transform_iaa=None, target_transform=None,
                 loader=None, is_valid_file=None, select_retry=False, randomselect=False, gt = None,
                 selectnum=10000,selectlabel=None):
        super(Dataset, self).__init__(root, transform=transform, target_transform=target_transform,
                                      is_valid_file=is_valid_file)
        self.transform_iaa = transform_iaa
        self.select_retry = select_retry
        self.randomselect = randomselect
        self.selectnum = selectnum
        self.gt = gt
        self.label = selectlabel
        #筛选retry-X
        splits =  {"images": [], "label": []} 
        splits_clean =  {"images": [], "label": [],"id":[]} 
        if self.select_retry:
            for single in self.samples:
                if ('retry' in single[0]) and ('retry-0' in single[0]):
                    splits["images"].append(single[0])
                    if self.gt is not None:
                        splits["label"].append(self.gt)
                    else:
                        splits["label"].append(single[1])                      
                elif ('retry' in single[0]) == False:
                    splits["images"].append(single[0])
                    if self.gt is not None:
                        splits["label"].append(self.gt)
                    else:
                        splits["label"].append(single[1])                     
        else:
            for single in self.samples:
                splits["images"].append(single[0])
                if self.gt is not None:
                    splits["label"].append(self.gt)
                else:
                    splits["label"].append(single[1]) 

        if self.label is not None:#如果要让假手指分id
            for index,lablename in enumerate(self.label):
                f = ([s for s in splits["images"] if lablename in s])  
                splits_clean["id"].extend(len(f)*[index])
                splits_clean["images"].extend(f)
                splits_clean["label"].extend(len(f)*[self.gt])
        else:
            splits_clean = splits

        if self.randomselect:
            Toalindex = range(len(splits_clean["images"]))
            perm = random.sample(Toalindex,self.selectnum)
            for obj in ["images", "label"]:
                splits_clean[obj] = np.array(splits_clean[obj])[perm].tolist()

        self.crawl_folders(splits_clean)#分数和标签成对处理

    def crawl_folders(self, splits):
        sequence_set = []
        if len(splits) == 3:
            for (img, lb, id) in zip(
                splits["images"], splits["label"], splits["id"]
            ):
                sample = {"images": img, "label": lb, "id": id}
                sequence_set.append(sample)
        else:
             for (img, lb) in zip( #测试集不需要id
                splits["images"], splits["label"]
            ):
                sample = {"images": img, "label": lb}
                sequence_set.append(sample)           
        self.samples = sequence_set     # 成对（.bmp&.npy）的数据列表  

    def __getitem__(self, index):
        sample_org = self.samples[index]
        if len(sample_org)==3:
            path, target, id = sample_org["images"],sample_org["label"], sample_org["id"]
        else:
            path, target = sample_org["images"],sample_org["label"]
        #sample = load_as_float(path)
        sample = self.loader(path)
        if self.transform is not None:#基础数据增强，PIL
            sample = self.transform(sample)
        if self.transform_iaa is not None:#利用imgaug包，加入噪声，NP
            sample = np.array(sample).astype(np.uint8)
            sample = sample[:, :, np.newaxis]
            sample = self.transform_iaa(sample)
            sample = sample[:, :, np.newaxis].squeeze()
            sample = Image.fromarray(sample)#转回PIL格式
			
        sample = transforms.ToTensor()(sample)#最后totensor
        if self.target_transform is not None:
            target = self.target_transform(target)
        if len(sample_org)==3:
            return sample, target, id
        else:
            return sample, target

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def log_print(log, content):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    print(time, content)
    log.append(time)
    log.append("  " + content + "\n")
    
class Net(object):
    def __init__(self, args):
        self.model = None
        self.symbol = None
        self.paraflag = args.paraflag
        self.lr = args.lr
        self.epoch = 0
        self.epochs = 240
        self.batch_size = args.batch_size
        self.criterion_softmax = None
        self.criterion_centreloss = None
        self.criterion_focalloss = None
        self.optimizer = None
        self.optimizer_centreloss = None
        self.scheduler = None
        self.device = None
        self.cuda = torch.cuda.is_available()
        self.train_loader = None
        self.val_loader = None
        self.log = []
        self.traindir0fp = args.traindir0fp
        self.traindirspoof = args.traindirspoof
        # self.traindir = args.traindir
        self.validdir = args.validdir
        self.resume_flag = args.resume
        self.classnum = 2
        self.resizefactor = 1.0
        self.samplerweight = None
        self.centreLflag = 0
        self.centrelossweight = 0.2
        self.Negative_MLS_loss_weight = 0.1
        self.Myimgaug = ImgAugTransform()
        self.uncertainty_batchnum = 10000
        self.uncertainty_counter = 0
        self.toallabel = ["0fp","红底打印指纹","白底打印指纹"]

    def load_data(self):
        train_transform = transforms.Compose([
                                              transforms.Grayscale(),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),
                                              transforms.ColorJitter(contrast=(0.8, 1.2)),#retry0,retry1对比度变化，还是先加上
                                              ])
        test_transform = transforms.Compose([#transforms.Resize((int(80*self.resizefactor),int(64*self.resizefactor))),
                                             transforms.Grayscale(),

                                             ])
											 
        # train_set = torchvision.datasets.ImageFolder(self.traindir, train_transform)
        train_set_0fp = Dataset(self.traindir0fp,train_transform,transform_iaa=None,select_retry=False,randomselect=True, gt=0,selectnum=60000,selectlabel=['0fp']) #训练集的数据加入噪声和运动模糊,选择是否挑选retry_x
        train_set_spoof = Dataset(self.traindirspoof,train_transform,transform_iaa=None,select_retry=False,randomselect=True, gt=1,selectnum=50000,selectlabel=self.toallabel) #训练集的数据加入噪声和运动模糊,选择是否挑选retry_x
        appear_times = Variable(torch.zeros(self.classnum , 1))
        appear_times[0] = len(train_set_0fp)
        appear_times[1] = len(train_set_spoof)
        # train_set = Dataset(self.traindir,train_transform,transform_iaa=None,select_retry=False) #训练集的数据加入噪声和运动模糊,选择是否挑选retry_x
        # print(train_set)
        # print("retry-x number for train: %d"%(len(train_set.samples)))
        # appear_times = Variable(torch.zeros(self.classnum , 1))
        # for _,label in train_set.samples:
        #     appear_times[label] += 1
        # print(appear_times)#训练样本总数
        #exit()
		#这里计算类别权重
        class_sample_count = np.array([appear_times[0].item(),appear_times[1].item()])
        print(class_sample_count)		
        self.samplerweight = np.sum(class_sample_count).astype('float')/class_sample_count # 计算每一类别的权重
        #self.samplerweight[1] = self.samplerweight[1] * 0.9 #调整一下权重
        print(self.samplerweight)
	
        self.train0fp_loader = torch.utils.data.DataLoader(train_set_0fp, batch_size=int(self.batch_size/4), shuffle=False, num_workers=8, pin_memory=True) #读取手指训练数据 
        self.trainspoof_loader = torch.utils.data.DataLoader(train_set_spoof, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True) #读取手指训练数据 
        # self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)    
        test_set = Dataset(self.validdir, test_transform)
        print(test_set)
        self.val_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True) #  
        appear_times = Variable(torch.zeros(self.classnum , 1))
        for label in test_set.targets:
            appear_times[label] += 1
        print(appear_times)#测试样本总数

    def get_symbol(self):
        if self.cuda:
            torch.cuda.current_device()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(self.device)
        self.symbol = MNV3_large2_uncertainty(self.classnum).to(self.device)
        #self.symbol.init_params()
        if self.paraflag == 1:
            self.symbol = nn.DataParallel(self.symbol)#开启GPU并行运算
        Enablecertainlayer_fortrain(self.symbol,'uncertainty')#只训练uncertainty模块,固定其它

        self.criterion_softmax = nn.CrossEntropyLoss(torch.Tensor(self.samplerweight)).to(self.device)#加上类别权重
        self.criterion_centreloss = MylossF.CenterLoss(num_classes=2, feat_dim=48).to(self.device)#加不同的loss试试吧
        self.Negative_MLS_loss = MylossF.Negative_MLS_loss()
        #self.criterion_focalloss = MylossF.FocalLoss().to(self.device)
        #self.criterion_GHMloss = MylossF.GHMC_loss().to(self.device)

        [print(p.requires_grad) for p in self.symbol.parameters()]
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad is not False, self.symbol.parameters()), lr=self.lr)#不确定度loss
        #self.optimizer = optim.SGD(filter(lambda x: x.requires_grad is not False ,self.symbol.parameters()), lr=self.lr, momentum=0.9, weight_decay=1e-4, nesterov=False)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min',factor = 0.5, patience = 5, verbose = True)#管理器
    def train(self):
        #torch.nn.Module.train()将网络模型设置为训练模式，此时Dropout, BatchNorm等方法会生效
        self.symbol.eval()
        total, train_correct, train_loss_softmax,train_loss_uncertainty = 0,0,0,0#指标初始化
		
        #train_loss_focal, train_loss_GHM = 0,0        #新的
        #train_loss_ret_focal, train_loss_ret_GHM = 0,0#loss
		
        confusionmap = np.zeros((self.classnum, self.classnum))#混淆矩阵
        #按batch_size的大小分批次读入数据
        for  batch_num,(data0fp, dataspoof) in enumerate(zip(self.train0fp_loader,cycle(self.trainspoof_loader))):
            self.uncertainty_counter += 1#不确定度训练批次+1

            data_fp, target_fp, id_fp = data0fp[0], data0fp[1], data0fp[2]
            data_spoof, target_spoof,id_spoof = dataspoof[0], dataspoof[1], dataspoof[2]
            data = torch.cat((data_fp ,data_spoof),0)
            target = torch.cat((target_fp ,target_spoof),0)
            id = torch.cat((id_fp,id_spoof),0)
            data, target, id = data.to(self.device), target.to(self.device), id.to(self.device)

            output, feature, log_sigma2 = self.symbol(data)#前向传播
			#加入领域适配训练#
            #L2正则化，在过拟合的时候再加入
			
            loss_softmax = self.criterion_softmax(output,target)#交叉熵损失		
            loss_uncertainty = self.Negative_MLS_loss(feature,log_sigma2,id)#不确定性loss
            #label_weight = torch.ones((len(target),1)).cuda()
            #loss_GHM = self.criterion_GHMloss(output,target,label_weight)#GHMloss损失			
			
            loss = loss_uncertainty #只用不确定度loss

            self.optimizer.zero_grad()
            loss.backward()
 
            self.optimizer.step()

            #到这里已经完成了一次参数更新
            train_loss_softmax += loss_softmax.detach().cpu().numpy()
            train_loss_uncertainty += loss_uncertainty.detach().cpu().numpy()

            prediction = torch.max(output, 1)
            #print(prediction[1].cpu().numpy())
            total += target.size(0)
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
			#这里可以算一下训练集的混淆矩阵
            for i in range(len(target)):
                confusionmap[target[i]][prediction.indices[i]] += 1
            			
        train_loss_ret = train_loss_softmax / (2*batch_num + 1)
        train_loss_uncertainty_ret = train_loss_uncertainty / (2*batch_num + 1)		
		
        return train_loss_ret, train_correct / total, train_loss_uncertainty_ret, confusionmap

    def resume(self):
        model_path=r'./Models/model__2022_02_23_11_youhua-shifen-hongdi/ckpt_23.pth'
        #print('Resuming from checkpoint...\n model_path:%s'%model_path)
        log_print(self.log,'Resuming from checkpoint...\n model_path:%s'%model_path)
		
        checkpoint=torch.load(model_path)

        #'''
        #加载模型方式一:从预训练模型当中挑选指定层，依次复制

        dd = self.symbol.state_dict()#先定义好一个想要训练的结构
        
        ckpt_net = MNV3_large2(self.classnum).to(self.device)#读取参数来源网络结构

        if self.paraflag == 1:
            ckpt_net = nn.DataParallel(ckpt_net)#加上'.module'

        ckpt_net.load_state_dict(checkpoint['net'])#读取参数来源网络的权重       
        new_state_dict=ckpt_net.state_dict()#和获取参数来源网络的字典
        if self.paraflag == 1:
            for name,p in ckpt_net.named_buffers():
                if name in dd.keys() and name.startswith('module.layer1'):
                    dd[name]=new_state_dict[name]
                    print(name)
                if name in dd.keys() and name.startswith('module.bneck'):
                    dd[name]=new_state_dict[name]
                    print(name)  
                if name in dd.keys() and name.startswith('module.classifier'):
                    dd[name]=new_state_dict[name]
                    print(name)
            for name,p in ckpt_net.named_parameters():
                if name in dd.keys() and name.startswith('module.layer1'):
                    dd[name]=new_state_dict[name]
                    print(name)
                if name in dd.keys() and name.startswith('module.bneck'):
                    dd[name]=new_state_dict[name]
                    print(name)  
                if name in dd.keys() and name.startswith('module.classifier'):
                    dd[name]=new_state_dict[name]
                    print(name)
        else:
             for name,p in ckpt_net.named_parameters():
                if name in dd.keys() and name.startswith('layer1'):
                    dd[name]=new_state_dict[name]
                    print(name)
                if name in dd.keys() and name.startswith('bneck'):
                    dd[name]=new_state_dict[name]
                    print(name)  
                if name in dd.keys() and name.startswith('classifier'):
                    dd[name]=new_state_dict[name]
                    print(name)           
             				
        self.symbol.load_state_dict(dd,strict=True)
		
        '''
        self.symbol.load_state_dict(checkpoint['net'])   
        if checkpoint['centers']:#如果有保存centres就复制过来
            self.criterion_centreloss.load_state_dict(checkpoint['net'])  		
        '''
        
    def test(self):
        print("test:")
        #torch.nn.Module.eval()将网络模型设置为测试模式，会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
        self.symbol.eval()
        total,test_loss,test_correct,test_loss_ret = 0,0,0,0
        #torch.no_grad()内的计算过程不传播梯度
        confusionmap = np.zeros((self.classnum, self.classnum))
        Uncertainty_fp = []
        Uncertainty_spoof = []
        Uncertainty_perbatch = []
        fingerscore_fp = []
        fingerscore_spoof = []
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output,feature,log_sigma2 = self.symbol(data)
                p = nn.Softmax(dim=1)(output)
                feature = feature.detach().cpu().numpy()#特征（均值）
                sigma_sq = torch.exp(log_sigma2).detach().cpu().numpy()#方差
                loss = self.criterion_softmax(output, target)
                test_loss += loss.detach().cpu().numpy()
                prediction = torch.max(output, 1)
				
				#这里可以算一下混淆矩阵
                for i in range(len(target)):
                    confusionmap[target[i]][prediction.indices[i]] += 1
                    if target[i]!=prediction.indices[i]:
                        if target[i] == 0:
                            Uncertainty_fp.append(MylossF.aggregate_PFE(feature[i][:],sigma_sq[i][:]))
                            fingerscore_fp.append(p[i][0].cpu().numpy()*65536)
                        else:
                            Uncertainty_spoof.append(MylossF.aggregate_PFE(feature[i][:],sigma_sq[i][:]))
                            fingerscore_spoof.append(p[i][0].cpu().numpy()*65536)
                    else:
                        Uncertainty_perbatch.append(MylossF.aggregate_PFE(feature[i][:],sigma_sq[i][:]))
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                test_loss_ret = test_loss / (batch_num + 1)
        
        print('failed fp sample uncertainty:')
        print(Uncertainty_fp)
        print('failed fp sample score:')
        print(fingerscore_fp)

        print('failed spoof sample uncertainty:')
        print(Uncertainty_spoof)
        print('failed spoof sample score:')
        print(fingerscore_spoof)

        print('average Uncertainty:%.10f\n'%(np.mean(np.array(Uncertainty_perbatch))))  
        log_print(self.log, "test loss: %1.5f, test acc：%1.5f" % (test_loss_ret, test_correct / total))
        print('Test CM:\r')
        cm = confusionmap.astype('float') / confusionmap.sum(axis=1)[:, np.newaxis]
        print(cm)
        return test_loss_ret, test_correct / total, cm[0][0], cm[1][1]
    '''
    调用torch.save保存参数文件
    '''
    def save(self,epoch):
        model_out_path = "%s/ckpt_%d.pth" % (model_path,epoch)     
        torch.save(self.model, model_out_path)
        log_print(self.log, "Checkpoint saved to {}".format(model_out_path))

    def start(self):
        print("train dir path for fp:{}".format(self.traindir0fp))
        print("train dir path for spoof:{}".format(self.traindirspoof))
        # print("train dir path:{}".format(self.traindir))
        print("valid dir path:{}".format(self.validdir))
        self.load_data()
        self.get_symbol()
        if self.resume_flag:
            self.resume()
			
        train_accuracy = 0
        test_accuracy = 0
        FP_accuracy = 0
        #train_result = [0., 0., 0.]
        test_result = [0., 0., 0., 0.]
		
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            issave = 0
            #得到每一轮训练时训练集的损失和准确率
            train_result = self.train()
            log_print(self.log, "\nEpoch[%03d]: %03d/%d    acc=%1.5f   lossvalue=%1.5f  NMloss=%1.5f  \tlearning_rate=%1.7f" % (epoch,epoch,self.epochs,train_result[1],train_result[0], train_result[2],get_lr(self.optimizer)))
            print('Train CM:\r')
            cm = train_result[3].astype('float') / train_result[3].sum(axis=1)[:, np.newaxis]
            print(cm)
            #以训练集上的损失作为指标，当损失不再下降时降低学习率
            self.scheduler.step(train_result[0])
            #当训练集准确率高于历史最高准确率或训练轮数达到设定值时，保存模型，更新训练集历史最高准确率
            if (train_result[1] > train_accuracy and train_result[1] > 0.98) or (epoch == self.epochs):
                state = {
                        'net': self.symbol.state_dict(),
                        'train_acc':train_result[1],
                        'test_acc': test_result[1],
                        'epoch': epoch,
                }
                if self.centreLflag == 1:
                    state.update({'center':self.criterion_centreloss.centers.cpu().detach().numpy()})
                self.model = state
                self.save(epoch)
                issave = 1
            train_accuracy = max(train_accuracy, train_result[1])
            test_term = 20
            if train_accuracy > 0.99:
                test_term = 1
            if epoch%test_term == 0 or epoch ==1:
                test_result = self.test()
                log_print(self.log,'fp accuracy: %1.5f  missing anti accuracy: %1.5f' % (test_result[2], test_result[3]))#算一下误卡率和漏卡率，很重要
                #当真手指准确率高于历史最高数据且漏卡率达标时，保存模型
                if (test_result[2] > 0.9985) and (test_result[3] > 0.99):#2D识别率大于99%,2.5D识别率大于99%,误卡率低于0.15%的模型
                    if issave != 1:
                        state = {
                            'net': self.symbol.state_dict(),
                            'train_acc':train_result[1],
                            'test_acc': test_result[1],
                            'epoch': epoch,
                        }
                        if self.centreLflag == 1:
                            state.update({'center':self.criterion_centreloss.centers.cpu().detach().numpy()})
                        self.model = state
                        self.save(epoch)
                        issave = 1
                    #FP_accuracy = max(FP_accuracy, test_result[2])#更新有效的FP_accuracy					
                #当测试集准确率高于历史最高准确率时，保存模型，更新测试集历史最高准确率
                if (test_result[1] > test_accuracy) and (issave!=1):
                    state = {
                        'net': self.symbol.state_dict(),
                        'train_acc':train_result[1],
                        'test_acc': test_result[1],
                        'epoch': epoch,
                    }
                    if self.centreLflag == 1:
                        state.update({'center':self.criterion_centreloss.centers.cpu().detach().numpy()})
                    self.model = state
                    self.save(epoch)
                    issave = 1
                test_accuracy = max(test_accuracy, test_result[1])
                log_print(self.log,'[%d]Accuracy-Highest=%1.5f  test_accuracy=%1.5f  lossvalue=%1.5f' % (epoch, test_accuracy,test_result[1],test_result[0]))
                #self.scheduler.step(test_result[0])
            f = open(model_path + "/log.txt", 'a')
            f.writelines(self.log)
            f.close()
            self.log = []
            #学习率过小时或达到最大训练轮数终止训练
            print('lr-epoch:\n', get_lr(self.optimizer), epoch)
            if get_lr(self.optimizer) < 1e-5 or epoch == self.epochs:
                state = {
                    'net': self.symbol.state_dict(),
                    'train_acc':train_result[1],
                    'test_acc': test_result[1],
                    'epoch': epoch,
                }
                if self.centreLflag == 1:
                    state.update({'center':self.criterion_centreloss.centers.cpu().detach().numpy()})
                self.model = state
                self.save(epoch)
                log_print(self.log,"Epoch: End of the train, And the Accuracy: %1.5f " % test_accuracy)
                break
            
            if self.uncertainty_counter >= self.uncertainty_batchnum:
                print('Uncertainty Training over!')
                break
                
                
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument('-l','--lr', default=0.01, type=float, help='start learning rate')
    parser.add_argument('-b','--batch_size', default=256, type=int, help='')
    parser.add_argument('-t','--traindir', default="", help='train dir')
    parser.add_argument('-t1','--traindir0fp', default="", help='train dir')
    parser.add_argument('-t2','--traindirspoof', default="", help='train dir')
    parser.add_argument('-v','--validdir', default="", help='valid dir')
    parser.add_argument('-r','--resume', default="",type=bool, help='resume from checkpoint')
    parser.add_argument('-f','--flag', default="", help='specific version')
    parser.add_argument('-p','--paraflag', default=0,type=int, help='specific version')

    #args = parser.parse_args()
    args = parser.parse_args([#'-t','../dataset/S92162/Spoofdata-newraw-newpre/2Dfake/train/',
                              '-t1','../dataset/S92162/Spoofdata-newraw-newpre/2Dfake/train/0fp',
                              '-t2','../dataset/S92162/Spoofdata-newraw-newpre/2Dfake/train/2_25D',
                              '-v','../dataset/S92162/Spoofdata-newraw-newpre/2Dfake/test/',
                              '-f','uncertainty',
                              '-l', '0.0001','-p','1','-r','1'])
    return args
    
def main():
    global model_path

    args = parse_args()
    time = datetime.datetime.now().strftime('_%Y_%m_%d_%H')
    model_path = "./Models/model_%s"%(time)+"_"+ args.flag
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    net = Net(args)

    net.start()
    print('the train work is over!')

if __name__ == '__main__':
    main()
    
