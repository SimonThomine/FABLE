import os
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from datasets.mvtec import MVTecDataset
from utils.util import  AverageMeter
from utils.functions import (
    cal_anomaly_maps,
    cal_loss
)
from models.resnet_reduced_backbone import modified_resnet18,reduce_student18



class AnomalyDistillation:
    def __init__(self, data,inference=False,big=True,modelName="reducedResnet18",out_indices=[1, 2, 3],DG=False):
        if data != None:
            self.threshold=0
            self.device = "cuda"
            self.data_path = data['Models']['data_path']
            if (DG):
                if (big):
                    self.obj = data['Models']['bigModelDG']
                    self.datasetName=data['Models']['bigModelDG']+"_"+str(data['Config']['imageSize'])
                else:
                    self.obj = data['Models']['smallModelDG']
                    self.datasetName=data['Models']['smallModelDG']+"_"+str(data['Config']['imageSize'])
            else :
                if (big):
                    self.obj = data['Models']['bigModel']
                    self.datasetName=data['Models']['bigModel']+"_"+str(data['Config']['imageSize'])
                else:
                    self.obj = data['Models']['smallModel']
                    self.datasetName=data['Models']['smallModel']+"_"+str(data['Config']['imageSize'])
                
            self.halfPrecision=data['Models']['halfPrecision']
            self.img_resize = data['Config']['imageSize']
            self.img_cropsize = data['Config']['imageSize']
            self.validation_ratio = 0.2
            self.num_epochs = 50
            self.lr = 0.0004
            self.batch_size = 8
            self.modelName= modelName
            self.phase=data['Training']['phase']
            self.out_indices=out_indices
            self.model_dir = self.data_path + '/models' + '/' + self.obj+ '_'+self.modelName+ '_'+str(self.img_resize)
            os.makedirs(self.model_dir, exist_ok=True)

            if(inference):
                self.load_model(False)
            else:
                self.load_model()

            
            

    def load_dataset(self):
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_dataset = MVTecDataset(self.data_path+"/datasets", class_name=self.datasetName, is_train=True, resize=self.img_resize, cropsize=self.img_cropsize)
        img_nums = len(train_dataset)
        valid_num = int(img_nums * self.validation_ratio)
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, **kwargs)

    def load_model(self,train=True):
        print("loading ==> " + self.obj)
    
        if self.modelName == "reducedResnet18":
            if train==False and self.halfPrecision==True:
                print("fp16 precision")
                self.model_t = modified_resnet18(img_size=self.img_resize).to(self.device).half() #! ajout half
                self.model_s = reduce_student18(pretrained=False,img_size=self.img_resize).to(self.device).half() #! ajout half
            else:
                self.model_t = modified_resnet18(img_size=self.img_resize).to(self.device)
                self.model_s = reduce_student18(pretrained=False,img_size=self.img_resize).to(self.device)
        else :
            print("wrong model specified")
            
        for param in self.model_t.parameters():
            param.requires_grad = False
        self.model_t.eval()
        if(train == False):
            try:
                checkpoint = torch.load(os.path.join(self.model_dir, 'model_s.pth'))
            except:
                raise Exception('Check saved model path.')
            self.model_s.load_state_dict(checkpoint['model'])
            for param in self.model_s.parameters():
                param.requires_grad = False
            self.model_s.eval()
            
        
        
    def train(self):
        self.load_dataset()
        self.optimizer = torch.optim.Adam(
                self.model_s.parameters(), lr=self.lr, betas=(0.5, 0.999)
            )
        print("training ==> " + self.obj)
        self.model_s.train()
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        epoch_bar = tqdm(
            total=len(self.train_loader) * self.num_epochs,
            desc="Training",
            unit="batch",
        )
        for epoch in range(1, self.num_epochs + 1):
            losses = AverageMeter()
            for data, label, _ in self.train_loader:
                data = data.to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    features_t, features_s = self.infer(data)
                    loss = cal_loss(features_s, features_t)
                    losses.update(loss.sum().item(), data.size(0))
                    loss.backward()
                    self.optimizer.step()
                epoch_bar.set_postfix({"Loss": loss.item()})
                epoch_bar.update()

            val_loss = self.val(epoch, epoch_bar)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()

            epoch_time.update(time.time() - start_time)
            start_time = time.time()
        epoch_bar.close()
        
    def val(self, epoch, epoch_bar):
        self.model_s.eval()
        losses = AverageMeter()
        for data, _, _ in self.val_loader:
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t, features_s = self.infer(data)
                loss = cal_loss(features_s, features_t)
                losses.update(loss.item(), data.size(0))
        epoch_bar.set_postfix({"Loss": loss.item()})
        return losses.avg

    def save_checkpoint(self):
        state = {"model": self.model_s.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "model_s.pth"))

    def test(self):
        self.load_model(False)
        try:
            checkpoint = torch.load(os.path.join(self.model_dir, "model_s.pth"))
        except:
            raise Exception("Check saved model path.")
        self.model_s.load_state_dict(checkpoint["model"])
        self.model_s.eval()
        
        kwargs = (
            {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
        )
        test_dataset = MVTecDataset(
            self.data_path+"/datasets",
            class_name=self.datasetName,
            is_train=False,
            resize=self.img_resize,
            cropsize=self.img_cropsize,
        )
        batch_size_test = 1  
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size_test, shuffle=False, **kwargs
        )
        scores = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        progressBar = tqdm(test_loader)
        for data, label, mask in test_loader:
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(label.cpu().numpy())
            gt_mask_list.append(mask.squeeze().cpu().numpy())
            data = data.to(self.device).half() if self.halfPrecision else data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t, features_s = self.infer(data)
                score = cal_anomaly_maps(features_s, features_t, 256)
                
                progressBar.update()

            if batch_size_test == 1:
                scores.append(score)
            else:
                scores.extend(score)
        progressBar.close()
        scores = np.asarray(scores)

        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)  
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print(self.obj + " image ROCAUC: %.3f" % (img_roc_auc))
        precision, recall, thresholds = precision_recall_curve(
                gt_list.flatten(), img_scores.flatten()
            )
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        cls_threshold = thresholds[np.argmax(f1)]
        self.threshold=(cls_threshold).item() 


    def infer(self, data):    
        features_t = self.model_t(data)
        features_s = self.model_s(data)
        return features_t, features_s
