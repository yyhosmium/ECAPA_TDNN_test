#!/usr/bin/env python
# coding: utf-8

# # **Voxceleb Speaker Verification Task**
# - train model using Voxceleb1 dataset, test model using Voxceleb2 dataset.

# ## **0. Import Library**

# In[ ]:


import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import math
from IPython.display import display, clear_output
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from operator import itemgetter
import math
import random
import os 

DEV = "cuda"

#print gpu info

VOX1_DIR = Path('voxceleb1_full_dataset/')
VOX2_DIR = Path('voxceleb_dataset/voxceleb2_dev_iip/voxceleb2_dev_iip/')


sampling_rate = 16000 #Hz
max_audio_sec = 2.0
max_audio_length = int(max_audio_sec * sampling_rate + 240)


# ## **1. Data loader**
# - On the fly method
# - Train/valid ratio -> 0.95/0.05

# In[ ]:


class VoxcelebDataset:
    def __init__(self, dir_path, max_audio_length, split='train', sr=16000):
        self.dir = Path(dir_path)
        self.sr = sr #sampling rate
        self.max_audio_length = max_audio_length #audio maximum length
        #if split=="test":
        #    self.labels = pd.read_table(self.dir/(str(self.dir).split('\\')[-1]+".txt"), sep=' ', header=None)
        
        
    def convert_label_to_tensor(self):
        return torch.LongTensor(self.labels.values[:,0].astype('bool'))
        
    def trim_pad_audio(self, audio, random_trim=False):
        if audio.shape[0] <= self.max_audio_length: #길이 부족할시 뒤에 zero-padding
            pad_size = max(0, self.max_audio_length - audio.shape[0])
            audio = F.pad(audio, (0, pad_size), 'constant', 0)
        else:
            if random_trim == True: #랜덤한 시작점에서부터 자르기
                st = np.int64(random.random()*(audio.shape[0]-self.max_audio_length))
                audio = audio[st:st+self.max_audio_length]
            else:
                audio = audio[:self.max_audio_length] #아닐시 시작점부터 자르기
            
        return audio
    
    def __len__(self):
        return len(self.labels)


# In[ ]:


class TrainValidLoader(VoxcelebDataset):
    def __init__(self, dir_path, max_audio_length, split, sr=16000):
        super().__init__(dir_path, max_audio_length, split, sr)
        self.split = split
        
        if split=="train":
            self.dir = self.dir/"wav_train"
            speaker_ids = list(set([d.name for d in self.dir.iterdir() if d.is_dir()]))
            speaker_ids.sort()
            self.train_speaker_ids = speaker_ids

            #one-hot encoding 위한 speaker_id:idx 매칭
            speaker_id_to_idx = {speaker_id:idx for idx, speaker_id in enumerate(self.train_speaker_ids)}
            indexed_speaker_id = [speaker_id_to_idx[s] for s in self.train_speaker_ids]
            self.mapping_dict = {self.train_speaker_ids[k]:indexed_speaker_id[k] for k in range(len(indexed_speaker_id))}
            
            #wav_train에서 모든 오디오 경로 획득
            self.audio_path_list = self.find_wav_files(self.dir)
            self.labels = self.audio_path_list #__len__ return 용도. 다른 용도는 없음
            
        elif split == "valid":
            self.labels = pd.read_table(self.dir/"veri_test2.txt", sep=' ', header=None)
            self.label_tensor = self.convert_label_to_tensor() #train은 labels 필요없음
            
            
            
    def find_wav_files(self, base_path):
        wav_files = []
        for dirpath, dirnames, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith('.wav'):
                    full_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(full_path, base_path)
                    slash_path = relative_path.replace('\\', '/')
                    wav_files.append(slash_path)

        return wav_files
    
    def __getitem__(self, idx):
        if self.split == "train":
            file_path = self.dir/self.audio_path_list[idx]
            waveform, orig_sr = torchaudio.load(file_path)
            audio = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=self.sr)[0]
            audio = self.trim_pad_audio(audio, random_trim=True)
            
            label = self.mapping_dict[self.audio_path_list[idx].split('/')[0]]
            #label = label
            
            return audio, label
        
        elif self.split == "valid":
            file_path = self.dir/"wav_test"/self.labels.iloc[idx][1]
            waveform, orig_sr = torchaudio.load(file_path)
            audio_1 = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=self.sr)[0]
            
            file_path = self.dir/"wav_test"/self.labels.iloc[idx][2]
            waveform, orig_sr = torchaudio.load(file_path)
            audio_2 = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=self.sr)[0]      
            
            label = self.label_tensor[idx].type(torch.FloatTensor)
            
            return audio_1, audio_2, label

        

class TestLoader(VoxcelebDataset):        
    def __init__(self, dir_path, max_audio_length, split="test", sr=16000):
        super().__init__(dir_path, max_audio_length, split, sr)        
        self.labels = pd.read_table(self.dir/(str(self.dir).split('\\')[-1]+".txt"), sep=' ', header=None)
        self.label_tensor = self.convert_label_to_tensor()
        
    def __getitem__(self, idx):
        file_path = self.dir/self.labels.iloc[idx][1]
        waveform, orig_sr = torchaudio.load(file_path)

        audio_1 = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=self.sr)[0]

        file_path = self.dir/self.labels.iloc[idx][2]
        waveform, orig_sr = torchaudio.load(file_path)
        audio_2 = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=self.sr)[0]      

        label = self.label_tensor[idx].type(torch.FloatTensor)

        return audio_1, audio_2, label
        
        

# In[ ]:





# ## **2. Model Preparation**
# - SE-Res2 Block, Pre-Emphasis, Log mel-spectrogram, SpecAugment
# 

# In[ ]:


class SE_Block(nn.Module):
    def __init__(self, in_channels, bottleneck=128):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_1 = nn.Conv1d(in_channels, bottleneck, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv1d(bottleneck, in_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_1(y)
        y = self.relu(y)
        y = self.conv_2(y)
        y = self.sigmoid(y)
        out = x*y
        return out
        
        
class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, k, d, s=8): #k:kernel_size, d:dilation, s:scale_dimension
        super().__init__()
        self.scale = s
        self.width = in_channels//self.scale #res2net 각 element 크기, out_channel%scale==0이여야함. assert추가하기 out_c->in_c로 바꿨음
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn_1 = nn.BatchNorm1d(out_channels)
        
        convs = []
        for i in range(self.scale - 1): #첫번째 split element는 conv 사용을 안해서 scale-1임.
            dilation_conv_pad = math.floor(k/2)*d
            convs.append(
                nn.Sequential(
                    nn.Conv1d(self.width, self.width, kernel_size=k, dilation=d, padding=dilation_conv_pad),
                    nn.BatchNorm1d(self.width),
                    self.relu,
                )
            )
        self.convs = nn.ModuleList(convs)
        
        self.conv_3 = nn.Conv1d(out_channels, out_channels, kernel_size=1) #1x1Conv
        self.bn_3 = nn.BatchNorm1d(out_channels)
        
        self.se_block = SE_Block(out_channels, bottleneck=128)
        
    def forward(self, x):
        #Conv1D + ReLU + BN
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.bn_1(x)
        
        #Res2 Dilated Conv1D + ReLU + BN 
        split_x = torch.split(x, self.width, dim=1)#scale 개로 분할, 각 길이는 width
        for idx in range(self.scale): #res2 conv, (https://paperswithcode.com/method/res2net 참조)
            if idx==0: #xi
                y = split_x[idx] #y[0]
                x = y
            elif idx==1: #K[i](x[i])
                y = self.convs[idx-1](split_x[idx]) #y[1]
                x = torch.cat((x,y), dim=1) #stack y[n]
            else: #idx>=2, K[i](x[i]+y[i-1])
                y = self.convs[idx-1](split_x[idx]+y) # +y는 이전에 연산했던 y, 즉 y[i-1]
                x = torch.cat((x,y), dim=1) #stack y[n]
        
        
        #Conv1D + ReLU + BN
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.bn_3(x)
        
        #SE-Block
        x = self.se_block(x)
        
        out = x
        return out 

    
class PreEmphasis(nn.Module):
    def __init__(self, alpha=0.97):
        super().__init__()
        self.alpha = alpha
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, bias=False)
        self.conv.weight.data[0, 0, 1] = 1
        self.conv.weight.data[0, 0, 0] = -self.alpha

    def forward(self, x):
        x = x.unsqueeze(1)
        emphasized = self.conv(x)
        return emphasized[:, :, :-1].squeeze(1)

    
class LogMelSpec(nn.Module):
    #Log-Mel Spectrogram, ECAPA-TDNN 참조
    def __init__(self, sr, n_fft, hop_length, n_mels):
        super().__init__()
        #window_length=sr*window_ms, hop_length=sr*frame_shift
        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=400, 
                                                                  hop_length=hop_length, f_min=20, f_max=7600,
                                                                  window_fn=torch.hamming_window, n_mels=n_mels)
    def forward(self, x):
        mel_spec = self.mel_converter(x)
        log_mel_spec = (mel_spec + 1e-6).log() #log(0) 방지
        return log_mel_spec

    
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_range=(0,10), time_mask_range=(0,5)): #range는 ECAPA-TDNN의 값을 따름
        super().__init__()
        #masking 범위 설정시 uniform distribution(random)에 의해 선택되어야함.
        self.freq_mask_range = freq_mask_range
        self.time_mask_range = time_mask_range
           
    def frequency_masking(self, x):#dim=1
        freq_mask_range = self.freq_mask_range
        batch_size, n_mels, time = x.shape
        freq_mask_len = torch.randint(freq_mask_range[0], freq_mask_range[1], (batch_size, ), device=x.device)
        freq_mask_pos = torch.randint(0, max(1, n_mels - freq_mask_len.max()), (batch_size, ), device=x.device)
        for i in range(batch_size):
            x[i,freq_mask_pos[i]:freq_mask_pos[i]+freq_mask_len[i],:] = 0
            
        return x
    
    def time_masking(self, x):#dim=2
        time_mask_range = self.time_mask_range
        batch_size, n_mels, time = x.shape
        time_mask_len = torch.randint(time_mask_range[0], time_mask_range[1], (batch_size, ), device=x.device)
        time_mask_pos = torch.randint(0, max(1, time - time_mask_len.max()), (batch_size, ), device=x.device)
        for i in range(batch_size):
            x[i,:,time_mask_pos[i]:time_mask_pos[i]+time_mask_len[i]] = 0
        
        return x

    def forward(self, x):
        x = self.frequency_masking(x)
        x = self.time_masking(x)
        #time wrapping은 딱히 큰 영향이 없음.
        return x
    

class AttentiveStatPool(nn.Module):
    def __init__(self, in_channels, bottleneck=64):
        super().__init__()
        #x의 통계 정보를 활용하는 것으로 1d conv에서 channel + context까지 고려 가능.
        #SE block을 attention으로 사용하면 AdaptiveAvgPool때문에 Context 정보에 손실 발생함. Pooling을 제거하고 변형해야함.
        self.conv_1 = nn.Conv1d(in_channels*3, bottleneck, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv1d(bottleneck, in_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2) #time 축에 대해서 softmax
        
    def forward(self, x):
        #Global properties of the utterance (x, mean(x), sd(x)) , shape : (batch_size,1536,T) -> (batch_size,1536*3,T)
        T = x.shape[-1]
        glob_prop_utt_x = torch.cat(
            (x, torch.mean(x, dim=2, keepdim=True).repeat(1,1,T), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,T)), 
            dim=1
        )
        
        #ASP weight(or attention)
        w = self.conv_1(glob_prop_utt_x)
        w = self.relu(w)
        w = self.conv_2(w)
        w = self.softmax(w) #shape : (batch_size,1536,T)
        
        #Weighted Statistics Pooling
        weighted_mean_x = torch.sum(x*w,dim=2) #평균값인데 데이터 개수 모두 같아서 굳이 안나눠줘도됨
        weighted_sd_x = torch.sqrt((torch.sum((x**2)*w, dim=2)-weighted_mean_x**2).clamp(min=1e-4)) #제곱*가중치의 평균 - 가중평균의 제곱
        
        x = torch.cat((weighted_mean_x, weighted_sd_x), dim=1)
        return x
    


# In[ ]:


class MyModel(nn.Module):
    def __init__(self, sr, n_fft, hop_length, n_mels, C):
        super().__init__()
        self.sr = sr
        
        self.relu = nn.ReLU() 
        
        self.pre_emphasis = PreEmphasis(alpha=0.97)
        self.log_mel_spec = LogMelSpec(sr, n_fft, hop_length, n_mels)
        self.spec_augment = SpecAugment()
        
        self.conv_1 = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2, dilation=1) #T 유지위해 padding
        self.bn_1 = nn.BatchNorm1d(C)
        
        self.se_res2_1 = SE_Res2Block(C, C, k=3, d=2)
        self.se_res2_2 = SE_Res2Block(C, C, k=3, d=3)
        self.se_res2_3 = SE_Res2Block(C, C, k=3, d=4)
        
        self.conv_2 = nn.Conv1d(3*C, int(3*C/2), kernel_size=1, dilation=1) #1536xT
        self.attentive_stat_pool = AttentiveStatPool(int(3*C/2))
        self.bn_2 = nn.BatchNorm1d(3*C) #3072x1
        
        self.fc_1 = nn.Linear(3*C,192)#int(3*C/16)) #192x1
        self.bn_3 = nn.BatchNorm1d(192)#int(3*C/16))
        
    def forward(self, x, spec_aug=False):
        with torch.no_grad():
            x = self.pre_emphasis(x)
            x = self.log_mel_spec(x)
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if(spec_aug==True):
                x = self.spec_augment(x)      
        #Conv1D + ReLU + BN (k=5, d=1) , shape:(C,T)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.bn_1(x)
        
        #SE-Res2Block (k=3, d=2) , shape:(C,T)
        x_1 = self.se_res2_1(x) #skip connection 사용시 결과가 더 좋았다고 conclusion에 있음
        #SE-Res2Block (k=3, d=3) , shape:(C,T)
        x_2 = self.se_res2_2(x+x_1)
        #SE-Res2Block (k=3, d=4) , shape:(C,T)
        x_3 = self.se_res2_3(x+x_1+x_2)
        
        #Conv1D + ReLU (k=1, d=1) , shape:(1536,T)
        x = torch.cat((x_1,x_2,x_3), dim=1) #3x(C*T)개 합치기
        x = self.conv_2(x)
        x = self.relu(x)

        #Attentive Stat Pooling + BN , shape:(3072x1)
        x = self.attentive_stat_pool(x)
        x = self.bn_2(x)
        
        #FC + BN , shape:(192,1)
        x = self.fc_1(x)
        x = self.bn_3(x)

        out = x
        return out


# ## **3. Evaluation function, Loss function**
# - Evaluation function : EER, minDCF
# - Loss function : AAM-Softmax

# In[ ]:


def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


# In[ ]:


class AAMsoftmax(nn.Module):
    def __init__(self, n_class=100, m=0.3, s=15.0, embed_size=48):
        
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, embed_size), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


# In[ ]:


def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr))) # np.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100

    return tunedThreshold, eer, fpr, fnr

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


# ## **4. Train model**
# - Train/validate model
# 

# In[ ]:


def train_model(model, train_loader, valid_loader, optimizer, num_epochs, loss_func, device="cuda"):
    loss_records = []
    valid_acc_records = []
    cur_valid_acc = []
    
    model.train()
    index, top1, loss = 0,0,0
    
    for epoch in tqdm(range(num_epochs)):
        for batch in train_loader:
            optimizer.zero_grad()
            
            audio, label = batch
            audio, label = audio.to(device), label.to(device)
            
            speaker_embedding = model(audio, spec_aug=True)
            
            nloss, prec = loss_func(speaker_embedding, label)
            nloss.backward()
            optimizer.step()
            
            index += len(label)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            loss_records.append(nloss.item())
            

            plot_title = 'Train Loss:'+str(nloss.item())+" Val Loss:"+str(cur_valid_acc)+"ACC:"+str(top1/index*len(label))+' epoch:'+str(epoch)
            print(plot_title)
            display(tqdm(range(num_epochs), position=epoch, leave=True))
        
        EER, minDCF = validate_model(model, valid_loader, device)
        cur_valid_acc = [EER, minDCF]
        valid_acc_records.append([EER, minDCF])
        save_model(model, optimizer, {"loss": loss_records, "valid_acc": valid_acc_records}, epoch)
    
    return {"loss": loss_records, "valid_acc": valid_acc_records}


def valid_audio_processing(model, audio, device):
    audio = audio.squeeze(0)
    if audio.shape[0] <= max_audio_length:
        pad_size = max(0, self.max_audio_length - audio.shape[0]) 
        audio = torch.nn.functional.pad(audio, (0, pad_size), 'constant', 0)
        
    audio = audio.detach().cpu().numpy()
    data_1 = torch.FloatTensor(np.stack([audio],axis=0)).to(device)
    max_audio = max_audio_length

    feats = []
    startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)

    for asf in startframe:
        feats.append(audio[int(asf):int(asf)+max_audio])


    feats = np.stack(feats, axis = 0).astype(np.float64)
    data_2 = torch.FloatTensor(feats).to(device)
    # Speaker embeddings

    embedding_1 = model(data_1)
    embedding_1 = F.normalize(embedding_1, p=2, dim=1)
    embedding_2 = model(data_2)
    embedding_2 = F.normalize(embedding_2, p=2, dim=1)
    return [embedding_1, embedding_2]


def validate_model(model, valid_loader, device):
    model.eval()
    model.to(device)
    scores, labels = [], []
    
    with torch.no_grad():
        for batch in valid_loader:
            audio_1, audio_2, label = batch
            audio_1, audio_2, label = audio_1.to(device), audio_2.to(device), label.to(device)
            
            embeddings_1 = valid_audio_processing(model, audio_1, device)
            embeddings_2 = valid_audio_processing(model, audio_2, device)
            
            embedding_11, embedding_12 = embeddings_1
            embedding_21, embedding_22 = embeddings_2

            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(label.squeeze(0).detach().cpu().numpy())
    
    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1) 
    
    model.train()
    
    return EER, minDCF


def save_model(model, optimizer, train_record, epoch):
    torch.save({
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),
    'train_record':train_record
    }, './save_models/model_epoch_'+str(epoch)+'.pth')
    
    return 0


# In[ ]:





# In[ ]:

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    
    print("GPU availalbe :", torch.cuda.is_available())
    print("GPU name :", torch.cuda.get_device_name(0))
    print("GPU count :", torch.cuda.device_count())

    
    trainset = TrainValidLoader(dir_path=VOX1_DIR, max_audio_length=max_audio_length, split="train")
    validset = TrainValidLoader(dir_path=VOX1_DIR, max_audio_length=max_audio_length, split="valid")
    testset = TestLoader(dir_path=VOX2_DIR, max_audio_length=max_audio_length, split="test")
    
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    
    model = MyModel(sr=sampling_rate, n_fft=512, hop_length=160, n_mels=80, C=512)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model = model.to(DEV)
    #embed_size : 3*C/16
    loss_func = AAMsoftmax(n_class=len(trainset.train_speaker_ids), m=0.3, s=15.0, embed_size=192).to(DEV)
    
    
    train_record = train_model(model, train_loader, valid_loader, optimizer, num_epochs=80, 
                            loss_func=loss_func, device=DEV)

