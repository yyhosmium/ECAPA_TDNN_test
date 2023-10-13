import torch
import torch.nn as nn
import torchaudio
import math



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
    def __init__(self, sr=16000, n_fft=512, hop_length=160, n_mels=80, C=1024):
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
        
    def forward(self, x, aug=False):
        with torch.no_grad():
            x = self.pre_emphasis(x)
            x = self.log_mel_spec(x)
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if(aug==True):
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
