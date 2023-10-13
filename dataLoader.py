'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal
import numpy as np
from pathlib import Path
class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
        #print(augment_files[0].split('\\'))
        for file in augment_files:
            if file.split('\\')[-3] not in self.noiselist:
                self.noiselist[file.split('\\')[-3]] = []
            self.noiselist[file.split('\\')[-3]].append(file)
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        print("#############",len(self.rir_files), len(augment_files))

        # Load data & labels
        self.data_list  = [] #audio paths
        self.data_label = [] #onehot labels

        self.dir = Path(self.train_path)
        print(self.dir)
        self.dictkeys = list(set([d.name for d in self.dir.iterdir() if d.is_dir()]))
        self.dictkeys.sort()
        self.dictkeys = { key : ii for ii, key in enumerate(self.dictkeys) }
        
        self.audio_path_list = list(set(self.find_wav_files(self.dir)))
        
        for audio_path in self.audio_path_list:
            speaker_label = self.dictkeys[audio_path.split('/')[0]]
            file_name = self.dir/audio_path
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)


    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])		
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_frame = np.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = np.stack([audio],axis=0)
        # Data Augmentation
        augtype = random.randint(0,3)
        if augtype == 0:   # Original
            audio = audio
        elif augtype == 1: # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2: # Babble
            audio = self.add_noise(audio, 'speech')
        #elif augtype == 3: # Music
        #    audio = self.add_noise(audio, 'music')
        elif augtype == 3: # Noise
            audio = self.add_noise(audio, 'noise')
        #elif augtype == 5: # Television noise
        #    audio = self.add_noise(audio, 'speech')
        #    audio = self.add_noise(audio, 'music')
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file    = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir         = np.expand_dims(rir.astype(np.float64),0)
        rir         = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db    = 10 * np.log10(np.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = np.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio],axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio

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