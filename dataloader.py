print ("Everything before librosa imported")
from librosa.core import resample, stft
import librosa
print ("Librosa imported too!")
import torch
from torch.utils.data import Dataset
#from torchvision import transforms
import pickle
import numpy as np
import scipy.io.wavfile as wavf
import soundfile as sf


class SONYC_UST(Dataset):

    def __init__(self, mode='train', root_dir='XXX/SONYC_UST', select_class=[], nfft=1024, hop=512, sr=44100, nmel=128):

        self.mode = mode  # mode = 'train', 'validate'
        self.root_dir = root_dir
        import csv
        self.info_file_obj = open(self.root_dir + '/annotations_with_test.csv', 'r')
        self.info_file = csv.reader(self.info_file_obj, delimiter=',')
        self.arr = []
        for row in self.info_file:
            self.arr.append(row)
        self.arr = np.array(self.arr[1:])
        self.info_file_obj.close()
        self.filenames = [] # Unique filenames corresponding to the mode (test or train)
        self.file_idx = [] # Index of start and end corresponding to each file w.r.t complete annotations
        self.build_info()
        self.hop = hop
        self.sr = sr
        self.nfft = nfft
        self.nmel = nmel


    def build_info(self):
        prev_name = self.arr[0, 2]
        self.filenames.append(prev_name)
        start_idx = 0
        end_idx = -1

        for i in range(self.arr.shape[0]):
            new_name = self.arr[i, 2]
            if not new_name == prev_name:
                self.filenames.append(new_name)
                end_idx = i - 1 
                self.file_idx.append([start_idx, end_idx]) # This file idx entry is for the previous file
                start_idx = end_idx + 1
                prev_name = new_name + ''
        self.file_idx.append((start_idx, self.arr.shape[0]-1))

        self.filenames = np.array(self.filenames)
        self.file_idx = np.array(self.file_idx)

        idx_list = [] # To select indices corresponding to the mode
        for i in range(self.filenames.shape[0]):
            if self.mode == 'train':
                if self.arr[self.file_idx[i, 0], 0] == self.mode:
                    idx_list.append(i)
            else:
                if self.arr[self.file_idx[i, 0], 0] == self.mode and (self.arr[self.file_idx[i, 0]:self.file_idx[i, 1]+1, 3] == '0').any():
                    idx_list.append(i)
                #if self.filenames[i] == '46_010846.wav': # Previously debugging code, now useless (Nov 24, 2021)
                #    print (self.arr[self.file_idx[i, 0], 0], self.arr[self.file_idx[i, 1], 3])
        self.filenames, self.file_idx = self.filenames[idx_list], self.file_idx[idx_list]

        return        
        

    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        start_idx, end_idx = self.file_idx[idx, 0], self.file_idx[idx, 1]
        wav_file_name = self.root_dir + '/audio/' + self.filenames[idx]

        fs, inp_audio = wavf.read(wav_file_name)
        inp_audio = inp_audio.astype(np.float32)
        inp_audio = inp_audio / inp_audio.max()
        inp_audio = resample(y = inp_audio, orig_sr=fs, target_sr=self.sr, res_type='scipy')
        if inp_audio.shape[0] < self.sr*10 - 1:
            inp_audio_padded = np.zeros(int(self.sr*10))
            inp_audio_padded[:inp_audio.shape[0]] = inp_audio
            inp_audio = inp_audio_padded + 0.0
            #print ('Unusual sample:', idx)
        Xs = stft(inp_audio, n_fft=self.nfft, hop_length=self.hop)
        Xmel = librosa.feature.melspectrogram(sr=self.sr, S=np.abs(Xs), n_fft=self.nfft, hop_length=self.hop, n_mels=self.nmel)
        Xls = np.log(1.0 + np.abs(Xs))
        Xlgmel = librosa.power_to_db(Xmel).T

        #y = ( np.mean(self.arr[start_idx:end_idx+1, -8:].astype(int), axis=0) > 0.5 ).astype(int) # Majority voting of annotations
        if self.mode == 'train':
            y = ( np.sum(self.arr[start_idx:end_idx+1, -8:].astype(int), axis=0) > 0.5 ).astype(int) # Minority voting of annotations
        else:
            zero_idx = start_idx + np.where(self.arr[start_idx:end_idx+1, 3] == '0')[0][0]
            y = self.arr[zero_idx, -8:].astype(int)

        return torch.as_tensor(Xlgmel).unsqueeze(0).float(), torch.as_tensor(y).float(), torch.as_tensor(np.abs(Xs)).float(), torch.as_tensor(Xls).float(), Xs/(1e-9 + np.abs(Xs)), wav_file_name




class ESC50(Dataset):

    def __init__(self, mode='train', num_FOLD=1, root_dir='XXX/ESC50', select_class=[], add_noise=False, nfft=1024, hop=512, sr=44100, nmel=128):

        self.mode = mode  # mode = 'train', 'validate', or 'test'
        self.num_FOLD = num_FOLD
        self.root_dir = root_dir # root_dir should be the one within which data folder is present
        if select_class == []:
            self.subset = list(range(50))
        else:
            self.subset = select_class

        import csv
        self.info_file_obj = open(self.root_dir + '/meta/esc50.csv', 'r')
        self.info_file = csv.reader(self.info_file_obj, delimiter=',')
        self.arr = []
        for rows in self.info_file:
            self.arr.append(rows)
        self.arr = np.array(self.arr) # arr[0] = ['filename', 'fold', 'target_class index', 'target_class name', 'esc10 true or false', 'src_file', 'take']
        #print (len(self.arr))
        self.info_file_obj.close()
        # Now select the data corresponding to the mode
        idx_list = []
        for i in range(1, self.arr.shape[0]):
            if (self.mode == 'test') and ( self.num_FOLD == int(self.arr[i][1]) ) and int(self.arr[i][2]) in self.subset:
                idx_list.append(i)
            elif self.mode == 'train' and (not self.num_FOLD == int(self.arr[i][1]) ) and int(self.arr[i][2]) in self.subset:
                idx_list.append(i)
        self.arr = self.arr[idx_list]
        self.hop = hop
        self.sr = sr
        self.nfft = nfft
        self.nmel = nmel
        self.noise = add_noise
        self.noise_strength = np.zeros([self.arr.shape[0]])
        self.signal_strength = np.zeros([self.arr.shape[0]])


    def __len__(self):
        return self.arr.shape[0]

    def generate_strength_stats(self):
        for idx in range(self.arr.shape[0]):
            samp_info = self.arr[idx]
            wav_file_name = self.root_dir + '/audio/' + samp_info[0]
            fs, inp_audio = wavf.read(wav_file_name)
            inp_audio = inp_audio.astype(np.float32)
            inp_audio = inp_audio / inp_audio.max() 
            if self.noise:
                noise = np.random.normal(0, 0.05, inp_audio.shape[0])
                energy_signal = (inp_audio ** 2).mean()
                noise = np.random.normal(0, 0.05, inp_audio.shape[0])
                energy_noise = (noise ** 2).mean()
                const = np.sqrt(energy_signal / energy_noise)
                noise = const * noise
                self.noise_strength[idx] = 10*np.log10((noise ** 2).mean())
                self.signal_strength[idx] = 10*np.log10((inp_audio ** 2).mean())

    
    def __getitem__(self, idx):
        samp_info = self.arr[idx]
        wav_file_name = self.root_dir + '/audio/' + samp_info[0]
        y = int(samp_info[2])
        if len(self.subset) < 50:
            y = int(self.subset.index(y))
        fs, inp_audio = wavf.read(wav_file_name)
        inp_audio = inp_audio.astype(np.float32)
        inp_audio = inp_audio / inp_audio.max() 
        if self.noise:
            energy_signal = (inp_audio ** 2).mean()
            noise = np.random.normal(0, 0.05, inp_audio.shape[0])
            energy_noise = (noise ** 2).mean()
            const = np.sqrt(energy_signal / energy_noise)
            noise = const * noise
            inp_audio = inp_audio + noise
        Xs = stft(inp_audio, n_fft=self.nfft, hop_length=self.hop)
        Xmel = librosa.feature.melspectrogram(sr=self.sr, S=np.abs(Xs), n_fft=self.nfft, hop_length=self.hop, n_mels=self.nmel)
        Xls = np.log(1.0 + np.abs(Xs))
        Xlgmel = librosa.power_to_db(Xmel).T
        return torch.as_tensor(Xlgmel).unsqueeze(0).float(), torch.as_tensor(y), torch.as_tensor(np.abs(Xs)).float(), torch.as_tensor(Xls).float(), Xs/(1e-9 + np.abs(Xs)), wav_file_name

    def overlap_two(self, idx1, idx2, lambda2=0.2):
        samp_info1, samp_info2 = self.arr[idx1], self.arr[idx2]
        wav_file_name1 = self.root_dir + '/audio/' + samp_info1[0]
        wav_file_name2 = self.root_dir + '/audio/' + samp_info2[0]
        y = int(samp_info1[2])
        if len(self.subset) < 50:
            y = int(self.subset.index(y))
        fs1, inp_audio1 = wavf.read(wav_file_name1)
        fs2, inp_audio2 = wavf.read(wav_file_name2)
        inp_audio1 = inp_audio1.astype(np.float32)
        inp_audio1 = inp_audio1 / inp_audio1.max()
        inp_audio2 = inp_audio2.astype(np.float32)
        inp_audio2 = inp_audio2 / inp_audio2.max()
        inp_audio = inp_audio1 + lambda2*inp_audio2
         
        Xs = stft(inp_audio, n_fft=self.nfft, hop_length=self.hop)
        Xmel = librosa.feature.melspectrogram(sr=self.sr, S=np.abs(Xs), n_fft=self.nfft, hop_length=self.hop, n_mels=self.nmel)
        Xls = np.log(1.0 + np.abs(Xs))
        Xlgmel = librosa.power_to_db(Xmel).T
        return torch.as_tensor(Xlgmel).unsqueeze(0).float(), torch.as_tensor(y), torch.as_tensor(np.abs(Xs)).float(), torch.as_tensor(Xls).float(), Xs/(1e-9 + np.abs(Xs)), wav_file_name1 + wav_file_name2




def make_weights_for_balanced_classes(dataset, nclasses=10):                        
    count = [0] * nclasses                                                      
    for batch_info in dataset:                                                         
        count[int(batch_info[1])] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(dataset)                                              
    for idx, val in enumerate(dataset):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight


if __name__ == '__main__':
    esc50 = ESC50()
    #data_val = SONYC_UST(mode='validate')
    #data_test = SONYC_UST(mode='test')




        
        
        
