#import matplotlib # Importing matplotlib for it working on remote server
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#import matplotlib.colors as color

import os, sys
home_dir = os.getcwd()
args = sys.argv[1:]
print (len(args), 'Arguments', args)

import torch # All the torch modules
#import librosa
import random
random.seed(108)
#torch.manual_seed(108)
#torch.cuda.manual_seed(108)
#torch.backends.cudnn.deterministic = True

import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
#from torch.optim import lr_scheduler
from torch.autograd import Variable
#from torchvision import datasets, transforms, models

#torch.cuda.set_device(1)

import dataloader as dl
import networks
import itertools, time

from sklearn.metrics import auc

import numpy as np
np.random.seed(108)

import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


print ('Imported all modules')

use_cuda = (True if args[2] == 'True' else False)
clf_train = (True if args[3] == 'True' else False)
device = torch.device("cuda" if use_cuda else "cpu")
train_shuffle = True
test_shuffle = False
train_epoch_info = []
test_epoch_info = []
W_list = []
H_list = []

dataset = str(args[1]) # Options: amnist, msdb, esc50, esc10, sust
all_datasets = ['esc50', 'sust']

if dataset == 'esc50':
    N_EPOCH = 35
    num_FOLD = 1
    esc50_dir = home_dir + '/datasets'
    #esc50_dir = '/tsi/clusterhome/jparekh/L2I/datasets'
    esc50_train1 = dl.ESC50(mode='train', num_FOLD=num_FOLD, root_dir=esc50_dir + '/ESC50')
    esc50_test1 = dl.ESC50(mode='test', num_FOLD=num_FOLD, root_dir=esc50_dir + '/ESC50', add_noise=False)
    train_loader = torch.utils.data.DataLoader(esc50_train1, batch_size=32, shuffle=train_shuffle, num_workers=1)
    test_loader = torch.utils.data.DataLoader(esc50_test1, batch_size=16, shuffle=test_shuffle, num_workers=1)

elif dataset == 'sust':
    N_EPOCH = 21
    #sonyc_dir = '/tsi/doctorants/jparekh/audio_datasets'
    sonyc_dir = home_dir + '/datasets'
    sust_train = dl.SONYC_UST(mode='train', root_dir= sonyc_dir + '/SONYC_UST')
    sust_valid = dl.SONYC_UST(mode='validate', root_dir= sonyc_dir +'/SONYC_UST')
    sust_test = dl.SONYC_UST(mode='test', root_dir= sonyc_dir +'/SONYC_UST')
    train_loader = torch.utils.data.DataLoader(sust_train, batch_size=32, shuffle=train_shuffle, num_workers=1)
    val_loader = torch.utils.data.DataLoader(sust_valid, batch_size=32, shuffle=test_shuffle, num_workers=1)
    test_loader = torch.utils.data.DataLoader(sust_test, batch_size=32, shuffle=test_shuffle, num_workers=16)



print ('Dataloader ready')

criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
criterion4 = nn.L1Loss()

if dataset == 'esc50':
    n_components = 100
    pool_frames = 1
    #pool_frames = 1
    n_classes = 50
    #init_nmf_filename = home_dir + '/output/esc50_output/initW_LMS_K100_sp0.8_Fold' + str(num_FOLD) + '_n1.npy'
    init_nmf_filename = home_dir + '/output/esc50_output/initW_LMS_K' + str(n_components) + '_sp0.8_Fold' + str(num_FOLD) + '_n1.npy'
    #init_nmf_filename = home_dir + '/output/esc50_output/initW_LMS_K40_sp0.8_Fold' + str(num_FOLD) + '_n1.npy'
    #init_nmf_filename = home_dir + '/output/esc50_output/initW_LMS_K70_sp0.8_Fold' + str(num_FOLD) + '_n1.npy'
    #init_nmf_filename = home_dir + '/output/esc50_output/initW_LMS_K128_sp0.8_Fold' + str(num_FOLD) + '_n1.npy'  
    f = networks.FtEx(n_classes=50).to(device)
    #H = networks.HNet_FtEx_v2(N_COMP = n_components, T=431, in_maps=256).to(device)
    #H = networks.HNet_FtEx(N_COMP = n_components, T=431).to(device)   # For hidden layer experiments 
    H = networks.HNet_FtEx_general(N_COMP = n_components, T=431, in_maps=256).to(device) 
    #h = networks.explainer(N_COMP = n_components, N_FRAMES = pool_frames, n_classes=50, dropout=True).to(device)
    h = networks.explainer_v4(N_COMP = n_components, n_classes=50, dropout=True).to(device)
    #W = networks.NMF_D(N_COMP = n_components, FREQ=513, init_file=home_dir + '/output/esc50_output/initW_LMS_All500.npy').to(device)  # for model try6.pt
    #W = networks.NMF_D(N_COMP = n_components, FREQ=513, init_file=home_dir + '/output/esc50_output/initW_LMS_joint_All200_a0.1_l0.9.npy').to(device)
    #W = networks.NMF_D(N_COMP = n_components, FREQ=513, init_file=home_dir + '/output/esc50_output/initW_LMS_joint_All80_aW_aH_0.0000001_l0.9.npy').to(device)
    W = networks.NMF_D(N_COMP = n_components, FREQ=513, init_file=init_nmf_filename).to(device)
    opt_f = optim.Adam(itertools.chain(f.layer.parameters(), f.fc.parameters()), lr=0.001) # Only for fine-tuning classifier model
    optimizer = optim.Adam(itertools.chain(H.parameters(), h.parameters()), lr=0.0002) # Interpretability parameters optimizer, the best model till now was trained with lr=0.005 initially


elif dataset == 'sust': # Add control arg for loss_cce in test, train functions
    n_components = 80 # for try13, 14, 16
    #n_components = 128 # for try 15
    pool_frames = 1
    n_classes = 8
    #init_nmf_filename = home_dir + '/output/sust_output/initW_LMS_K80_sp0.8_NS5600_n5.npy' # file for try13
    init_nmf_filename = home_dir + '/output/sust_output/initW_LMS_SingleK10_sp0.8_NS1000_n6.npy' # file for try14 noise modelling used, 10 noise components fixed
    #class_thresh = n_classes*[0.5]
    class_thresh = np.array([0.6, 0.6, 0.5, 0.6, 0.5, 0.6, 0.6, 0.6])
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 4.0, 8.0, 8.0, 3.0, 10.0, 1.5, 10.0]).to(device)) # earlier pos_weight = 8*[3.0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 3.0, 3.0, 3.0, 3.0, 5.0, 1.5, 10.0]).to(device))
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8*[3.0]).to(device))
    f = networks.FtEx_v2(n_classes=8, mid_size=512).to(device)
    H = networks.HNet_FtEx_general(N_COMP = n_components, T=862, in_maps=256).to(device)
    #h = networks.explainer(N_COMP = n_components, N_FRAMES = pool_frames, n_classes=8, dropout=True).to(device)
    h = networks.explainer_v4(N_COMP = n_components, n_classes=8).to(device)
    #W = networks.NMF_D(N_COMP = n_components, FREQ=513, init_file=home_dir + '/output/sust_output/initW_LMS_K80_sp0.2_v4.npy').to(device)
    #W = networks.NMF_D(N_COMP = n_components, FREQ=513, init_file=home_dir + '/output/sust_output/initW_LMS_K80_sp0.02_NS5600_n5.npy').to(device)  # file for try12
    #W = networks.NMF_D(N_COMP = n_components, FREQ=513, init_file=home_dir + '/output/sust_output/initW_LMS_K80_sp0.8_NS5600_n5.npy').to(device) # file for try13
    W = networks.NMF_D(N_COMP = n_components, FREQ=513, init_file=init_nmf_filename).to(device)
    #opt_f = optim.Adam(itertools.chain(f.layer.parameters(), f.fc.parameters()), lr=0.0004) # Only for fine-tuning classifier model
    opt_f = optim.Adam(f.parameters(), lr=0.0001)
    #opt_f = optim.SGD(itertools.chain(f.layer.parameters(), f.fc.parameters()), lr=0.05, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.Adam(itertools.chain(H.parameters(), h.parameters()), lr=0.0005) # Interpretability parameters optimizer


print ("Networks, optimizer ready")
    

def compute_component_similarity(W):
    # Assume W of shape #FREQ_BINS x N_COMP
    # Computes 1 - pairwise distance
    from sklearn.metrics import pairwise_distances
    sim = 1-pairwise_distances(W.T, metric="cosine")
    return sim


def analyze(f, H, h, W, device, test_loader, n_classes=10):
    f.eval(), H.eval(), h.eval(), W.eval()
    f, H, h, W = f.to(device), H.to(device), h.to(device), W.to(device)
    batch_size = int(sample_data.shape[0])
    conf_matx_fy = np.zeros([n_classes, n_classes]) # n_classes x n_classes
    conf_matx_hf = np.zeros([n_classes, n_classes])
    conf_matx_hy = np.zeros([n_classes, n_classes])
    top5_fid, top3_fid = 0, 0
    for batch_info in test_loader:
        data, target = batch_info[0].to(device), batch_info[1].to(device)
        output, inter = f(data)
        embed = H(inter)
        rec_data, expl = W(embed), h(embed)
        pred_f = output.argmax(dim=1).cpu().data.numpy()
        pred_h = expl.argmax(dim=1).cpu().data.numpy()
        hp_full = expl.cpu().data.numpy()
        y = target.cpu().data.numpy()
        for j in range(y.shape[0]):
            conf_matx_fy[pred_f[j], y[j]] += 1
            conf_matx_hf[pred_h[j], pred_f[j]] += 1
            conf_matx_hy[pred_h[j], y[j]] += 1
            if pred_f[j] in (-1*hp_full[j]).argsort()[:5]:
                top5_fid += 1
            if pred_f[j] in (-1*hp_full[j]).argsort()[:3]:
                top3_fid += 1
    print ('Top-3 fidelity total:', top3_fid)
    print ('Top-5 fidelity total:', top5_fid)
    return conf_matx_fy, conf_matx_hf, conf_matx_hy


def faithfulness_eval(dataload, idx, f, H, W, h, component_wise=False, save_audio=False, thresh=0.2, select_random=False):
    # If component_wise=True, remove one component at a time and observe the difference

    import librosa
    import librosa.core as core
    import soundfile as sf

    batch = dataload[idx]
    data, target, file_name = batch[0], batch[1], batch[-1]
    output, inter = f(data.to(device).unsqueeze(0))
    embed = H(inter)
    out_interpreter = h(embed)
    pred_int = (nn.Sigmoid()(out_interpreter[0]).cpu().data.numpy() > 0.4).astype(int)
    pred_f = (nn.Sigmoid()(output[0]).cpu().data.numpy() > (class_thresh)).astype(int)
    #all_pred_classes = np.where((pred_int == 1)*(pred_f == 1))[0]
    #all_pred_classes = np.where(target.cpu().data.numpy() == 1)[0]
    all_pred_classes = np.where(pred_f == 1)[0]
    inp_mag_spec = np.exp(batch[3].cpu().data.numpy()) - 1
    rec_data = W(embed)
    log_rec_spec = rec_data[0].cpu().data.numpy()
    spec_shape = log_rec_spec.shape
    comp = np.ones([n_components, spec_shape[0], spec_shape[1]])
    ratio = np.ones([n_components, spec_shape[0], spec_shape[1]])

    W_mat = W.return_W()
    weights = h.fc1.weight
    #pool = nn.AdaptiveAvgPool1d(pool_frames)
    pool = nn.AdaptiveMaxPool1d(pool_frames)
    pool2 = nn.MaxPool1d(pool_frames, stride=pool_frames)
    #pooled_activ = torch.flatten(pool(embed), 1)  # For linear layer as theta function
    pooled_activ = h.return_pooled_activ(embed) # For the case when attention based theta function
    #output = nn.Sigmoid()(output)
    return_data = []
  
    for pred_cl in all_pred_classes:
        imp_full = pooled_activ + 0
        imp_full[0] = pooled_activ[0] * weights[int(pred_cl)]
        imp_full = imp_full / imp_full.max()
        imp_max_comp = pool2(imp_full.unsqueeze(0))[0, 0].cpu().data.numpy() # unsqueeze was done for performing pooling
        softmask_weights = np.exp(imp_max_comp)/(np.exp(imp_max_comp).sum())
        activ_max_comp = pool2(pooled_activ.unsqueeze(0))[0, 0].cpu().data.numpy() # unsqueeze was done for performing pooling
        #print (imp_full.shape, imp_max_comp.shape)
        #print (expl)
        imp_components = (-1*imp_max_comp).argsort()[:5]
        if select_random:
            #other_components = np.where(imp_max_comp < 0.001)[0]
            other_components = np.where(imp_max_comp > -10)[0]
            np.random.shuffle(other_components)
            avg_expl_comp = [4, 3, 2, 1, 3, 1, 4, 1]
            comps = other_components[:avg_expl_comp[pred_cl]]
            #print ('Selecting random components', comps)
        top_comp_drop = [0.0, 0.0] # index 0 for top components, index 1 for non-important components
        top_comp_count = [0, 0]
        residual_lgmag_spec = np.log(1 + inp_mag_spec) + 0.0
        enhanced_lgmag_spec = np.log(1 + inp_mag_spec) * 0 + 0.0

        if component_wise:
            for i in range(n_components):
                comp[i], ratio[i] = select_component(i, batch[3].cpu().data.numpy(), embed[0].cpu().data.numpy(), W_mat)
                #residual_mag_spec = inp_mag_spec - comp[i]
                residual_mag_spec = get_MS( np.log(1 + inp_mag_spec) - comp[i] )
                new_time_signal = core.istft(residual_mag_spec*batch[-2], hop_length=dataload.hop)
                #new_time_signal = new_time_signal / new_time_signal.max()
                new_Xs = core.stft(new_time_signal, n_fft=dataload.nfft, hop_length=dataload.hop)
                new_Xmel = librosa.feature.melspectrogram(sr=dataload.sr, S=np.abs(new_Xs), n_fft=dataload.nfft, hop_length=dataload.hop, n_mels=dataload.nmel)
                new_Xlgmel = librosa.power_to_db(new_Xmel).T
                new_inp = torch.as_tensor(new_Xlgmel).unsqueeze(0).float().to(device).unsqueeze(0)
                output2, inter2 = f(new_inp)
                output2 = nn.Sigmoid()(output2)
                if imp_max_comp[i] > 0.1:
                    top_comp_drop[0] += ((output[0][pred_cl] - output2[0][pred_cl]) / output[0][pred_cl]).item()
                    top_comp_count[0] += 1
                else:
                    top_comp_drop[1] += ((output[0][pred_cl] - output2[0][pred_cl]) / output[0][pred_cl]).item()
                    top_comp_count[1] += 1
                    #print ('Probability drop for component ' + str(i), 'Class ', pred_cl, imp_max_comp[i], (output[0][pred_cl] - output2[0][pred_cl]).item())
            print ('Class', pred_cl, 'Top components average drop:', top_comp_drop[0]/top_comp_count[0], top_comp_count[0], top_comp_count[1])
            print ('Class', pred_cl, 'Others average drop:', top_comp_drop[1]/top_comp_count[1])

        else:
            for i in range(n_components):
                if imp_max_comp[i] > thresh and (not select_random):
                    comp[i], ratio[i] = select_component(i, batch[3].cpu().data.numpy(), embed[0].cpu().data.numpy(), W_mat)
                    residual_lgmag_spec = residual_lgmag_spec - comp[i]
                    enhanced_lgmag_spec = enhanced_lgmag_spec + softmask_weights[i] * comp[i]
                elif select_random:
                    if i in comps:
                        comp[i], ratio[i] = select_component(i, batch[3].cpu().data.numpy(), embed[0].cpu().data.numpy(), W_mat)
                        residual_lgmag_spec = residual_lgmag_spec - comp[i]
                        enhanced_lgmag_spec = enhanced_lgmag_spec + softmask_weights[i] * comp[i]
            new_time_signal = core.istft(get_MS(residual_lgmag_spec)*batch[-2], hop_length=dataload.hop)
            #energy_ratio = np.sqrt(np.mean(inp_mag_spec**2))/np.mean(np.mean(get_MS(enhanced_lgmag_spec)**2))
            #print ('Energy ratio:', energy_ratio)
            #new_time_signal = core.istft(0.4*get_MS(enhanced_lgmag_spec)*batch[-2], hop_length=dataload.hop)
            original_time_signal = core.istft(inp_mag_spec*batch[-2], hop_length=dataload.hop)
            res = original_time_signal - new_time_signal
            #plt.plot(original_time_signal, color='green')
            #plt.plot((original_time_signal - new_time_signal), color='red')
            #plt.plot(new_time_signal, color='black')
            #plt.show()
            #print ('Residual stats:', res.max(), res.mean(), res.min(), (res**2).mean())
            #new_time_signal = new_time_signal / new_time_signal.max()
            new_Xs = core.stft(new_time_signal, n_fft=dataload.nfft, hop_length=dataload.hop)
            new_Xmel = librosa.feature.melspectrogram(sr=dataload.sr, S=np.abs(new_Xs), n_fft=dataload.nfft, hop_length=dataload.hop, n_mels=dataload.nmel)
            new_Xlgmel = librosa.power_to_db(new_Xmel).T
            new_inp = torch.as_tensor(new_Xlgmel).unsqueeze(0).float().to(device).unsqueeze(0)
            output2, inter2 = f(new_inp)
            prob2 = nn.Sigmoid()(output2)
            prob = nn.Sigmoid()(output)
            #print ('Logit drop for Class ', pred_cl, ((output[0][pred_cl] - output2[0][pred_cl])).item())
            return_data.append([((output[0][pred_cl] - output2[0][pred_cl])).item(), ((prob[0][pred_cl] - prob2[0][pred_cl])).item(), ((prob[0][pred_cl] - prob2[0][pred_cl]) / prob[0][pred_cl]).item(), pred_cl, np.sum(imp_max_comp > thresh), pred_f[pred_cl]])
            
    return return_data # [Absolute logit drop, Absolute probability drop, Relative probability drop, predicted class, number of components, f-output]



def faithfulness_eval_multiclass(dataload, idx, f, H, W, h, component_wise=False, save_audio=False, thresh=0.2, select_random=False):
    # If component_wise=True, remove one component at a time and observe the difference

    import librosa
    import librosa.core as core
    import soundfile as sf

    batch = dataload[idx]
    data, target, file_name = batch[0], batch[1], batch[-1]
    output, inter = f(data.to(device).unsqueeze(0))
    embed = H(inter)
    out_interpreter = h(embed)
    pred_cl = int(output.argmax(dim=1)[0])
    #print ('Predicted class by interpreter:', int(out_interpreter.argmax(dim=1)[0]))
    #print ('Predicted class by classifier:', int(output.argmax(dim=1)[0]) )
    #print ('Target class:', int(target))

    inp_mag_spec = np.exp(batch[3].cpu().data.numpy()) - 1
    rec_data = W(embed)
    log_rec_spec = rec_data[0].cpu().data.numpy()
    spec_shape = log_rec_spec.shape
    comp = np.ones([n_components, spec_shape[0], spec_shape[1]])
    ratio = np.ones([n_components, spec_shape[0], spec_shape[1]])

    W_mat = W.return_W()
    weights = h.fc1.weight
    #pool = nn.AdaptiveAvgPool1d(pool_frames)
    pool = nn.AdaptiveMaxPool1d(pool_frames)
    pool2 = nn.AvgPool1d(pool_frames, stride=pool_frames)
    pooled_activ = torch.flatten(pool(embed), 1)  # For linear layer as theta function
    #pooled_activ = h.return_pooled_activ(embed) # For the case when attention based theta function
  
    imp_full = pooled_activ + 0
    imp_full[0] = pooled_activ[0] * weights[int(pred_cl)]
    imp_full = imp_full / imp_full.max()
    imp_max_comp = pool2(imp_full.unsqueeze(0))[0, 0].cpu().data.numpy() # unsqueeze was done for performing pooling
    softmask_weights = np.exp(imp_max_comp)/(np.exp(imp_max_comp).sum())
    activ_max_comp = pool2(pooled_activ.unsqueeze(0))[0, 0].cpu().data.numpy() # unsqueeze was done for performing pooling
    #print (imp_full.shape, imp_max_comp.shape)
    #print (expl)
    imp_components = (-1*imp_max_comp).argsort()[:5]
    if select_random:
        other_components = np.where(imp_max_comp > -10)[0]
        #other_components = np.where(imp_max_comp < 0.001)[0]
        np.random.shuffle(other_components)
        comps = other_components[:7]
    top_comp_drop = [0.0, 0.0] # index 0 for top components, index 1 for non-important components
    top_comp_count = [0, 0]
    residual_lgmag_spec = np.log(1 + inp_mag_spec) + 0.0
    enhanced_lgmag_spec = np.log(1 + inp_mag_spec) * 0.0

    if component_wise:
        for i in range(n_components):
            comp[i], ratio[i] = select_component(i, batch[3].cpu().data.numpy(), embed[0].cpu().data.numpy(), W_mat)
            #residual_mag_spec = inp_mag_spec - comp[i]
            residual_mag_spec = get_MS( np.log(1 + inp_mag_spec) - comp[i])
            new_time_signal = core.istft(residual_mag_spec*batch[-2], hop_length=dataload.hop)
            #new_time_signal = new_time_signal / new_time_signal.max()
            new_Xs = core.stft(new_time_signal, n_fft=dataload.nfft, hop_length=dataload.hop)
            new_Xmel = librosa.feature.melspectrogram(sr=dataload.sr, S=np.abs(new_Xs), n_fft=dataload.nfft, hop_length=dataload.hop, n_mels=dataload.nmel)
            new_Xlgmel = librosa.power_to_db(new_Xmel).T
            new_inp = torch.as_tensor(new_Xlgmel).unsqueeze(0).float().to(device).unsqueeze(0)
            output2, inter2 = f(new_inp)
            #output2 = (output2)
            if imp_max_comp[i] > thresh:
                top_comp_drop[0] += ((output[0][pred_cl] - output2[0][pred_cl]) / output[0][pred_cl]).item()
                top_comp_count[0] += 1
            else:
                top_comp_drop[1] += ((output[0][pred_cl] - output2[0][pred_cl]) / output[0][pred_cl]).item()
                top_comp_count[1] += 1
                #print ('Probability drop for component ' + str(i), 'Class ', pred_cl, imp_max_comp[i], (output[0][pred_cl] - output2[0][pred_cl]).item())
        print ('Class', pred_cl, 'Top components average drop:', top_comp_drop[0]/top_comp_count[0], top_comp_count[0], top_comp_count[1])
        print ('Class', pred_cl, 'Others average drop:', top_comp_drop[1]/top_comp_count[1])

    else:
        for i in range(n_components):
            #if imp_max_comp[i] > 0.01:
            if imp_max_comp[i] > thresh and (not select_random):
                comp[i], ratio[i] = select_component(i, batch[3].cpu().data.numpy(), embed[0].cpu().data.numpy(), W_mat)
                residual_lgmag_spec = residual_lgmag_spec - comp[i]
                enhanced_lgmag_spec = enhanced_lgmag_spec + softmask_weights[i] * comp[i]
            elif select_random:
                if i in comps:
                    comp[i], ratio[i] = select_component(i, batch[3].cpu().data.numpy(), embed[0].cpu().data.numpy(), W_mat)
                    residual_lgmag_spec = residual_lgmag_spec - comp[i]
        new_time_signal = core.istft(get_MS(residual_lgmag_spec)*batch[-2], hop_length=dataload.hop)
        #energy_ratio = np.sqrt(np.sum(inp_mag_spec))/np.sqrt(np.sum(enhanced_mag_spec))
        #print ('Energy ratio:', energy_ratio)
        #new_time_signal = core.istft(energy_ratio*enhanced_mag_spec*batch[-2], hop_length=dataload.hop)
        original_time_signal = core.istft(inp_mag_spec*batch[-2], hop_length=dataload.hop)
        res = original_time_signal - new_time_signal
        #plt.plot(original_time_signal, color='green')
        #plt.plot((original_time_signal - new_time_signal), color='red')
        #plt.plot(new_time_signal, color='black')
        #plt.show()
        #print ('Residual stats:', res.max(), res.mean(), res.min(), (res**2).mean())
        #new_time_signal = new_time_signal / new_time_signal.max()
        new_Xs = core.stft(new_time_signal, n_fft=dataload.nfft, hop_length=dataload.hop)
        new_Xmel = librosa.feature.melspectrogram(sr=dataload.sr, S=np.abs(new_Xs), n_fft=dataload.nfft, hop_length=dataload.hop, n_mels=dataload.nmel)
        new_Xlgmel = librosa.power_to_db(new_Xmel).T
        new_inp = torch.as_tensor(new_Xlgmel).unsqueeze(0).float().to(device).unsqueeze(0)
        output2, inter2 = f(new_inp)
        #print ('Logit values before and after', output[0][pred_cl].item(), output2[0][pred_cl].item())
        prob2 = nn.Softmax(dim=1)(output2)
        prob = nn.Softmax(dim=1)(output)
        #print ('Probability values before and after', output[0][pred_cl].item(), output2[0][pred_cl].item())
        #print ('Relative Logit drop for Class ', pred_cl, ((output[0][pred_cl] - output2[0][pred_cl]) / output[0][pred_cl]).item())

        #return ((output[0][pred_cl] - output2[0][pred_cl]) / output[0][pred_cl]).item()
        return [((output[0][pred_cl] - output2[0][pred_cl])).item(), ((prob[0][pred_cl] - prob2[0][pred_cl])).item(), ((prob[0][pred_cl] - prob2[0][pred_cl]) / prob[0][pred_cl]).item(), pred_cl, int( pred_cl in (-out_interpreter).argsort(dim=1)[0][:1] ), int( pred_cl in (-out_interpreter).argsort(dim=1)[0][:3] ), int( pred_cl in (-out_interpreter).argsort(dim=1)[0][:5] ), np.sum(imp_max_comp > thresh)] # [Absolute logit drop, Absolute probability drop, Relative probability drop, predicted class, top-1 agreement, top-3 agreement, top-5 agreement, num_relevant_components]

    
    

def explanation_multilabel(dataload, idx, f, H, W, h, model_name, save_files=True):
    import librosa.core as core
    import soundfile as sf

    path_info_sample = 'output/' + dataset + '_output/explanation_' + str(model_name) + '/info_sample/'

    if save_files:
        try:
            os.mkdir(path_info_sample + 'Sample_' + str(idx))
        except:
            print ('Folder for this sample already existing. May overwrite files or create additional unwanted ones')
    path_info_sample = path_info_sample + 'Sample_' + str(idx) + '/'

    batch = dataload[idx]
    data, target, file_name = batch[0], batch[1], batch[-1]
    #print (file_name)
    #plt.imshow(data[0].cpu().data.numpy())
    #plt.show()
    output, inter = f(data.to(device).unsqueeze(0))
    embed = H(inter)
    out_interpreter = h(embed)
    pred_int = (nn.Sigmoid()(out_interpreter[0]).cpu().data.numpy() > 0.3).astype(int)
    pred_f = (nn.Sigmoid()(output[0]).cpu().data.numpy() > class_thresh).astype(int)
    all_pred_classes = np.where(pred_f == 1)[0]
    print ('Interpreter output:', pred_int)
    print ('Classifier output:', pred_f )
    print ('Target output:', target.cpu().data.numpy())

    rec_data = W(embed)
    log_rec_spec = rec_data[0].cpu().data.numpy()
    spec_shape = log_rec_spec.shape
    comp = np.ones([n_components, spec_shape[0], spec_shape[1]])
    ratio = np.ones([n_components, spec_shape[0], spec_shape[1]])
    rec_spec = np.exp(log_rec_spec) - 1 
    rec_time = core.istft(rec_spec*batch[-2], hop_length=dataload.hop)

    W_mat = W.return_W()
    weights = h.fc1.weight
    #pool = nn.AdaptiveAvgPool1d(pool_frames)
    pool = nn.AdaptiveMaxPool1d(pool_frames)
    pool2 = nn.MaxPool1d(pool_frames, stride=pool_frames)
    #pooled_activ = torch.flatten(pool(embed), 1)  # For linear layer as theta function
    pooled_activ = h.return_pooled_activ(embed) # For the case when attention based theta function 

    imp_cl_list = [] # To store importances of components for each interpretation

    for pred_cl in all_pred_classes:
        imp_full = pooled_activ + 0
        imp_full[0] = pooled_activ[0] * weights[int(pred_cl)]
        imp_full = imp_full / imp_full.max()
        imp_max_comp = pool2(imp_full.unsqueeze(0))[0, 0].cpu().data.numpy() # unsqueeze was done for performing pooling
        activ_max_comp = pool2(pooled_activ.unsqueeze(0))[0, 0].cpu().data.numpy() # unsqueeze was done for performing pooling
        imp_cl_list.append( (imp_max_comp, pred_cl) )
        #print (imp_full.shape, imp_max_comp.shape)
        #print (expl)
        imp_components = (-1*imp_max_comp).argsort()[:5]
        print ('\nClass:', pred_cl, 'Important components:', imp_components)
        print ('Component relevance:', imp_max_comp[imp_components])
        print ('Component max activation:', activ_max_comp[imp_components])
        print ('Component mean activation:', embed[0].cpu().data.numpy()[imp_components].mean(axis=1))
        expl_comp = comp[0] * 0.0
        ratio_comp = ratio[0] * 0.0
        for i in imp_components:
            comp[i], ratio[i] = select_component(i, batch[3].cpu().data.numpy(), embed[0].cpu().data.numpy(), W_mat)
            if imp_max_comp[i] > 0.3:
                expl_comp += comp[i]
                ratio_comp += ratio[i]
            rec_comp = core.istft(get_MS(comp[i])*batch[-2], hop_length=dataload.hop)
            #if save_files:
            #    sf.write(path_info_sample + 'Samp' + str(idx) + '_' + 'Comp' + str(i) + '.wav', rec_comp, dataload.sr)

        #rec_expl_comp = core.griffinlim(expl_comp, hop_length=dataload.hop) # for amnist
        rec_expl_comp = core.istft(get_MS(expl_comp)*batch[-2], hop_length=dataload.hop)
        if save_files:
            sf.write(path_info_sample + 'Expl_' + str(idx) + 'Cl_' + str(pred_cl) + '.wav', rec_expl_comp, dataload.sr)     

    if save_files:
        plt.imshow(embed[0].cpu().data.numpy(), origin='lower')
        plt.colorbar()
        plt.savefig(path_info_sample + 'H_' + str(idx))
        plt.close()

        #sf.write(path_info_sample + 'reconstruct_' + str(idx) + '.wav', rec_time, dataload.sr)
        #orig_time, fs = sf.read(file_name)
        inp_mag_spec = np.exp(batch[3].cpu().data.numpy()) - 1
        orig_time = core.istft(inp_mag_spec*batch[-2], hop_length=dataload.hop)
        #orig_time = core.griffinlim(batch[2].cpu().data.numpy(), hop_length=dataload.hop)
        sf.write(path_info_sample + 'original_' + str(idx) + '.wav', orig_time, dataload.sr)

    return imp_cl_list



def class_comp_relavance(dataloader, f, H, h, device, multilabel=True):
    pool = nn.AdaptiveMaxPool1d(pool_frames)
    pool2 = nn.MaxPool1d(pool_frames, stride=pool_frames)
    rel = np.zeros([n_classes, n_components])
    cnt_class = np.zeros([n_classes])
    weights = h.fc1.weight

    for batch_info in dataloader:
        data, target = batch_info[0].to(device), batch_info[1].to(device)
        n_samp = int(data.shape[0])
        output, inter = f(data.to(device))
        embed = H(inter)
        out_interpreter = h(embed)        
        z = torch.flatten(pool(embed), 1)

        if multilabel:
            pred_int = (nn.Sigmoid()(out_interpreter) > 0.1).cpu().data.numpy().astype(int)
            pred_f = (nn.Sigmoid()(output) > 0.1).cpu().data.numpy().astype(int)
        else:
            pred_int = out_interpreter.argmax(dim=1).cpu().data.numpy().astype(int)

        for k in range(n_samp):
            if multilabel:
                pred_classes = np.where(pred_int[k] == 1)[0]
                cnt_class[pred_classes] += 1
                #print (pred_classes)
                for pred_cl in pred_classes:
                    imp_full = z[k] + 0
                    imp_full = z[k] * weights[int(pred_cl)]
                    imp_max_comp = pool2(imp_full.unsqueeze(0).unsqueeze(0))[0, 0].cpu().data.numpy()
                    imp_full = imp_max_comp / imp_max_comp.max()
                    #print (imp_full.shape, rel[pred_cl].shape)
                    rel[pred_cl] = rel[pred_cl] + imp_full

    for j in range(n_classes):
        rel[j] = rel[j] / cnt_class[j]        

    return rel
                    


def explanation_multiclass(dataload, idx, f, H, W, h, model_name):
    import librosa.core as core
    import soundfile as sf

    path_info_sample = 'output/' + dataset + '_output/explanation_' + str(model_name) + '/info_sample/'

    try:
        os.mkdir(path_info_sample + 'Sample_' + str(idx))
    except:
        print ('Folder for this sample already existing. May overwrite files or create additional unwanted ones')
    path_info_sample = path_info_sample + 'Sample_' + str(idx) + '/'

    batch = dataload[idx]
    data, target, file_name = batch[0], batch[1], batch[-1]
    #print (file_name)
    #plt.imshow(data[0].cpu().data.numpy())
    #plt.show()
    output, inter = f(data.to(device).unsqueeze(0))
    embed = H(inter)
    out_interpreter = h(embed)
    pred_class = int(output.argmax(dim=1)[0])
    print ('Probability of predicted class:', nn.Softmax(dim=1)(output)[0, pred_class])
    #if not (pred_class in (-out_interpreter).argsort(dim=1)[0][:5]):
    #    print ('No top-5 fidelity!')

    print ('Predicted class by interpreter:', int(out_interpreter.argmax(dim=1)[0]))
    print ('Predicted class by classifier:', int(output.argmax(dim=1)[0]) )
    print ('Target class:', int(target))
    rec_data = W(embed)
    log_rec_spec = rec_data[0].cpu().data.numpy()
    #plt.imshow(log_rec_spec)
    #plt.show()
    spec_shape = log_rec_spec.shape
    comp = np.ones([n_components, spec_shape[0], spec_shape[1]])
    ratio = np.ones([n_components, spec_shape[0], spec_shape[1]])
    #comp30, ratio = select_component(30, log_rec_spec, embed[0].cpu().data.numpy(), W.W.cpu().data.numpy())
    rec_spec = np.exp(log_rec_spec) - 1 
    #rec_time = core.griffinlim(rec_spec, hop_length=35)
    #rec_time = core.griffinlim(rec_spec, hop_length=dataload.hop)
    rec_time = core.istft(rec_spec*batch[-2], hop_length=dataload.hop)
    W_mat = W.return_W()
    #plt.imshow(W_mat)
    #plt.show()
    weights = h.fc1.weight
    #plt.imshow(weights)
    #plt.show()
    pool = nn.AdaptiveAvgPool1d(pool_frames)
    pool2 = nn.MaxPool1d(pool_frames, stride=pool_frames)
    z = h.return_pooled_activ(embed) # For the case when attention based theta function     
    # z = torch.flatten(pool(embed), 1)  # For case when max/avg pool based theta function

    imp_full = z + 0
    imp_full[0] = z[0] * weights[int(pred_class)]
    imp_full = imp_full / imp_full.max()
    imp_max_comp = pool2(imp_full.unsqueeze(0))[0, 0].cpu().data.numpy() # unsqueeze was done for performing pooling
    activ_max_comp = pool2(z.unsqueeze(0))[0, 0].cpu().data.numpy() # unsqueeze was done for performing pooling
    
    #print (imp_full.shape, imp_max_comp.shape)
    #print (expl)
    imp_components = (-1*imp_max_comp).argsort()[:5]
    print ('Class:', pred_class, 'Important components:', imp_components)
    print ('Component relevance:', imp_max_comp[imp_components])
    print ('Component max activation:', activ_max_comp[imp_components])
    print ('Component mean activation:', embed[0].cpu().data.numpy()[imp_components].mean(axis=1))
    
    expl_comp = comp[0] * 0.0
    ratio_comp = ratio[0] * 0.0
    for i in imp_components:
        comp[i], ratio[i] = select_component(i, batch[3].cpu().data.numpy(), embed[0].cpu().data.numpy(), W_mat)
        #plt.imshow(ratio[i])
        #plt.show()
        #plt.imshow(comp[i])
        #plt.show()
        if imp_max_comp[i] > 0.2:
            expl_comp += comp[i]
            ratio_comp += ratio[i]
        #rec_comp = core.griffinlim(comp[i], hop_length=35) # for amnist
        #rec_comp = core.griffinlim(comp[i], hop_length=dataload.hop) # griffin lim for msdb
        rec_comp = core.istft(get_MS(comp[i])*batch[-2], hop_length=dataload.hop)
        #sf.write(path_info_sample + 'Samp' + str(idx) + '_' + 'Comp' + str(i) + '.wav', rec_comp, dataload.sr)
    #rec_expl_comp = core.griffinlim(expl_comp, hop_length=dataload.hop) # for amnist
    rec_expl_comp = core.istft(get_MS(expl_comp)*batch[-2], hop_length=dataload.hop)
    sf.write(path_info_sample + 'Expl_' + str(idx) + '.wav', rec_expl_comp, dataload.sr)     
    #rec_comp30 = core.griffinlim(comp30, hop_length=35)
    #print (rec_time.shape)


    plt.imshow(embed[0].cpu().data.numpy(), origin='lower')
    plt.colorbar()
    plt.savefig(path_info_sample + 'H_' + str(idx))
    plt.close()

    inp_mag_spec = np.exp(batch[3].cpu().data.numpy()) - 1
    orig_time = core.istft(inp_mag_spec*batch[-2], hop_length=dataload.hop)
    #orig_time = core.griffinlim(batch[2].cpu().data.numpy(), hop_length=dataload.hop)
    sf.write(path_info_sample + 'original_' + str(idx) + '.wav', orig_time, dataload.sr)
    return


def select_component(idx, inp_lg_spec, H, W):
    # Selects the contribution of component j (j=idx) from the given input log magnitude spectrogram
    # Assume integer/numpy arrays for all input arguments, W of shape N_FREQ x N_COMP and H of N_COMP x N_TIME
    # Do W.abs() to force positive values
    W_mat = np.abs(W)
    ratio = np.outer(W_mat[:, idx], H[idx]) / (0.000001 + np.dot(W_mat, H))
    #comp = np.exp(inp_lg_spec * ratio) - 1
    comp = inp_lg_spec * ratio
    return comp, ratio

def get_MS(X):
    return (np.exp(X) - 1)

    
    

def loss_cce(prediction, target):
    # Assume shape of batch_size x n_classes with unnormalized class scores for prediction and target
    # Compute softmax to get class probabilities and then take compute cross entropy loss
    p = nn.Softmax(dim=1)(prediction)
    t = nn.Softmax(dim=1)(target)
    loss = (p.log() * -t).sum(dim=1).mean()
    return loss

def loss_cce_v2(prediction, target):
    p = nn.Sigmoid()(prediction)
    t = nn.Sigmoid()(target)
    loss = ( (p.log() * -t) + ((1-p).log() * (t-1)) ).mean()
    return loss

def loss_cce_v3(prediction, target):
    eps = 0.0000000001
    p = nn.Sigmoid()(prediction) + eps
    t = nn.Sigmoid()(target)
    loss = ( (p.log() * -t) + ((1-p + eps).log() * (t-1)) ).mean()
    return loss



def test(f, H, h, W, device, test_loader, lmbd_acc=1.0, lmbd_rec=1.0, lmbd_expl=1.0, lmbd_spa=1.0):
    # f denotes the main classifier
    # H maps intermediate layer of f to NMF-H:
    test_loss_acc = 0.0
    test_loss_rec = 0.0
    test_loss_exp = 0.0
    test_loss_spa = 0.0
    all_pred, all_target = [], []
    if dataset in ['sust']:
        auprc_t = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        auprc_t.reverse()
        auprc_t = np.array(auprc_t)
        tp = np.zeros([auprc_t.shape[0], n_classes])
        tn = np.zeros([auprc_t.shape[0], n_classes])
        fp = np.zeros([auprc_t.shape[0], n_classes])
        fn = np.zeros([auprc_t.shape[0], n_classes])
    f.eval(), H.eval(), h.eval(), W.eval()
    print (lmbd_acc, lmbd_rec, lmbd_expl, lmbd_spa)
    batch_idx = -1
    correct = 0
    exp_overlap = 0
    batch_size = int(test_loader.batch_size)
    for batch_info in test_loader:
        batch_idx += 1
        data, target = batch_info[0].to(device), batch_info[1].to(device)
        output, inter = f(data)
        loss_acc = criterion(output, target)       
        embed = H(inter)
        loss_spa = nn.L1Loss()(embed, torch.zeros(embed.shape).to(device))
        rec_data = W(embed)
        loss_rec = criterion2(rec_data, batch_info[3].to(device)) # MSE loss, L1 for CIFAR10
        expl = h(embed)

        if dataset in ['sust']:
            loss_expl = loss_cce_v3(expl, output.detach())
        else:
            loss_expl = loss_cce(expl, output.detach())

        if dataset in ['sust']:
            pred = (nn.Sigmoid()(output) > torch.tensor(class_thresh).to(device))
            correct += (pred == target).sum().item()
            exp_overlap += (pred == (nn.Sigmoid()(expl) > 0.5)).sum().item()
            all_pred.append(pred.cpu().data.numpy())
            all_target.append(target.cpu().data.numpy())
            for k in range(auprc_t.shape[0]): # Iterating over different thresholds
                pred_int = (nn.Sigmoid()(expl) > auprc_t[k])
                for j in range(n_classes):
                    tp[k, j] += ((pred_int[:, j] == 1) * (pred[:, j] == 1)).sum().item()
                    tn[k, j] += ((pred_int[:, j] == 0) * (pred[:, j] == 0)).sum().item()
                    fp[k, j] += ((pred_int[:, j] == 1) * (pred[:, j] == 0)).sum().item()
                    fn[k, j] += ((pred_int[:, j] == 0) * (pred[:, j] == 1)).sum().item()
        else:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            exp_overlap += float(torch.sum(expl.argmax(dim=1) == output.argmax(dim=1)))

        test_loss_acc += loss_acc.item()
        test_loss_rec += loss_rec.item()
        test_loss_spa += loss_spa.item()
        test_loss_exp += loss_expl.item()

    test_loss_acc = test_loss_acc / (len(test_loader.dataset) / batch_size)
    test_loss_rec = test_loss_rec / (len(test_loader.dataset) / batch_size)
    test_loss_exp = test_loss_exp / (len(test_loader.dataset) / batch_size)
    test_loss_spa = test_loss_spa / (len(test_loader.dataset) / batch_size)
    
    if dataset in ['sust']:
        mac_auprc, mic_auprc, max_f1 = calc_PRC_metrics(tp, fp, fn)
        print ('Fidelity Macro_AUPRC: ', np.mean(mac_auprc))
        print ('Fidelity Macro AUPRC full: ', mac_auprc)
        print ('Fidelity Micro AUPRC: ', mic_auprc)
        print ('Fidelity F1 max', max_f1)

    if dataset in ['sust']:
        print('\nTest set: Average loss: {:.4f}, Acc: {}/{} ({:.0f}%)   Rec: {:.5f}   Exp: {:.5f})   ExpAgree: {:.5f}\n'.format(
        test_loss_acc, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), test_loss_rec, test_loss_exp, 100. * exp_overlap / len(test_loader.dataset)))
        return [100. * correct / len(test_loader.dataset), test_loss_acc, test_loss_rec, test_loss_exp, exp_overlap, test_loss_spa, tp, fp, fn, all_pred, all_target]
    else:
        print('\nTest set: Average loss: {:.4f}, Acc: {}/{} ({:.0f}%)   Rec: {:.5f}   Exp: {:.5f})   ExpAgree: {:.5f}\n'.format(
        test_loss_acc, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), test_loss_rec, test_loss_exp, 100. * exp_overlap / len(test_loader.dataset)))
        return [100. * correct / len(test_loader.dataset), test_loss_acc, test_loss_rec, test_loss_exp, exp_overlap, test_loss_spa]



def train(f, H, h, W, device, train_loader, optimizer, epoch, lmbd_rec=0, lmbd_expl=0, lmbd_spa=0):
    # f denotes the main classifier
    # H maps intermediate layer of f to NMF-H:
    train_loss_acc = 0.0
    train_loss_rec = 0.0
    train_loss_exp = 0.0
    train_loss_spa = 0.0
    f.eval(), H.train(), h.train(), W.train()
    print (lmbd_rec, lmbd_expl, lmbd_spa)
    batch_idx = -1
    W_init = W.W + 0.0
    batch_size = int(train_loader.batch_size)
    for batch_info in train_loader:
        batch_idx += 1
        data, target = batch_info[0].to(device), batch_info[1].to(device)
        batch_size = int(target.shape[0])
        optimizer.zero_grad()
        output, inter = f(data)
        #print ('Event 1')
        loss_acc = criterion(output, target)       
        embed = H(inter)
        loss_spa = nn.L1Loss()(embed, torch.zeros(embed.shape).to(device))
        rec_data = W(embed)
        loss_rec = criterion2(rec_data, batch_info[3].to(device))
        expl = h(embed)

        if dataset in ['sust']:
            loss_expl = loss_cce_v3(expl, output.detach())
        else:
            loss_expl = loss_cce(expl, output.detach())

        #loss_expl = criterion(expl, target)
        #loss_cont = criterion2(embed[:, :, 0:-1], embed[:, :, 1:].detach())
        #print ((W.W - W_init).norm())
        #print ("Things computed")
        loss = lmbd_rec*loss_rec + lmbd_expl*loss_expl  + lmbd_spa*loss_spa #+ 10.0*loss_cont
        #loss  =lmbd_expl * loss_expl
        #print (loss)
        loss.backward()
        optimizer.step()
        #print ('Event 2')

        train_loss_acc += loss_acc.item()
        train_loss_rec += loss_rec.item()
        train_loss_spa += loss_spa.item()
        train_loss_exp += loss_expl.item()
        ''' 
        if batch_idx % 10 == 0:
            W_list.append(W.W.cpu().data.numpy())
            H_list.append(H(f(sample_data.to(device))[1]).cpu().data.numpy())
        '''
        if batch_idx % 45 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.5f}\t Rec: {:.5f}\t Exp: {:.5f}\t Spa: {:.5f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss_acc.item(), loss_rec.item(), loss_expl.item(), loss_spa.item()))

    train_loss_acc = train_loss_acc / (len(train_loader.dataset) / batch_size)
    train_loss_rec = train_loss_rec / (len(train_loader.dataset) / batch_size)
    train_loss_exp = train_loss_exp / (len(train_loader.dataset) / batch_size)
    train_loss_spa = train_loss_spa / (len(train_loader.dataset) / batch_size)
    #train_loss_ent = train_loss_ent / (len(train_loader.dataset) / batch_size)

    return [train_loss_acc, train_loss_rec, train_loss_exp, train_loss_spa]


def compute_multilabel_threshold(f, H, h, val_loader):
    auprc_t = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    auprc_t.reverse()
    auprc_t = np.array(auprc_t)
    tp = np.zeros([auprc_t.shape[0], n_classes])
    tn = np.zeros([auprc_t.shape[0], n_classes])
    fp = np.zeros([auprc_t.shape[0], n_classes])
    fn = np.zeros([auprc_t.shape[0], n_classes])

    for batch_info in val_loader:
        data, target = batch_info[0].to(device), batch_info[1].to(device)
        output, inter = f(data)
        loss = criterion(output, target)
        for k in range(auprc_t.shape[0]): # Iterating over different thresholds
            pred = (nn.Sigmoid()(output) > auprc_t[k])
            for j in range(n_classes):
                tp[k, j] += ((pred[:, j] == 1) * (target[:, j] == 1)).sum().item()
                tn[k, j] += ((pred[:, j] == 0) * (target[:, j] == 0)).sum().item()
                fp[k, j] += ((pred[:, j] == 1) * (target[:, j] == 0)).sum().item()
                fn[k, j] += ((pred[:, j] == 0) * (target[:, j] == 1)).sum().item()

    mac_p = (tp + 1e-9)/ (tp + fp + 1e-9)
    mac_r = (tp + 1e-9)/ (tp + fn + 1e-9)
    mac_f1 = 2 * mac_p * mac_r/(mac_p + mac_r)
    class_thresh = auprc_t[np.argmax(mac_f1, axis=0)]
    print (mac_f1)
    return class_thresh
    #np.argmax(mac_f1, )


def train_classifier(f, device, train_loader, test_loader, opt_f, N_EPOCH=21, dataset_name='sust'):    
    for epoch in range(1, N_EPOCH+1):
        f.train()
        batch_idx = -1
        train_loss_progress = []
        test_loss_progress = []
        test_acc_progress = [0.0]
        train_loss_acc, test_loss_acc = 0.0, 0.0
        correct = 0
        if dataset == 'sust':
            auprc_t = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
        auprc_t.reverse()
        auprc_t = np.array(auprc_t)
        tp = np.zeros([auprc_t.shape[0], n_classes])
        tn = np.zeros([auprc_t.shape[0], n_classes])
        fp = np.zeros([auprc_t.shape[0], n_classes])
        fn = np.zeros([auprc_t.shape[0], n_classes])

        for batch_info in train_loader:
            batch_idx += 1
            data, target = batch_info[0].to(device), batch_info[1].to(device)
            if batch_idx < 1:
                batch_size = int(target.shape[0])
            opt_f.zero_grad()
            output, inter = f(data)
             
            loss = criterion(output, target)
            #loss = nn.BCEWithLogitsLoss(weight)(output, target) 
            loss.backward()
            opt_f.step()      
            train_loss_acc += loss.item()
            if batch_idx % 100 == 0:
                print('Classifier Train Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.5f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        train_loss_progress.append( train_loss_acc / (len(train_loader.dataset) / batch_size) )

        f.eval()
        for batch_info in test_loader:
            batch_idx += 1
            data, target = batch_info[0].to(device), batch_info[1].to(device)
            if batch_idx < 1:
                batch_size = int(target.shape[0])
            output, inter = f(data)
            if dataset in ['sust']:
                for k in range(auprc_t.shape[0]): # Iterating over different thresholds
                    pred = (nn.Sigmoid()(output) > auprc_t[k])
                    for j in range(n_classes):
                        tp[k, j] += ((pred[:, j] == 1) * (target[:, j] == 1)).sum().item()
                        tn[k, j] += ((pred[:, j] == 0) * (target[:, j] == 0)).sum().item()
                        fp[k, j] += ((pred[:, j] == 1) * (target[:, j] == 0)).sum().item()
                        fn[k, j] += ((pred[:, j] == 0) * (target[:, j] == 1)).sum().item()
                
            else:
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
            test_loss_acc += loss.item()

        test_loss = test_loss_acc / (len(test_loader.dataset) / batch_size)
        test_loss_progress.append(test_loss)  
        test_acc = 100. * correct / len(test_loader.dataset)
        print ('\nClassifier Test Epoch: ', epoch, 'Loss:', test_loss, 'Acc:', test_acc, '\n')
        if dataset in ['sust']:
            mac_auprc, mic_auprc, max_f1 = calc_PRC_metrics(tp, fp, fn)
            print ('Macro_AUPRC: ', np.mean(mac_auprc))
            print ('Macro AUPRC full: ', mac_auprc)
            print ('Micro AUPRC: ', mic_auprc)
            print ('F1 max', max_f1)
        
        test_acc_progress.append(test_acc)
        #if test_acc > max(test_acc_progress):
        best_f = f.eval()

    return train_loss_progress, test_loss_progress, test_acc_progress[1:], best_f


def test_classifier(f, device, test_loader):
    test_loss_acc = 0.0
    correct = 0
    if dataset == 'sust':
        auprc_t = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
    auprc_t.reverse()
    auprc_t = np.array(auprc_t)
    tp = np.zeros([auprc_t.shape[0], n_classes])
    tn = np.zeros([auprc_t.shape[0], n_classes])
    fp = np.zeros([auprc_t.shape[0], n_classes])
    fn = np.zeros([auprc_t.shape[0], n_classes])
    batch_idx = -1
    for batch_info in test_loader:
        batch_idx += 1
        data, target = batch_info[0].to(device), batch_info[1].to(device)
        if batch_idx < 1:
            batch_size = int(target.shape[0])
        output, inter = f(data)
        if dataset in ['sust']:
            for k in range(auprc_t.shape[0]): # Iterating over different thresholds
                pred = (nn.Sigmoid()(output) > auprc_t[k])
                for j in range(n_classes):
                    tp[k, j] += ((pred[:, j] == 1) * (target[:, j] == 1)).sum().item()
                    tn[k, j] += ((pred[:, j] == 0) * (target[:, j] == 0)).sum().item()
                    fp[k, j] += ((pred[:, j] == 1) * (target[:, j] == 0)).sum().item()
                    fn[k, j] += ((pred[:, j] == 0) * (target[:, j] == 1)).sum().item()
        else:
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        loss = criterion(output, target)
        test_loss_acc += loss.item()

    test_loss = test_loss_acc / (len(test_loader.dataset) / batch_size)  
    test_acc = 100. * correct / len(test_loader.dataset)
    print ('\nLoss:', test_loss, 'Acc:', test_acc, '\n')
    if dataset in ['sust']:
        mac_auprc, mic_auprc, max_f1 = calc_PRC_metrics(tp, fp, fn)
        print ('Macro_AUPRC: ', np.mean(mac_auprc))
        print ('Macro AUPRC full: ', mac_auprc)
        print ('Micro AUPRC: ', mic_auprc)
        print ('F1 max', max_f1)
    return


def calc_PRC_metrics(tp, fp, fn):
    # Assume all three of shapes (n_threshold x n_classes)
    # Calculate classwise auprc and also auprc of overall precision, recall, and also max F1 among all thresholds
    mac_p = (tp + 1e-9)/ (tp + fp + 1e-9)
    mac_r = (tp)/ (tp + fn)
    mac_auprc = np.zeros([tp.shape[1]])
    for k in range(tp.shape[1]): # Iterating over classes
        class_p = np.array([1.0] + list(mac_p[:, k]) + [0.0])
        class_r = np.array([0.0] + list(mac_r[:, k]) + [1.0])
        mac_auprc[k] = auc(class_r, class_p)
    mic_p = (1e-9 + np.sum(tp, axis=1)) / (1e-9 + np.sum(tp, axis=1) + np.sum(fp, axis=1))
    mic_r = (1e-9 + np.sum(tp, axis=1)) / (1e-9 + np.sum(tp, axis=1) + np.sum(fn, axis=1))
    mic_p = np.array([1.0] + list(mic_p) + [0.0])
    mic_r = np.array([0.0] + list(mic_r) + [1.0])
    mic_f1 = 2 * mic_p * mic_r/(mic_p + mic_r)
    mic_auprc = auc(mic_r, mic_p)
    return mac_auprc, mic_auprc, mic_f1.max()



def save_model(weight_str, name, save_best=False):
    # don't add .pt in the name
    model_dict = {}
    model_dict['f_state_dict'] = f.state_dict()
    model_dict['H_state_dict'] = H.state_dict()
    model_dict['h_state_dict'] = h.state_dict()
    model_dict['W_state_dict'] = W.state_dict()
    model_dict['last_weights'] = weight_str
    model_dict['train_info'] = train_epoch_info
    model_dict['test_info'] = test_epoch_info
    model_dict['NMF-Component initializing file'] = init_nmf_filename
    torch.save(model_dict, 'output/' + dataset + '_output/' + name + '.pt')
    if save_best:
        model_dict['H_state_dict'] = best_H.state_dict()
        model_dict['h_state_dict'] = best_h.state_dict()
        model_dict['W_state_dict'] = best_W.state_dict()
        torch.save(model_dict, 'output/' + dataset + '_output/' + 'best_' + name + '.pt')
    return model_dict


def subjective_samp_select(all_pred, all_target, samp_per_class=2):
    indices = []
    if dataset in ['sust']:
        for i in [0, 1, 2, 4, 5, 6, 7]:
            idx = np.where((all_pred[:, i] * all_target[:, i]) == 1)[0]
            select = np.random.choice(idx, size=samp_per_class, replace=False)
            indices = indices + list(select)
    return indices


def cluster_sust(imp_cl_full):
    n_samples = len(imp_cl_full)
    rel_all = []
    cl_all = []
    for idx in range(n_samples):
        rel_all.append(imp_cl_full[idx][0])
        cl_all.append(imp_cl_full[idx][1])
    rel_all = np.array(rel_all)
    cl_all = np.array(cl_all)
    from sklearn.manifold import TSNE
    rel_embed = TSNE(n_components=2).fit_transform(rel_all)
    cl_dict = {0: 'Engine', 1: 'Powered-Saw', 2: 'Machinery-Impact', 4: 'AlertSignal', 5: 'Music', 6: 'Human', 7: 'Dog'}
    cmap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    for i in list(cl_dict.keys()):
        idx = np.where(cl_all == i)[0]
        plt.scatter(rel_embed[idx, 0], rel_embed[idx, 1], c=cmap[i], label=cl_dict[i])
    #plt.scatter(rel_embed[:, 0], rel_embed[:, 1], c=cl_all)
    plt.legend()
    plt.show()
    return rel_all, cl_all

def overlap_exp(dataload, idx1, idx2, lambda2, f, H, W, h, model_name):
    import librosa.core as core
    import soundfile as sf

    path_info_sample = 'output/' + dataset + '_output/explanation_' + str(model_name) + '/overlap/'

    try:
        os.mkdir(path_info_sample + 'Sample_' + str(idx1) + '_' + str(idx2))
    except:
        print ('Folder for this sample already existing. May overwrite files or create additional unwanted ones')
    path_info_sample = path_info_sample + 'Sample_' + str(idx1) + '_' + str(idx2) + '/'

    batch = dataload.overlap_two(idx1, idx2, lambda2)
    data, target, file_name = batch[0], batch[1], batch[-1]
    output, inter = f(data.to(device).unsqueeze(0))
    embed = H(inter)
    out_interpreter = h(embed)

    print ('Predicted class by interpreter:', int(out_interpreter.argmax(dim=1)[0]))
    print ('Predicted class by classifier:', int(output.argmax(dim=1)[0]) )
    print ('Target class:', int(target))
    pred_class = int(target)
    #print ('Probability of target class:', nn.Softmax(dim=1)(output)[0, pred_class].item())
    rec_data = W(embed)
    log_rec_spec = rec_data[0].cpu().data.numpy()
    spec_shape = log_rec_spec.shape
    comp = np.ones([n_components, spec_shape[0], spec_shape[1]])
    ratio = np.ones([n_components, spec_shape[0], spec_shape[1]])
    rec_spec = np.exp(log_rec_spec) - 1 
    rec_time = core.istft(rec_spec*batch[-2], hop_length=dataload.hop)
    W_mat = W.return_W()
    weights = h.fc1.weight
    pool = nn.AdaptiveAvgPool1d(pool_frames)
    pool2 = nn.MaxPool1d(pool_frames, stride=pool_frames)
    z = h.return_pooled_activ(embed) # For the case when attention based theta function     
    # z = torch.flatten(pool(embed), 1)  # For case when max/avg pool based theta function

    imp_full = z + 0
    imp_full[0] = z[0] * weights[int(pred_class)]
    imp_full = imp_full / imp_full.max()
    imp_max_comp = pool2(imp_full.unsqueeze(0))[0, 0].cpu().data.numpy() # unsqueeze was done for performing pooling
    activ_max_comp = pool2(z.unsqueeze(0))[0, 0].cpu().data.numpy() # unsqueeze was done for performing pooling

    imp_components = (-1*imp_max_comp).argsort()[:5]
    print ('Class:', pred_class, 'Important components:', imp_components)
    print ('Component relevance:', imp_max_comp[imp_components])
    print ('Component max activation:', activ_max_comp[imp_components])
    print ('Component mean activation:', embed[0].cpu().data.numpy()[imp_components].mean(axis=1))
    
    expl_comp = comp[0] * 0.0
    ratio_comp = ratio[0] * 0.0
    for i in imp_components:
        comp[i], ratio[i] = select_component(i, batch[3].cpu().data.numpy(), embed[0].cpu().data.numpy(), W_mat)

        if imp_max_comp[i] > 0.35:
            expl_comp += comp[i]
            ratio_comp += ratio[i]

        rec_comp = core.istft(get_MS(comp[i])*batch[-2], hop_length=dataload.hop)
        #sf.write(path_info_sample + 'Samp' + str(idx1) + '_' + str(idx2) + '_' + 'Comp' + str(i) + '.wav', rec_comp, dataload.sr)
    #rec_expl_comp = core.griffinlim(expl_comp, hop_length=dataload.hop) # for amnist
    rec_expl_comp = core.istft(get_MS(expl_comp)*batch[-2], hop_length=dataload.hop)
    sf.write(path_info_sample + 'Expl_' + str(idx1) + '_' + str(idx2) + '.wav', rec_expl_comp, dataload.sr)     
    #rec_comp30 = core.griffinlim(comp30, hop_length=35)
    #print (rec_time.shape)

    plt.imshow(embed[0].cpu().data.numpy(), origin='lower')
    plt.colorbar()
    plt.savefig(path_info_sample + 'H_' + str(idx1) + '_' + str(idx2))
    plt.close()

    #sf.write(path_info_sample + 'reconstruct_' + str(idx1) + '_' + str(idx2) + '.wav', rec_time, dataload.sr)
    inp_mag_spec = np.exp(batch[3].cpu().data.numpy()) - 1
    orig_time = core.istft(inp_mag_spec*batch[-2], hop_length=dataload.hop)
    sf.write(path_info_sample + 'original_' + str(idx1) + '_' + str(idx2) + '.wav', orig_time, dataload.sr)

    batch = dataload[idx1]
    inp_mag_spec = np.exp(batch[3].cpu().data.numpy()) - 1
    orig_time_2 = core.istft(inp_mag_spec*batch[-2], hop_length=dataload.hop)
    sf.write(path_info_sample + 'original_' + str(idx1) + '.wav', orig_time_2, dataload.sr)

    batch = dataload[idx2]
    inp_mag_spec = np.exp(batch[3].cpu().data.numpy()) - 1
    orig_time_3 = lambda2 * core.istft(inp_mag_spec*batch[-2], hop_length=dataload.hop)
    sf.write(path_info_sample + 'mixer_' + str(idx2) + '.wav', orig_time_3, dataload.sr)
    return
    
    


batch = next(iter(test_loader))
sample_data, sample_target, file_names = batch[0], batch[1], batch[2]

### Code for handling classifier training/fine-tuning for different datasets or loading weights

if clf_train:
    if dataset == 'sust':
        progress = train_classifier(f, device, train_loader, test_loader, opt_f, 4)
        #opt_f = optim.Adam(itertools.chain(f.layer.paramesaters(), f.fc.parameters()), lr=0.0005)
        opt_f = optim.Adam(f.parameters(), lr=0.00005)
        progress3 = train_classifier(f, device, train_loader, test_loader, opt_f, 2)
        #opt_f = optim.Adam(itertools.chain(f.layer.parameters(), f.fc.parameters()), lr=0.0001)
        opt_f = optim.Adam(itertools.chain(f.layer.parameters(), f.fc.parameters()), lr=0.00002)
        progress_final = train_classifier(f, device, train_loader, test_loader, opt_f, 2)
        #opt_f = optim.Adam(itertools.chain(f.layer.parameters(), f.fc.parameters()), lr=0.00004)
        #progress5 = train_classifier(f, device, train_loader, test_loader, opt_f, 10)

    else:
        progress = train_classifier(f, device, train_loader, test_loader, opt_f, 10, dataset_name=dataset)
        opt_f = optim.Adam(itertools.chain(f.layer.parameters(), f.fc.parameters()), lr=0.0001)
        progress_final = train_classifier(f, device, train_loader, test_loader, opt_f, 4, dataset_name=dataset)

    best_f = progress_final[-1]
    model_dict = {}
    model_dict['f_state_dict'] = best_f.state_dict()
    if dataset == 'sust':
        torch.save(model_dict, 'output/' + dataset + '_output/' + 'classifier_full_FT_v2' + '.pt')
    elif dataset == 'esc50':
        torch.save(model_dict, 'output/' + dataset + '_output/' + 'classifier_fold' + str(num_FOLD) + '.pt')
    else:
        torch.save(model_dict, 'output/' + dataset + '_output/' + 'classifier.pt')
    print ('Saved a trained')
    #torch.save(model_dict, 'output/' + dataset + '_output/' + 'classifier_5Class' + '.pt')
else:
    if dataset == 'sust' and args[0] == 'train':
        checkpoint = torch.load('output/' + dataset + '_output/' + 'classifier_full_FT.pt', map_location='cpu')
        f.load_state_dict(checkpoint['f_state_dict'])
        f = f.eval().to(device)
    elif dataset == 'esc50'  and args[0] == 'train':
        checkpoint = torch.load('output/' + dataset + '_output/' + 'classifier_fold' + str(num_FOLD) + '.pt', map_location='cpu')
        f.load_state_dict(checkpoint['f_state_dict'])
        f = f.eval().to(device)


### Code for L2I training and evaluation

if args[0] == 'train':
    time1 = time.time()
    max_fid = 0
    for epoch in range(1, N_EPOCH + 1):
        test_info = test(f, H, h, W, device, test_loader, 1.0, 1.0, 1.0)
        if dataset == 'amnist':
            train_info = train(f, H, h, W, device, train_loader, optimizer, epoch, 0.5*epoch, 0.1*int(epoch > 1), 0.1*int(epoch > 2))
        elif dataset == 'msdb':
            if epoch == 27:
                optimizer = optim.Adam(itertools.chain(H.parameters(), h.parameters()), lr=0.0001)
            train_info = train(f, H, h, W, device, train_loader, optimizer, epoch, min(100.0, 50.0*epoch), 0.1*int(epoch > 1), 1.0*int(epoch > 0))
        elif dataset == 'esc50':
            #if epoch == 10:
            #    optimizer = optim.Adam(itertools.chain(W.parameters(), H.parameters(), h.parameters()), lr=0.0003)
            #if epoch == 19:
            #    optimizer = optim.Adam(itertools.chain(W.parameters(), H.parameters(), h.parameters()), lr=0.0001)
            if test_info[-2] > max_fid:
                print ('Found new best model', test_info[-2])
                max_fid = test_info[-2] + 0
                best_H, best_h, best_W = copy.deepcopy(H), copy.deepcopy(h), copy.deepcopy(W)
            train_info = train(f, H, h, W, device, train_loader, optimizer, epoch, 5.0, 0.5*int(epoch > 0) + 0.5*int(epoch > 19), 0.4*int(epoch > 0))
        elif dataset == 'esc10':
            if epoch == 19:
                optimizer = optim.Adam(itertools.chain(W.parameters(), H.parameters(), h.parameters()), lr=0.0001)
            train_info = train(f, H, h, W, device, train_loader, optimizer, epoch, 5.0, 0.4*int(epoch > 3), 2.0*int(epoch > 0))
        elif dataset == 'sust':
            #if epoch == 19:
            #    optimizer = optim.Adam(itertools.chain(W.parameters(), H.parameters(), h.parameters()), lr=0.0001)
            train_info = train(f, H, h, W, device, train_loader, optimizer, epoch, 5.0, 0.5*int(epoch > 0) + 0.5*int(epoch > 19), 0.4*int(epoch > 0)) # Original set of params
            #train_info = train(f, H, h, W, device, train_loader, optimizer, epoch, 5.0, 0.5*int(epoch > 0) + 0.0*int(epoch > 19), 0.04*int(epoch > 0))
        test_epoch_info.append(test_info)
        train_epoch_info.append(train_info)

    time2 = time.time()
    print ('Time taken for training', time2 - time1)

    test_info = test(f, H, h, W, device, test_loader, 1.0, 1.0, 1.0)
    test_epoch_info.append(test_info)
    train_epoch_info = np.array(train_epoch_info)
    test_epoch_info = np.array(test_epoch_info)
    #weight_str = 'lmbd_acc=0.08*int(epoch < 6) + 0.02, lmbd_rec=0.5*epoch, lmbd_expl=0.1*int(epoch>1), lmbd_spa=0.1*int(epoch > 2)' # For AMNIST, change accordingly to your dataset
    #weight_str = '5.0, 0.5*int(epoch > 0) + 0.5*int(epoch > 19), 0.2*int(epoch > 0)'
    weight_str = '5.0, 0.5*int(epoch > 0) + 0.5*int(epoch > 19), 0.4*int(epoch > 0)'

    #model_name = 'N4_lr0.0005_E' + str(N_EPOCH) + '_Wl2'
    if dataset == 'sust':
        model_name = 'full_model'
        model_dict = save_model(weight_str, model_name)
    elif dataset == 'esc50':
        a, b, c = analyze(f, H, h, W, device, test_loader, n_classes=50)
        model_name = 'full_model_Fold' + str(num_FOLD) + '_K' + str(n_components)
        model_dict = save_model(weight_str, model_name, save_best=True)
    print ('Saved model', model_name)
    #print ('Model Name:', model_name)


elif args[0] == 'test':
    model_name = args[4][:-3]
    try:
        os.mkdir('output/' + dataset + '_output/explanation_' + str(model_name))	
        os.mkdir('output/' + dataset + '_output/explanation_' + str(model_name) + '/info_sample')
        os.mkdir('output/' + dataset + '_output/explanation_' + str(model_name) + '/info_global')
        if dataset == 'esc50':
            os.mkdir('output/' + dataset + '_output/explanation_' + str(model_name) + '/overlap')
    except:
        print ('Writing files in an old folder. May overwrite some files or create additional unwanted ones')

    checkpoint1 = torch.load('output/' + dataset + '_output/' + model_name + '.pt', map_location='cpu')
     
    f.load_state_dict(checkpoint1['f_state_dict'])
    H.load_state_dict(checkpoint1['H_state_dict'])
    h.load_state_dict(checkpoint1['h_state_dict'])
    W.load_state_dict(checkpoint1['W_state_dict'])
    f, H, h, W = f.eval(), H.eval(), h.eval(), W.eval()

    # Glabal analysis and saving in info_global folder
    plt.imshow(W.return_W(), origin='lower')
    plt.colorbar()
    plt.savefig('output/' + dataset + '_output/explanation_' + str(model_name) + '/info_global/' + 'W', bbox_inches='tight', pad_inches = 0.03)
    plt.close()

    plt.imshow(h.fc1.weight.cpu().data.numpy())
    plt.savefig('output/' + dataset + '_output/explanation_' + str(model_name) + '/info_global/' + 'Theta_weights', bbox_inches='tight', pad_inches = 0.03)
    plt.close()


    ff = []   

    # EVALUATION Rooster+ Interpretation generations on samples
    #test_info = test(f, H, h, W, device, test_loader)
    if dataset == 'esc50':
        # To run these experiments ensure add_noise=False where esc50_test1 is declared (~ line 64).
        print ('Top-k fidelity correct out of 400 test samples')
        a, b, c = analyze(f, H, h, W, device, test_loader, n_classes=50)
        print ('Top-1 fidelity total:', int(np.sum(np.diag(b))))
        
        overlap_exp(esc50_test1, 300, 157, 0.6, f, H, W, h, model_name)
        overlap_exp(esc50_test1, 157, 300, 0.15, f, H, W, h, model_name)
        overlap_exp(esc50_test1, 199, 178, 1.0, f, H, W, h, model_name)
        overlap_exp(esc50_test1, 157, 229, 0.09, f, H, W, h, model_name)

        
        # To run noise experiments. First make add_noise=True at the start where esc50_test1 is declared (~ line 64). This will add noise to test samples. Or uncomment the line below
        #esc50_test1 = dl.ESC50(mode='test', num_FOLD=num_FOLD, root_dir=esc50_dir + '/ESC50', add_noise=True)
        #for i in [178, 199, 229, 237, 300, 323]:
        #    print ('Sample index', i)
        #    explanation_multiclass(esc50_test1, i, f, H, W, h, model_name)
        
        '''
        print ('\nComputing faithfulness results\n')
        for i in range(400):        
            ff.append(faithfulness_eval_multiclass(esc50_test1, i, f, H, W, h, save_audio=False, thresh=0.3, select_random=False))
            # [Absolute logit drop, Absolute probability drop, Relative probability drop, predicted class, top-1 agreement, top-3 agreement, top-5 agreement, num_relevant_components]
        ff = np.array(ff)
        print ('absolute logit drop, num_components used', np.median(ff[:, 0]), ff[:, 7].mean())
        print ('absolute prob drop', np.median(ff[:, 1]))
        '''

        
    elif dataset == 'sust':
        #class_thresh = compute_multilabel_threshold(f, H, h, train_loader)
        class_thresh = np.array([0.6, 0.6, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5]) # This array was determined using code in previous line
        #test_info = test(f, H, h, W, device, test_loader, 1.0, 1.0, 1.0)
        #all_pred, all_target = np.concatenate(test_info[-2]), np.concatenate(test_info[-1])
        #print ('Computed class-wise thresholds for classifier', class_thresh)
        #print ('Ready to analyze')
        #indices = subjective_samp_select(all_pred, all_target, 6)
        #print ('Selected indices', indices)
        imp_cl_full = []

        # UNCOMMENT THE FOLLOWING PART FOR FAITHFULNESS RESULTS ON SONYC-UST or to generate interpretations on full test data
        
        for i in range(663):
        #for i in indices:
            print ('Sample index', i)
            #imp_cl_full += explanation_multilabel(sust_test, i, f, H, W, h, model_name, save_files=False)
            ff += faithfulness_eval(sust_test, i, f, H, W, h, save_audio=False, thresh=0.1, select_random=True)
        ff = np.array(ff)
        stats = []
        for cl in range(8):
            idx = np.where(ff[:, 3] == cl)[0]
            print ('\nClass', cl, ' Number of samples considered', idx.shape[0])
            print (np.median(ff[idx, 0]), ff[idx, 4].mean())
            print (np.median(ff[idx, 1]), ff[idx, 4].mean())
            print (np.median(ff[idx, 2]), ff[idx, 4].mean())
            #stats.append( [ff[idx, 2].mean(), ff[idx, 2].std(), np.median(ff[idx, 2]), ff[idx, 0].mean(), ff[idx, 0].std(), np.median(ff[idx, 0]), ff[idx, 1].mean(), ff[idx, 1].std(), np.median(ff[idx, 1]), ff[idx, 4].mean(), idx.shape[0]] )
        #stats = np.array(stats)
        
    





