This repository contains code for our NeurIPS 2022 Paper: Listen to Interpret: Post-hoc Interpretability for Audio Networks with NMF

There are TWO parts to this documentation:
1. Data part (How to acquire the datasets and which directory to extract them in) 
2. Code part for experiments (Brief description for all functions available for you)


DATA PART Documentation:

    We perform experiments on two datasets: ESC-50 AND SONYC-UST. YOu need to first download and extract the dataset if you want to run experiment on it. We give instructions for downlading the two below along with references to the original public pages about them

    (1) ESC-50: Download link on their github page: https://github.com/karolpiczak/ESC-50. Extract the downloaded zip to the path 'working_directory/datasets'
        Additional info from the github page: "The dataset is available under the terms of the Creative Commons Attribution Non-Commercial license."
        After downloading the .zip, you should extract it into the 'datasets' folder in the current working directory. Please ensure the name of the extracted folder is: 'ESC50'.

    (2) SONYC-UST: Download link and instructions on the Zenodo page: https://zenodo.org/record/3966543#.Yo6v1n9Bw5k 
        Additional info from the Zenodo page: "The SONYC-UST dataset is offered free of charge under the terms of the Creative Commons Attribution 4.0 International (CC BY 4.0) license:"

    If the datasets are downloaded and extracted in the expected directory, the paths defined in L61 -- L74 of audint_posthoc.py will match their location and names


CODE PART Documentation:


(A) SETUP THE CONDA ENVIRONMENT. 

    The 'env_audio.yml' file is in the main directory. Try creating environment using this file. This file contains the minimal amount of packages (almost) needed for our code. If conda throws an error (which it shouldn't but unfortunately can't be ascertained for all OS, machine configs), you should try and manually install each package given in 'env_minimal.yml' according to your machine.

    Command to create environment: conda env create -f env_audio.yml

    Important NOTE: The default pytorch installed would be cpu usage only. IF YOU WANT to TRAIN and USE GPU, please also run this command (or an equivalent command according to your CUDA compatibility): conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=9.2 -c pytorch
  

(B) Available trained models

    The trained models on ESC-50 fold-1, SONYC-UST are available in the 'output/esc50_output/', 'output/sust_output/' directories respectively. Other models not shared due to space constraints



(C) Usage of the code (IN GENERAL INCL. TRAINING PHASE)

    audio_posthoc.py is our main file

    1. Command: python audint_posthoc.py [mode test or train] [dataset name] [use-gpu True or False] [fine-tune-classifier True or False] [if mode is test, enter model name here as 4th argument]

    2. Dataset options: esc50, sust.

    3. Example commands: 
       "python audint_posthoc.py test esc50 False False try10_AttExpv2.pt" 
       "python audint_posthoc.py test sust False False try14_AttExp_v2.pt" (if you don't have GPU). It is preferrable if you use gpu during training.
       These commands should generate the fidelity metrics by default and in case of ESC-50 generate interpretations from overlap experiment. You can run other experiments by uncommenting various parts from L1283 -- L1336 of the main file accordingly. 

    4. functions explanation_multiclass(...), explanation_multilabel(...) can help you generate interpretations for any sample of a dataset, spcified by its index. Their usage is covered in L1283 -- L1336.    
   
    ADDITIONAL NOTE: You can train and test (also generates interpretations) your models from the above command lines. Randomness in training will influence to some extent. However training times are expected to be much higher because (1) GPU-usage by default is assumed False, (2) the dataloader in the code is forced to use only one thread (num_workers=1).  Both steps taken to reduce possibilities of errors. Fix for (1) was already discussed earlier in environment setup. For (2) You will need to change the num_workers values passed in dataset declaration (L61 -- L74). Please be aware that for some systems num_workers > 1 might cause scheduling errors from PyTorch. 




