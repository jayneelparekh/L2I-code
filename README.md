# Listen to Interpret (L2I)

This repository constains code for the paper ["Listen to Interpret: Post-hoc Interpretability for Audio Networks with NMF"](https://arxiv.org/abs/2202.11479) by *Jayneel Parekh, Sanjeel Parekh, Pavlo Mozharovskyi, Florence d'Alché-Buc, Gaël Richard* (Accepted at NeurIPS 2022)

Link for the [project webpage](https://jayneelparekh.github.io/listen2interpret/). Contains audio samples and interpretations for all the experiments.

## Setup
Setup a new conda environment with the ```env_audio.yml``` file.

```sh
   conda env create -f env_audio.yml
```

NOTE: The default pytorch installed would be cpu usage only. IF YOU WANT to TRAIN and USE GPU, please also run this command (or an equivalent command according to your CUDA compatibility) after creating the environment.

```
   conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=9.2 -c pytorch
```

We perform experiments on two datasets: ESC-50 AND SONYC-UST. You will need to download and extract the datasets. Instructions for downlading the two are given below.

(1) ESC-50: Download link on their [github page](https://github.com/karolpiczak/ESC-50). Extract the downloaded zip to the path 'L2I-code/datasets'. Please ensure the name of the extracted folder is: 'ESC50'. The final path of dataset should look like 'L2I-code/datasets/ESC50'

(2) SONYC-UST: Download link and instructions on their [Zenodo page](https://zenodo.org/record/3966543#.Yo6v1n9Bw5k). The dataset is offered under the CC BY 4.0 license by its authors. Ensure the name of dataset folder is 'SONYC_UST'. The final path of dataset should look like 'L2I-code/datasets/SONYC_UST'

If the datasets are downloaded and extracted in a different directory, please ensure the paths defined in L61 -- L75 of audint_posthoc.py match their location and names.


### Network architecture and trained models

The trained models on ESC-50 fold-1, SONYC-UST are available in the 'output/esc50_output/', 'output/sust_output/' directories respectively along with the trained dictionaries on them.

The backbone network that we fine-tune and perform post-hoc interpretation on is based on the work of [Kumar et al.](https://github.com/anuragkr90/weak_feature_extractor). We have not uploaded the pre-trained weights of this network. We have directly provided our fine-tuned weights on our datasets. 


## Usage

```audio_posthoc.py``` is our main file

    1. Command: python audint_posthoc.py [mode test or train] [dataset name] [use-gpu True or False] [fine-tune-classifier True or False] [if mode is test, enter model name here as 4th argument]

    2. Dataset options: esc50, sust.

    3. Example commands: 
       python audint_posthoc.py test esc50 False False try10_AttExpv2.pt 
       python audint_posthoc.py test sust False False try14_AttExp_v2.pt (if you don't have GPU).
       
These commands should generate the fidelity metrics by default and in case of ESC-50, also generate interpretations from overlap experiment. You can run other experiments (faithfulness or noise experiments) by uncommenting various parts from L1283 -- L1336 of ```audio_posthoc.py```. 
   
NOTE 1: The functionality for fine-tuning the classifier is not setup properly. Contact me separately if you wish to add it. 
    
    
     4. Command for training:
     python audint_posthoc.py train esc50 True False
     
     GPU-usage by default is preferred for training. Refer to Setup above for adding GPU-support.
    
    
    
## License

Add license







