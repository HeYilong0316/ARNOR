# ARNOR
A  implementation relation extraction model ARNOR (Tensorflow) 
Here is the original [paper](https://www.aclweb.org/anthology/P19-1135) and [repository](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2019-ARNOR) 

## Requirement:
python 3.7

tensorflow 1.13
## Data
Please refer to [here](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2019-ARNOR)

Download the data and put them in the datas folder.
## How to run the code
1. run the code to train the model and then config_file and maps.npy will be generated to store model parameters
>python main.py --train=True --clean=True
2. test
>python main.py --restore=True
3. major parameter

`--clean=True` means remove maps.npy and config_file before training, 

`--user_small=True` means use 100 datas to debug

## Structure
\│  main.py
\│  tree.txt
\│  
\├─datas
\│      dev.json
\│      test.json
\│      train.json
\│      
└─model
        data_loader.py
        model.py
        util.py
    
## Main results
continue...
        
