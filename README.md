# ARNOR
A  implementation relation extraction model ARNOR (Tensorflow) 
Here is the original [paper](https://www.aclweb.org/anthology/P19-1135) and [repository](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2019-ARNOR) 

Beta version. Can not get the result of the data version2.0 described by his github.
## Requirement:
python 3.7
tensorflow 1.13
## Data
Please refer to [here](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2019-ARNOR)
## How to run the code
1. Download the data and put them in the datas folder.
2. run the code t0 train the model and then config_file and maps.npy will be generated to store model parameters

==--clean=True== means remove maps.npy and config_file before training, 

==--user_small=True== means use 100 datas to debug
>python main.py --train=True --clean=True
3. test
>python main.py --restore=True

## Structure
│  main.py
│  tree.txt
│  
├─datas
│      dev.json
│      test.json
│      train.json
│      
└─model
        data_loader.py
        model.py
        util.py
    
## Main results
continue...
        
