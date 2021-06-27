Dataset

data文件夹对应Criminal-S数据集，其中attributes文件和words.vec文件是通用的，train为训练集，valid为验证集，test为测试集

data_20w文件夹对应Criminal-M数据集

data_38w文件夹对应Criminal-L数据集

Run

    cd charge
    
    python2 train.py

PS：数据集的位置需要替换为自己的位置

Dependencies

- Tensorflow == 0.12
- Numpy == 1.16.0
- Python == 2.7
