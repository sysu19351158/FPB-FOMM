MRAA 
train
设置数据集路径
CUDA_VISIBLE_DEVICES=1,2 python run.py --config config/my_dataset_train.yaml --device_ids 0,1

test
设置数据集路径
设置测试集csv
CUDA_VISIBLE_DEVICES=0,1 python run.py --mode animate --config config/my_dataset_test.yaml --checkpoint /home/amax/Titan_Five/WYH/Facial-Prior-Based-FOMM/log/my_dataset_train_29_08_21_06.15.04/00000099-checkpoint.pth.tar --device_ids 0,1












