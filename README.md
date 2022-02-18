# Facial Prior Guided Micro-Expression Generation
Implementation of FOMM_with_AWP and MRAA_with_AWP

## 0. Table of Contents

* [0. Table of Contents](#0-table-of-contents)

* [1. Authors & Maintainers](#1-authors---maintainers)

* [2. Change Log](#2-change-log)

* [3. Run the Code](#3-run-the-code)

* [4. License](#4-license)

  

## 1. Authors & Maintainers

- [Yi Zhang|@zylye123](https://github.com/zylye123)
- [Xinhua Xu|@sysu19351158](https://github.com/sysu19351158)
- [Youjun Zhao|@zhaoyjoy](https://github.com/zhaoyjoy)
- [Yuhang Wen|@Necolizer](https://github.com/Necolizer)

## 2. Change Log

- [2022/02/18] Upload code.

## 3. Run the Code

1. Operations of FOMM_with_AWP and MRAA_with_AWP are the same.

2. Prepare your dataset. Recommend CASME2, SAMM, SMIC-HS

   Divide into `your_dataset/train` and `your_dataset/test`

   Create or modify `yaml` format file `your_dataset_train.yaml` in `./config`

3. Train

   ```shell
   python run.py --config config/your_dataset_train.yaml
   ```

   Log, parameters and checkpoints would be saved in `./log`

4. Test

   Create or modify `csv` format file `your_dataset_test.csv` in `./data`
   
   ```shell
   python run.py --config config/my_dataset_test.yaml --mode animate --checkpoint path/to/checkpoint
   ```
   
   Generated videos would be saved in `path/to/checkpoint/animation`

## 4. License

[MIT](https://github.com/Necolizer/Facial-Prior-Based-FOMM/blob/main/LICENSE)
