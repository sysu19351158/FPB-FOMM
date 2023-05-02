# Facial Prior Guided Micro-Expression Generation
Implementation of **FOMM with AWP** and **MRAA with AWP**.

## 0. Table of Contents

* [0. Table of Contents](#0-table-of-contents)

* [1. Authors & Maintainers](#1-authors---maintainers)

* [2. Change Log](#2-change-log)

* [3. Visualizations](#3-visualizations)

* [4. Run the Code](#4-run-the-code)

* [5. License](#5-license)

  

## 1. Authors & Maintainers

- [Yi Zhang|@zylye123](https://github.com/zylye123)
- [Xinhua Xu|@sysu19351158](https://github.com/sysu19351158)
- [Youjun Zhao|@zhaoyjoy](https://github.com/zhaoyjoy)
- [Yuhang Wen|@Necolizer](https://github.com/Necolizer)
- [Zixuan Tang|@sysu19351118](https://github.com/sysu19351118)

## 2. Change Log

- [2022/02/18] Upload code.
- [2023/05/02] + Case Visualizations.

## 3. Visualizations

| CASE  |                            Driving                            |                             FOMM                             |                             FOMM w/ EWP                             |                            FOMM w/ AWP                             |                            MRAA                             |                            MRAA w/ AWP                             |
| :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  1   |    <img src="./result-gifs/Driving/022_3_3.gif" style="zoom: 50%;" />    | <img src="./result-gifs/FOMM/022_3_3-normalized_asianFemale.gif" style="zoom: 50%;" />  | <img src="./result-gifs/FOMM_with_EWP/022_3_3-normalized_asianFemale.gif" style="zoom: 50%;" />  | <img src="./result-gifs/FOMM_with_AWP/022_3_3-normalized_asianFemale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/MRAA/022_3_3-normalized_asianFemale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/MRAA_with_AWP/022_3_3-normalized_asianFemale.gif" style="zoom: 50%;" /> |
|  2   |   <img src="./result-gifs/Driving/s3_po_05.gif" style="zoom: 50%;" />    | <img src="./result-gifs/FOMM/s3_po_05-normalized_asianFemale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM_with_EWP/s3_po_05-normalized_asianFemale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM_with_AWP/s3_po_05-normalized_asianFemale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/MRAA/s3_po_05-normalized_asianFemale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/MRAA_with_AWP/s3_po_05-normalized_asianFemale.gif" style="zoom: 50%;" /> |
|  3   | <img src="./result-gifs/Driving/sub19_EP01_01f.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM/sub19_EP01_01f-normalized_asianFemale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM_with_EWP/sub19_EP01_01f-normalized_asianFemale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM_with_AWP/sub19_EP01_01f-normalized_asianFemale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/MRAA/sub19_EP01_01f-normalized_asianFemale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/MRAA_with_AWP/sub19_EP01_01f-normalized_asianFemale.gif" style="zoom: 50%;" /> |
|  4   | <img src="./result-gifs/Driving/018_3_1.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM/018_3_1-normalized_westernMale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM_with_EWP/018_3_1-normalized_westernMale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM_with_AWP/018_3_1-normalized_westernMale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/MRAA/018_3_1-normalized_westernMale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/MRAA_with_AWP/018_3_1-normalized_westernMale.gif" style="zoom: 50%;" /> |
|  5   | <img src="./result-gifs/Driving/sub17_EP01_13.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM/sub17_EP01_13-normalized_westernMale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM_with_EWP/sub17_EP01_13-normalized_westernMale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/FOMM_with_AWP/sub17_EP01_13-normalized_westernMale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/MRAA/sub17_EP01_13-normalized_westernMale.gif" style="zoom: 50%;" /> | <img src="./result-gifs/MRAA_with_AWP/sub17_EP01_13-normalized_westernMale.gif" style="zoom: 50%;" /> |

- To facilitate micro-expression observations and reduce the storage space of this repository, all Gifs have been slowed down and compressed.
- Case 1-3: Positive. Case 4: Negative. Case 5: Surprise.


## 4. Run the Code

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

## 5. License

[MIT](https://github.com/sysu19351158/FPB-FOMM/blob/main/LICENSE)
