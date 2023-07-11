# Exploring the User Guidance for More Accurate Building Segmentation

This repo is the official PyTorch implementation of **Exploring the User Guidance for More Accurate Building Segmentation from High-Resolution Remote Sensing Images**. Since this work is still under review, we only provide the test code and related models, and the full code will be released soon.

##  1. Requirements

   - Hardware: 4-8 GPUs (better with >=11G GPU memory)
   - Software: PyTorch>=1.0.0, Python3, [tensorboardX](https://github.com/lanpa/tensorboardX),  and so on.
   - to install tensorboardX, you could do as follows:

     ```
     pip install tensorboardX
     pip install tensorboard
     ```
- install python packages: `pip install -r requirements.txt`

## 2. Dataset

### Segmentation

- SpaceNet (Las Vegas): [processed test set](https://drive.google.com/file/d/1trVOeaDWKUey6c5Rk0ZTHZHES54w8J3A/view?usp=share_link)
- Inria-building dataset: [dataset](https://project.inria.fr/aerialimagelabeling/download/), [dataset split file](https://drive.google.com/file/d/1TnPz5G7hpavsqbnvBO7njgqLZJpyWTji/view?usp=share_link)

<details>
  <summary>Segmentation data structure</summary>
  <pre>
  DATA_ROOT
  ├── vegas
  │   ├── vegas_trainval_img_224_jpg
  │   ├── vegas_trainval_label_224_png_01
  │   ├── vegas_test_img_224_jpg
  │   ├── vegas_test_label_224_png_01
  ├── Inria_dataset
  │   ├── inria
  │   │   ├── img
  │   │   ├── inria_split
  │   │   ├── mask_all
  │   │   ├── mask_one
  │   │   ├── polygons.csv
  </pre>
</details>



## 3. Model Zoo

### Segmentation

train: SpaceNet (Las Vegas)

|      Annotations      |                            Models                            | IoU  | BF-score | B-IoU | Notes |
| :-------------------: | :----------------------------------------------------------: | :--: | :------: | :---: | ----- |
|     Bounding box      | [gdrive](https://drive.google.com/file/d/1c5jU82xNymisqAs6ZAqJwsR628qYDKSq/view?usp=sharing) | 93.7 |   80.7   | 39.9  |       |
|    Extreme points     | [gdrive](https://drive.google.com/file/d/1C04utTNe17hYk9wvsC0az4IB9SDZxZmn/view?usp=sharing) | 95.3 |   87.3   | 48.5  |       |
| Inside-outside points | [gdrive](https://drive.google.com/file/d/10nSvDqUo8kwhsUtUdxIsVqtpUyoFYSY_/view?usp=sharing) | 95.4 |   86.3   | 49.4  |       |

train: Inria-building dataset

|      Annotations      |                            Models                            | IoU  | WCov | B-Fscore | Dice | Notes |
| :-------------------: | :----------------------------------------------------------: | :--: | :--: | :------: | :--: | ----- |
|     Bounding box      | [gdrive](https://drive.google.com/file/d/1we5bI-TdyVGsh-FIRhvT4rFli0xg3Hyi/view?usp=sharing) | 92.1 | 92.2 |   84.3   | 95.9 |       |
|    Extreme points     | [gdrive](https://drive.google.com/file/d/1we5bI-TdyVGsh-FIRhvT4rFli0xg3Hyi/view?usp=sharing) | 93.1 | 93.1 |   86.9   | 96.4 |       |
| Inside-outside points | [gdrive](https://drive.google.com/file/d/1ttV0-rSssoekS02EyzcmtXGhM-o-BpGp/view?usp=share_link) | 92.8 | 92.9 |   86.0   | 96.3 |       |

## 4. Inference

Run on one GPU to evaluate the model, the examples are as follow:

Run the inference job:

```shell
sh {tool_name}/evaluate_banet.sh {dataset} {task_name}
```

An example of testing our method on Vegas:

```shell
sh seg_tool/evaluate_banet.sh vegas banet_dextr_res101
```

Note:

1. To evaluate segmentation methods with the metrics of [CVNet](https://github.com/xzq-njust/CVNet) (IoU, WCov, BF-score and Dice),  you shoud use `seg_tool/evaluate_cvnet.sh`

## 5. License

This project is under the MIT license.

## 6. Citation

```
Coming soon
```
