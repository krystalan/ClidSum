# ClidSum
This repository contains the data, codes and model checkpoints for our paper ["ClidSum: A Benchmark Dataset for Cross-Lingual Dialogue Summarization"](https://arxiv.org/abs/2202.05599).   

In this work, we introduce cross-lingual dialogue summarization task and present `ClidSum` benchmark dataset together with `mDialBART` pre-trained language model.
- `ClidSum` contains `XSAMSum`, `XMediaSum40k` and `MediaSum424k` three subsets.    
- `mDialBART` extends mBART-50 via the second pre-training stage.    

## ClidSum Benchmark Dataset  
<ins>**Please restrict your usage of this dataset to research purpose only.**</ins>

You can download our ClidSum from the share links: [XSAMSum](https://drive.google.com/file/d/1zmKKF5xX1RJCk0x_cyKgrVzkQf5B5awy/view?usp=sharing), [XMediaSum40k](https://drive.google.com/file/d/1ETwdHFKEp-DZYLejHvoMp3CXn-kTsmoB/view?usp=sharing) (note that `MediaSum424k` is the subset of MediaSum. You can obtain it from [original work](https://github.com/zcgzcgzcg1/MediaSum/)). 

XMediaSum40k is constructed based on 40k samples randomly selected from the MediaSum corpus (totally 463k). To know which samples are selected, you can find original `ID` of each sample via [train_val_test_split.40k.json](https://drive.google.com/file/d/1gi5Q_P-ANxULualTtZITTJ8YDu6jNQAQ/view?usp=sharing) file.  

License: [TBD]

## mDialBART

The checkpoints of `mDialBART` will be released soon.

## Usage
[TODO]

## Citation
If you find this work is useful or use the data in your work, please consider cite our paper:
```
@article{Wang2022ClidSumAB,
  title={ClidSum: A Benchmark Dataset for Cross-Lingual Dialogue Summarization},
  author={Jiaan Wang and Fandong Meng and Ziyao Lu and Duo Zheng and Zhixu Li and Jianfeng Qu and Jie Zhou},
  year={2022},
  eprint={2202.05599},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
