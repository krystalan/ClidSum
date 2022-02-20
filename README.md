# ClidSum
This repository contains the data, codes and model checkpoints for our paper ["ClidSum: A Benchmark Dataset for Cross-Lingual Dialogue Summarization"](https://arxiv.org/abs/2202.05599).   

In this work, we introduce cross-lingual dialogue summarization task and present `ClidSum` benchmark dataset together with `mDialBART` pre-trained language model.
- `ClidSum` contains `XSAMSum`, `XMediaSum40k` and `MediaSum424k` three subsets.    
- `mDialBART` extends mBART-50 via the second pre-training stage.    

## ClidSum Benchmark Dataset  
<ins>**Please restrict your usage of this dataset to research purpose only.**</ins>

You can obtain `ClidSum` from the share links: [XSAMSum](https://drive.google.com/file/d/1zmKKF5xX1RJCk0x_cyKgrVzkQf5B5awy/view?usp=sharing), [XMediaSum40k](https://drive.google.com/file/d/1ETwdHFKEp-DZYLejHvoMp3CXn-kTsmoB/view?usp=sharing).   
Note that `MediaSum424k` is the subset of MediaSum. Please refer to the [original work](https://github.com/zcgzcgzcg1/MediaSum/).   

The format of obtained JSON files is as follows:
```
[
    {
        "dialogue": "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it.\nAmanda: Ask Larry\nAmanda: He called her last time we were at the park together\nHannah: I don't know him well\nHannah: <file_gif>\nAmanda: Don't be shy, he's very nice\nHannah: If you say so..\nHannah: I'd rather you texted him\nAmanda: Just text him 🙂\nHannah: Urgh.. Alright\nHannah: Bye\nAmanda: Bye bye",
        "summary": "Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry.",
        "summary_de": "hannah braucht bettys nummer, aber amanda hat sie nicht. sie muss larry kontaktieren.",
        "summary_zh": "汉娜需要贝蒂的电话号码，但阿曼达没有。她得联系拉里。"
    },
    {
        "dialogue": "Eric: MACHINE!\r\nRob: That's so gr8!\r\nEric: I know! And shows how Americans see Russian ;)\r\nRob: And it's really funny!\r\nEric: I know! I especially like the train part!\r\nRob: Hahaha! No one talks to the machine like that!\r\nEric: Is this his only stand-up?\r\nRob: Idk. I'll check.\r\nEric: Sure.\r\nRob: Turns out no! There are some of his stand-ups on youtube.\r\nEric: Gr8! I'll watch them now!\r\nRob: Me too!\r\nEric: MACHINE!\r\nRob: MACHINE!\r\nEric: TTYL?\r\nRob: Sure :)",
        "summary": "Eric and Rob are going to watch a stand-up on youtube.",
        "summary_de": "eric und rob werden sich ein stand-up auf youtube ansehen.",
        "summary_zh": "埃里克和罗伯要在youtube上看一场单口相声。"
    },
    ...
]
```
`summary` represents the original English summary of the corresponding `dialogue`. `summary_de` and `summary_zh` indicate the human-translated German and Chinese summaries, respectively.

In addition, as described in our paper, `XMediaSum40k` is constructed based on 40k samples randomly selected from the MediaSum corpus (totally 463k samples). To know which samples are selected, you can find original `ID` of each sample from [train_val_test_split.40k.json](https://drive.google.com/file/d/1gi5Q_P-ANxULualTtZITTJ8YDu6jNQAQ/view?usp=sharing) file.  

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
