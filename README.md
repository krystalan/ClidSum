## ClidSum: A Benchmark Dataset for Cross-Lingual Dialogue Summarization
This repository contains the data, codes and model checkpoints for our paper ["ClidSum: A Benchmark Dataset for Cross-Lingual Dialogue Summarization"](https://arxiv.org/abs/2202.05599).   

###  updates
- *2022.10.07*: Our paper is accepted to EMNLP 2022 (main conference) as a long paper.
- *2022.02.25*: We released our training [code](#5-finetune-mdialbart).
- *2022.02.23*: We released our [model checkpoints](#3-model-list) and [model outputs](#model-outputs).
- *2022.02.22*: We released the [ClidSum benchmark dataset](#2-clidsum-benchmark-dataset).
- *2022.02.14*: We released [our paper](https://arxiv.org/abs/2202.05599). Check it out!

## Quick Links
- [1. Overview](#1-overview)
- [2. ClidSum Benchmark Dataset](#2-clidsum-benchmark-dataset)
- [3. Model List](#3-model-list)
- [4. Use mDialBART with Huggingface](#4-use-mdialbart-with-huggingface)
- [5. Finetune mDialBART](#5-finetune-mdialbart)
    - [Requirements](#requirements)
    - [Fine-tuning](#finetuning)
    - [Model Outputs](#model-outputs)
    - [Evaluation](#evaluation)
- [6. Recommendation](#6-recommendation)
- [7. Citation and Contact](#7-citation-and-contact)

## 1. Overview

In this work, we introduce cross-lingual dialogue summarization task and present `ClidSum` benchmark dataset together with `mDialBART` pre-trained language model.
- `ClidSum` contains `XSAMSum`, `XMediaSum40k` and `MediaSum424k` three subsets.    
- `mDialBART` extends mBART-50 via the second pre-training stage. The following figure is an illustration of our `mDialBART`.

![](figure/model.png)

## 2. ClidSum Benchmark Dataset  
<ins>**Please restrict your usage of this dataset to research purpose only.**</ins>  

You can obtain `XMediaSum40k` from the [share link](https://drive.google.com/file/d/1ETwdHFKEp-DZYLejHvoMp3CXn-kTsmoB/view?usp=sharing).
For `MediaSum424k`, please refer to the [MediaSum Repository](https://github.com/zcgzcgzcg1/MediaSum/) since `MediaSum424k` is the subset of MediaSum.

For `XSAMSum`, please send an application email to jawang1[at]stu.suda.edu.cn to obtain it. Note that, we cannot directly release the share link of `XSAMSum` due to the [CC BY-NC-ND 4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/) of original `SAMSum` dataset. 

The following table shows the statistics of our `ClidSum`.  

![](figure/dataset.png)
  

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


License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 3. Model List
Our released models are listed as following. You can import these models by using [HuggingFace's Transformers](https://github.com/huggingface/transformers).   

| Model | Checkpoint |
| :--: | :--: |
| mDialBART (En-De) | [Krystalan/mdialbart_de](https://huggingface.co/Krystalan/mdialbart_de) |
| mDialBART (En-Zh) | [Krystalan/mdialbart_zh](https://huggingface.co/Krystalan/mdialbart_zh) |


## 4. Use mDialBART with Huggingface
You can easily import our models with HuggingFace's `transformers`:

```python
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBart50TokenizerFast

# The tokenizer used in mDialBART is based on the mBART50's tokenizer, and we only add a special token [SUM] to indicate the summarization task during the second pre-training stage.
tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', src_lang='en_XX', tgt_lang='de_DE')
tokenizer.add_tokens(['[summarize]']) 

# Import our models. The package will take care of downloading the models automatically
model = MBartForConditionalGeneration.from_pretrained('Krystalan/mdialbart_de')
```

## 5. Finetune mDialBART
In the following section, we describe how to finetune a mdialbart model by using our code.

### Requirements
Please run the following script to install the dependencies:
```
pip install -r requirements.txt
```  

### Code Structure Overview
    .
    ├── run_XMediaSum40k.py
    │
    ├── data
    │   └── XSAMSum
    │   │       ├── train.json
    │   │       ├── val.json
    │   │       └── test.json
    │   └── XMediaSum40k
    │           ├── train.json
    │           ├── val.json
    │           └── test.json
    └── model_output

### Finetuning


```bash
# Finetuning mDialBART on XMediaSum40k (En-De):
python -u run_XMediaSum40k.py \
    --model_path Krystalan/mdialbart_de \
    --data_root data/XMediaSum40k \
    --tgt_lang de_DE \
    --save_prefix mdialbart_de \
    --fp32

# Finetuning mDialBART on XMediaSum40k (En-Zh):
python -u run_XMediaSum40k.py \
    --model_path Krystalan/mdialbart_zh \
    --data_root data/XMediaSum40k \
    --tgt_lang zh_CN \
    --save_prefix mdialbart_zh \
    --fp32

```

Moreover, if you want to fine-tune `mBART-50` by using our code, you should change the `model_path` to mbart-50:  
```bash
# Finetuning mBART-50 on XMediaSum40k (En-De):
python -u run_XMediaSum40k.py \
    --model_path facebook/mbart-large-50-many-to-many-mmt \
    --data_root data/XMediaSum40k \
    --tgt_lang de_DE \
    --save_prefix mbart50_de \
    --fp32

# Finetuning mBART50 on XMediaSum40k (En-Zh):
python -u run_XMediaSum40k.py \
    --model_path facebook/mbart-large-50-many-to-many-mmt \
    --data_root data/XMediaSum40k \
    --tgt_lang zh_CN \
    --save_prefix mbart50_zh \
    --fp32
```


### Model Outputs
Output summaries are available at [outputs](https://github.com/krystalan/ClidSum/tree/main/outputs) directory.   
- `mdialbart_de.txt`: the output German summaries of fine-tuning `mDialBART` on `XMediaSum40k`.
- `mdialbart_zh.txt`: the output Chinese summaries of fine-tuning `mDialBART` on `XMediaSum40k`.
- `mdialbart_de_da.txt`: the output German summaries of fine-tuning `mDialBART` on `XMediaSum40k` with the help of pseudo samples (DA).
- `mdialbart_zh_da.txt`: the output Chinese summaries of fine-tuning `mDialBART` on `XMediaSum40k` with the help of pseudo samples (DA).

### Evaluation
For ROUGE Scores, we utilize the [Multilingual ROUGE Scoring](https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring) toolkit. The evaluation command like this:
```bash
python rouge.py \
    --rouge_types=rouge1,rouge2,rougeL \
    --target_filepattern=gold.txt \
    --prediction_filepattern=generated.txt \
    --output_filename=scores.csv \
    --lang="german" \ # "chinese" for Chinese
    --use_stemmer=true
```

For BERTScore, you should first download the [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext) and [bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased) models, and then use the [bert_score](https://github.com/Tiiiger/bert_score) toolkit. The evaluation command like this:

```bash
model_path=xxx/chinese-bert-wwm-ext # For Chinese
model_path=xxx/bert-base-german-uncased # For German

bert-score -r $gold_file_path -c $generate_file_path --lang zh --model $model_path --num_layers 8 # For Chinese
bert-score -r $gold_file_path -c $generate_file_path --lang de --model $model_path --num_layers 8 # For German
```

## 6. Recommendation
We also kindly recommend two highly related great papers for cross-lingual dialogue summarization research: 
- [MSAMSum: Towards Benchmarking Multi-lingual Dialogue Summarization](https://aclanthology.org/2022.dialdoc-1.1/) (DialDoc@ACL 2022) \| [[Data](https://github.com/xcfcode/MSAMSum)]
- [The Cross-lingual Conversation Summarization Challenge](https://arxiv.org/abs/2205.00379) (INIG 2022)

## 7. Citation and Contact
If you find this work is useful or use the data in your work, please consider cite our paper:
```
@inproceedings{wang-etal-2022-clidsum,
    title = "{C}lid{S}um: A Benchmark Dataset for Cross-Lingual Dialogue Summarization",
    author = "Wang, Jiaan  and
      Meng, Fandong  and
      Lu, Ziyao  and
      Zheng, Duo  and
      Li, Zhixu  and
      Qu, Jianfeng  and
      Zhou, Jie",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.526",
    pages = "7716--7729",
    abstract = "We present ClidSum, a benchmark dataset towards building cross-lingual summarization systems on dialogue documents. It consists of 67k+ dialogue documents and 112k+ annotated summaries in different target languages. Based on the proposed ClidSum, we introduce two benchmark settings for supervised and semi-supervised scenarios, respectively. We then build various baseline systems in different paradigms (pipeline and end-to-end) and conduct extensive experiments on ClidSum to provide deeper analyses. Furthermore, we propose mDialBART which extends mBART via further pre-training, where the multiple objectives help the pre-trained model capture the structural characteristics as well as key content in dialogues and the transformation from source to the target language. Experimental results show the superiority of mDialBART, as an end-to-end model, outperforms strong pipeline models on ClidSum. Finally, we discuss specific challenges that current approaches faced with this task and give multiple promising directions for future research. We have released the dataset and code at https://github.com/krystalan/ClidSum.",
}
```
Please feel free to email Jiaan Wang (jawang1[at]stu.suda.edu.cn) for questions and suggestions.
