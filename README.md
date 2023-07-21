# EMCR
Enhanced Machine Reading Comprehension Method for Aspect Sentiment Quadruplet Extraction

Code and data of the paper "Enhanced Machine Reading Comprehension Method for Aspect Sentiment Quadruplet Extraction, ECAI 2023"
Authors: 	Shuqin Ye, Zepeng Zhai and Ruifan Li

#### Requirements:

```
  python==3.6.9
  torch==1.2.0
  transformers==2.9.0
```
#### Original Datasets:

You can download the datasets from https://github.com/nustm/acos

#### Data Preprocess:

```
  python dataProcess.py
  python makeData_dual.py
  python makeData_standard.py
```

#### How to run:

```
  python main.py --mode train # For training
```
