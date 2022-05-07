# Complementary Newer College LiDAR-Camera Synchronization Annotations

## Download Link

As described in the paper, we have annotated a very small datasets with the setting of stride/window as 1 / 6.
The annotation targets were created with 
[training_data_serialization/newer_college_labelling_data_genearator.py](../training_data_serialization/newer_college_labelling_data_genearator.py),
which you can take a closer look to see how it works.

Please download the datasets with this link: https://drive.google.com/file/d/1-mFaLKPZSeVgdZWgCx4rym-0sWlURw4u/view


## Data Walk-Thru

Unzip it and you will find the structure as:

```
├── 0
│   ├── agreed_labels.txt
│   └── testing
└── 1
    ├── agreed_labels.txt
    └── testing
           ├── 39_seqid_13823_4490553f59a047d3a344c0e202b0da51_gt5.png
           ├── ....png
```

The testing folder contains the exact visualization image with the following information encoded (taking
`39_seqid_13823_4490553f59a047d3a344c0e202b0da51_gt5.png` as one example):
- [X] id in the sub-testing folder, e.g. `39`
- [X] unique sequence id, e.g. `13823`, which can help you identify the frame in the raw data
- [x] a hash 
- [x] psudo ground truth/baseline synced results provided by Newer College, e.g. `5`

The agreed_labels.txt reads as:

```
id,label
0,disagreed
1,disagreed
2,disagreed
3,5
4,disagreed
5,5
6,3
7,3
8,disagreed
....
```

where the first column is the id in the sub-testing folder, which can be matched to image in the testing folder; 
the second column shows the annotated results, with one of the following:

- a single or several number(s), which all labelers agree
- disagreed
- unsure

In our test, only the agreed ones are used.

## A Backup

Technically, we don't need the images in the folder at all. However, I still provide it in this way since this 
shows how the data are annotated and this is also a format that directly works 
with [newer_college_gt_runner.py](../evaluation/evaluation_over_datasets/newer_college_gt_runner.py).

In the meanwhile, I also provide a processed [annotation.csv](annotation.csv), which contains all information, without requiring the images.
This is in case that the download link is broken and someone still would like to use these datasets. But, you will
need to make some changes to the [newer_college_gt_runner.py](../evaluation/evaluation_over_datasets/newer_college_gt_runner.py)
to get it work. The [annotation.csv](annotation.csv) is created by the [converter.py](converter.py).
