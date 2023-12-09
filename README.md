[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/f7NyygB-)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13193397&assignment_repo_type=AssignmentRepo)

# Fine Tuning DETR 

Robin Gould

The goal of this project is to fine tune DETR (DEtection TRansformer) from [End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr) to do object detection on trail cameras for deer and boars. Detecting deer and boars is useful for automatically getting population statistics for both animal populations. This can help independantly check the reliability of other population counting methods, and increase their robustness. Understanding the population variances year over year can help understand the impact wild boar populations have on deer populations, as well as the impact different hunting rules and regulations have on the populations of both. If we utilize a wide array of trail cameras, we can map out the geographical locations these animals tend towards, further giving us insight into the ecosystem. 

# Dataset

We used the Roboflow trail camera dataset, which is a dataset with deer and hogs taken from trail cameras. There is 1311 images, so this is a perfect opportunity to fine tune DETR, due to our small dataset size.

Here is the dataset below, simply download it as a zip file then place it into Google drive to be easily processed by colab.

https://universe.roboflow.com/roboflow-100/trail-camera

# Data preprocessing

While the data was initially in COCO format, which DETR expects, we needed to move it into following format. 
The data was split into 239 validation images, 941 training images, and 131 test images. 

The training images went into train2017, and the validation images went into val2017.

```
path/to/coco/
├ annotations/  # JSON annotations
│  ├ annotations/custom_train.json
│  └ annotations/custom_val.json
├ train2017/    # training images
└ val2017/      # validation images
```

# Model - DETR

To train the model, a line like the following is used

```
!python main.py \
  --dataset_file "/path/to/dataset/ \
  --coco_path "/path/to/images/" \
  --output_dir "/path/to/output/files/" \
  --resume "detr-r50_no-class-head.pth" \
  --num_classes $num_classes \
  --epochs 10
```

We are resuming training from some DETR checkpoint, in order to fine tune the model. 

We also only ran 10 epochs.

# Eval
We used DETR plot_logs to gather the performance from the outputs file specified above. We plot and measure the average precision [(the most important metric for COCO datasets)](https://cocodataset.org/#detection-eval), total loss, classification loss, bounding box distance loss, GIoU loss, as well as the class error and unscaled cardinality error.


![image](https://github.com/UConnAI/cse5097-final-project-fine-tuning-detr/assets/13643473/051d5a11-b6d4-4eab-b581-7af37dd2d473)

![image](https://github.com/UConnAI/cse5097-final-project-fine-tuning-detr/assets/13643473/46278ba6-8b6e-4091-8b4f-92c839bacb47)

![image](https://github.com/UConnAI/cse5097-final-project-fine-tuning-detr/assets/13643473/08ab19f0-aa14-464a-9351-adb3d18344da)

As we can see, the general trend is that our losses and errors trend downwards, while the statistic we are most interested in, average precision, rises to ~90%. 

These results are quite good for such a small dataset trained on only 10 epochs. 


# Example of it working on trailcam picture from validation dataset

```
img_name = '/content/data/custom/val2017/I__00055_JPG_jpg.rf.b0cf74d12bfe0d81f434218000137988.jpg'
im = Image.open(img_name)

run_worflow(im,
            model)
```

![image](https://github.com/UConnAI/cse5097-final-project-fine-tuning-detr/assets/13643473/2df782a9-3be5-450e-8de2-5c9c9d53bdb2)


