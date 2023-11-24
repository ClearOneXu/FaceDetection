# FaceDetection in Python

The FacceDetection code is depend on https://github.com/biubug6/Pytorch_Retinaface

## Clone and install
1. Download 
```Shell
git clone git@github.com:ClearOneXu/FaceDetection.git
cd FaceDetection
git git@github.com:biubug6/Pytorch_Retinaface.git
```

2. Download the Resnet50_Final.pth through https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1 and put it in pretrained_model. Password: fstq .
3. Codes are based on Python 3.10. Run python -V to check python version.
4. Install requirements.
```Shell
pip install -r requirements.txt
```

## Run the code
The new detected face images will be saved in p_data/detected/, the txt file of the position of predicted images will be saved in p_data/pred/, and the ground truth should be put in p_data/ground_truth.

#### Program Input

The input to the program is a single argument either specifying a path to a folder of images or to a single image as follows. If there is no input image, the default images path will be p_data/photos/
```Shell
python script.py 1_faces_1.jpg
python script.py images/
```

#### Program Output
```Shell
1_faces_1.png 1
4_faces_2.png 4
3_faces_4.png 3
```


#### Evaluation metric
The evaluation metric contains Precision, Recall, F1-score and Average Precision. 
##### Program Input with Evaluation metric
```Shell
python script.py 1_faces_1.jpg -e
python script.py images/ -e
```
##### Program Output with Evaluation metric
```Shell
Precision: 0.9583
Recall: 1.0952
F1-score: 1.0222
Average Precision (AP) with IoU threshold 0.5: 0.7250
```

## Unitest
```Shell
python detect_unitest.py
```

## Note.pdf
A separate note briefly explaining challenges, thought processes and justification for technical choices.
