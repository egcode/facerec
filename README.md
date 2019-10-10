
# TODO: - Image Here



- [1. QuickStart](#1.-QuickStart)
- [2. Train](#1.-Train)
- [3. Dataset Cleanup](#1.-Dataset-Cleanup)


## 1. QuickStart
```
python3 app/export_embeddings.py \
--model_path ./data/pth/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962.pth \
--data_dir ./data/dataset_got/dataset_lanister_raw/ \
--output_dir data/out_embeddings/  \
--model_type IR_50 \
--is_aligned 0 \
--with_demo_images 1 \
--image_size 112 \
--image_batch 5 \
--h5_name dataset_lanister.h5

```
## 2. Train

## 3. Dataset Cleanup




## Face Recognition

This is a Pytorch implementation of the face recognizer described in the papers:
* [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf). (recomended)
* [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf). 
* [A Discriminative Feature Learning Approach for Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf). 

## Face detection
Face detection on images and live camera performed using Tensorflow MTCNN implementation. [Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). 
A OpenCV C++ implementation can be found [here](https://github.com/egcode/mtcnn-opencv)


## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962](https://drive.google.com/open?id=1itqqp1EWf6sfi0K4i6QYBR_j3NS7gw2i) | 0.9962        | M1-Celeb    | [IR_50](https://github.com/egcode/facerec/blob/master/models/irse.py) |

NOTE: If you use any of the models, please do not forget to give proper credit to those providing the training dataset as well.


## Compatibility
The code is tested using Pytorch 1.0 under OSX 10.14.5 and Ubuntu 16.04 with Python 3.6. 

