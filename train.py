import warnings, os
os.environ["CUDA_VISIBLE_DEVICES"]="4"    
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/ACD-DETR.yaml')
    # model.load('') # loading pretrain weights
    # /root/ty/RTDETR-main/dataset/DOTA/DOTAv1.yaml 
    # /root/ty/RTDETR-main/dataset/VisDrone/data.yaml
    model.train(data='/root/ty/RTDETR-main/dataset/VisDrone/data.yaml',
                cache=True,
                imgsz=640,
                epochs=200,
                batch=4, 
                workers=4, 
                # device='0,1',
                # resume='/root/ty/RTDETR-main/runs/train/rtdetr-r34/weights/last.pt', # last.pt path
                project='runs/train',
                name='ACD-DETR', # exp
                )