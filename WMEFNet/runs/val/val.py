import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'F:\RT-DETR\RTDETR-20250111\RTDETR-main\runs\train\XBAI2\weights\best.pt')
    model.val(data=r'F:\RT-DETR\RTDETR-20250111\RTDETR-main\dataset\AI-TODdata.yaml',
              split='test',
              imgsz=640,
              batch=4,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='AI',
              )