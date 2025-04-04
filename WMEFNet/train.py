import warnings, os


warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'E:\WMEFNet\ultraylytics\cfg\models\WMEFNet\wmef.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=r'E:\WMEFNet\dataset\Vsdata.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=4,
                project='runs/train',
                name='XXX',
                )

