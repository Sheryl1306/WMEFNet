import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'<drive>:\<project_name>\<version>\<main_folder>\runs\train\wmef\weights\best.pt') # select your model.pt path
    model.predict(#source=r'F:\projct2\tr2\wmef\dataset\AI-TOD\test\images\0000197_01958_d_0000151__600_0.png',
                  source=r'<drive>:\<project_name>\<version>\<main_folder>\dataset\AI-TOD\test\images\P2802__1.0__3000___600.png',
                  conf=0.25,
                  project='runs/detect',
                  name='AI55',
                  save=True,
                #   visualize=True # visualize model features maps
                  #show_conf=False,
                  #show_labels=False
                  )