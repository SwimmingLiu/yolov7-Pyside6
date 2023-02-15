# yolov7-Pyside6 Image and Video Inference with UI


This project is based on [YOLOv7](https://github.com/WongKinYiu/yolov7) & [PyQt5-YOLOv5](https://github.com/Javacr/PyQt5-YOLOv5)

When the GUI is launched, it will automatically detect the presence of the [YOLOv7 models](https://github.com/WongKinYiu/yolov7/releases/) that were previously added to the "ptmodel" folder.


With this project, our aim was to offer end users quick inferences that would simplify their work. To achieve this, we focused on developing a lightweight and fast project that would meet their needs.

The system features:

    OS : Windows 11 
    CPU : Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz
    GPU : NVIDIA GeForce GTX 1660Ti 6GB
    
    
## Inferences on Image and Video

   <img src="https://user-images.githubusercontent.com/53814462/218943807-1563fe4f-81b6-4148-89c3-71bedf5d2714.png" width="800"/><br/>
   <img src="https://user-images.githubusercontent.com/53814462/218943813-17173906-1ee8-4293-90ab-bf1b11bf47a8.png" width="800"/><br/>
   <img src="https://user-images.githubusercontent.com/53814462/218943823-bc03dadd-af32-43fb-a873-64741d8c9c6d.png" width="800"/><br/>

## Prepare
```bash
    ### create virtual env 
    conda create -n yolov7=python 
    ### requirements
    pip install -r requirements.txt
    ### Download related version Pytorch 
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```
   Download [Pytorch](https://pytorch.org/get-started/locally/) 

## Run 
```bash
  python main.py
```

## Function

1. support image/video as input
2. change model
3. change IoU
4. change confidence
5. paly/pause/stop
6. show result 
8. export detected image/video
## Demo Video
Check in [YOLOv7检测界面-Pyside6-GUI](https://www.bilibili.com/video/BV1oy4y1f7t1/?spm_id_from=333.999.0.0)

## Reference
- [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [PyQt5-YOLOv5](https://github.com/Javacr/PyQt5-YOLOv5)
- [YOLOv7-Predict-with-UI](https://github.com/swiminggay/YOLOv7-Predict-with-UI)
- [Python Qt 简介](https://www.byhy.net/tut/py/gui/qt_01/)

## End
Please consider giving me a star if this project is helpful to you. ⭐⭐⭐
