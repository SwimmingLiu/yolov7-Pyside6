# yolov7-Pyside6 可视化界面检测图像和视频


这个项目基于 [YOLOv7](https://github.com/WongKinYiu/yolov7) & [PyQt5-YOLOv5](https://github.com/Javacr/PyQt5-YOLOv5) 开发

当GUI启动时，会自动加载"ptmodel"文件夹下中，从[YOLOv7 models](https://github.com/WongKinYiu/yolov7/releases/) 下载的pt文件

开发这个项目，是为了给用户提供一个操作方便且能快速检测的程序。

系统环境参数:

    OS : Windows 11 
    CPU : Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz
    GPU : NVIDIA GeForce GTX 1660Ti 6GB


​    
## 检测图像和视频的案例

   <img src="https://user-images.githubusercontent.com/53814462/218943807-1563fe4f-81b6-4148-89c3-71bedf5d2714.png" width="800"/><br/>
   <img src="https://user-images.githubusercontent.com/53814462/218943813-17173906-1ee8-4293-90ab-bf1b11bf47a8.png" width="800"/><br/>
   <img src="https://user-images.githubusercontent.com/53814462/218943823-bc03dadd-af32-43fb-a873-64741d8c9c6d.png" width="800"/><br/>

## 准备工作
```bash
    ### 创建虚拟环境
    conda create -n yolov7=python 
    ### 安装依赖
    pip install -r requirements.txt
    ### 安装Cuda对应版本的pytorch
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

Download [Pytorch](https://pytorch.org/get-started/locally/)

## 运行方式

```bash
  python main.py
```

## 功能

1. 支持图片和视频
2. 切换模型
3. 调节IoU
4. 调节置信度
5. 开始/暂停/停止 检测
6. 统计结果数量
8. 导出已检测的图片/视频

## 案例视频
在Bilibili [YOLOv7检测界面-Pyside6-GUI](https://www.bilibili.com/video/BV1oy4y1f7t1/?spm_id_from=333.999.0.0)

## 参考
- [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [PyQt5-YOLOv5](https://github.com/Javacr/PyQt5-YOLOv5)
- [YOLOv7-Predict-with-UI](https://github.com/swiminggay/YOLOv7-Predict-with-UI)
- [Python Qt 简介](https://www.byhy.net/tut/py/gui/qt_01/)

## 结尾
如果你觉得这个项目对你有所帮助请为我star. ⭐⭐⭐
