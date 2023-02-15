import argparse
import time
from pathlib import Path
from PySide6.QtCore import Signal
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PySide6.QtCore import QThread
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from lib import glo


class YoloThread(QThread):
    send_input = Signal(np.ndarray)
    send_output = Signal(np.ndarray)
    send_result = Signal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = Signal(str)
    send_percent = Signal(int)
    send_fps = Signal(str)

    def __init__(self):
        super(YoloThread, self).__init__()
        self.weights = './ptmodel/best.pt'
        self.current_weight = './ptmodel/best.pt'
        self.conf = 0.25
        self.iou = 0.40
        self.is_continue = True  # continue/pause
        self.jump_out = False  # jump out of the loop
        self.percent_length = 1000  # progress bar

    @torch.no_grad()
    def run(self,
            imgsz=640,
            device='',
            view_img=False,
            save_conf=False,
            nosave=False,
            classes=None,
            agnostic_nms=False,
            augment=False,
            update=False,
            project='result',
            name='exp',
            exist_ok=False,
            no_trace=False
            ):

        # Initialize
        try:
            source = glo.get_value('inputPath')
            device = select_device(device)
            half = False
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            # Directories
            save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
            save_dir.mkdir(parents=True, exist_ok=True)  # make dir
            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
            if half:
                model.half()  # to FP16
            # Set Dataloader
            vid_path, self.vid_writer = None, None
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            old_img_w = old_img_h = imgsz
            old_img_b = 1

            dataset = iter(dataset)

            # 参数设置
            t0 = time.time()
            count = 0
            # 开始处理每一张图片
            while True:

                # 停止检测
                if self.jump_out:
                    self.send_percent.emit(0)
                    if self.vid_cap is not None:
                        self.vid_cap.release()
                    self.send_msg.emit('Stop')
                    if self.vid_writer is not None:
                        self.vid_writer.release()
                    break

                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check img_size
                    if half:
                        model.half()  # to FP16
                    # Get names and colors
                    names = model.module.names if hasattr(model, 'module') else model.names
                    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    old_img_w = old_img_h = imgsz
                    old_img_b = 1
                    self.current_weight = self.weights

                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    # 原始图片送入 input框
                    self.send_input.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    # 处理processBar
                    count += 1

                    if self.vid_cap:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    # 处理图片
                    statistic_dic = {name: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    # Warmup
                    if device.type != 'cpu' and (
                            old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                        old_img_b = img.shape[0]
                        old_img_h = img.shape[2]
                        old_img_w = img.shape[3]
                        for i in range(3):
                            model(img, augment=augment)[0]

                    # Inference
                    t1 = time_synchronized()
                    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                        pred = model(img, augment=augment)[0]
                    t2 = time_synchronized()

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf, self.iou, classes=classes,
                                               agnostic=agnostic_nms)
                    t3 = time_synchronized()
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                        p = Path(p)  # to Path
                        self.save_path = str(save_dir / p.name)  # img.jpg
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                # Add bbox to image
                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)
                    # Stream results
                    self.send_output.emit(im0)
                    self.send_result.emit(statistic_dic)

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(self.save_path, im0)
                            print(f" The image with the result is saved in: {self.save_path}")
                        else:  # 'video' or 'stream'
                            if vid_path != self.save_path:  # new video
                                vid_path = self.save_path
                                if isinstance(self.vid_writer, cv2.VideoWriter):
                                    self.vid_writer.release()  # release previous video writer
                                fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                self.vid_writer = cv2.VideoWriter(self.save_path,
                                                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                                                  fps,
                                                                  (w, h))
                            self.vid_writer.write(im0)
                    if percent == self.percent_length:
                        print(count)
                        self.send_percent.emit(0)
                        self.send_msg.emit('Finished')
                        if self.vid_writer is not None:
                            self.vid_writer.release()
                        break

        except Exception as e:
            self.send_msg.emit("程序出错啦!!!   " + str(e))
