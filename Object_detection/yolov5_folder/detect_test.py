from IPython.display import Image
import os

#val_img_path = val_img_list[4]
val_img_path = /home/minjaelee/Desktop/yolov5_folder/dataset/export/images/armas (1)_jpg.rf.c4150f819f9dc2ae75336e5b64e67d67.jpg
#python detect.py --weights /home/minjaelee/Desktop/yolov5_folder/yolov5/runs/train/gun_yolov5s_results5/weights/best.pt --img 416 --conf 0.5 --source “{val_img_path}”
python detect.py --weights /home/minjaelee/Desktop/yolov5_folder/yolov5/runs/train/gun_yolov5s_results5/weights/best.pt --img 416 --conf 0.5 --source “{/home/minjaelee/Desktop/yolov5_folder/dataset/export/images/ar1.jpg}”


Image(os.path.join(‘/home/minjaelee/Desktop/yolov5_folder/yolov5/inference/output’, os.path.basename(val_img_path)))