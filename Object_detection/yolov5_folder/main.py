import torch
from glob import glob
from sklearn.model_selection import train_test_split
import yaml
#print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
#print(torch.cuda.get_arch_list())
#print(torch.__version__)


# img_list = glob('/home/minjaelee/Desktop/yolov5_folder/dataset/export/images/*.jpg')
#print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# with open('/home/minjaelee/Desktop/yolov5_folder/dataset/train.txt', 'w') as f:
#   f.write('\n'.join(train_img_list) + '\n')

# with open('/home/minjaelee/Desktop/yolov5_folder/dataset/val.txt', 'w') as f: kernel image is available for execution on the device
# print(data)

# data['train'] = '/home/minjaelee/Desktop/yolov5_folder/dataset/train.txt'
# data['val'] = '/home/minjaelee/Desktop/yolov5_folder/dataset/val.txt'

# with open('/home/minjaelee/Desktop/yolov5_folder/dataset/data.yaml', 'w') as f:
#   yaml.dump(data, f)

# print(data)

# print(torch.__version__)

# train.py가 있는 yolov5 폴더에서 돌려야한다.
# python train.py --img 416 --batch 16 --epochs 5 --data /home/minjaelee/Desktop/yolov5_folder/dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name gun_yolov5s_results