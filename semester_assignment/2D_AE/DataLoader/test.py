from glob import glob
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import cv2


# 위치를 반환함
# DATA_PATH_LIST = glob('/Users/minjaelee/Desktop/mnist_png/training/1/*.png')
# print(DATA_PATH_LIST)

img = cv2.imread('/Users/minjaelee/Desktop/5.png')
#img = cv2.imread('../6.png')
aug = iaa.Cutout(nb_iterations=2, size = 0.3) # 개수 2개, 사이즈 0.3
image_aug = aug(image = img) # 인자 이름을 image로 설정(단일 이미지 적용)
save_path = '/Users/minjaelee/Desktop/save.png'
cv2.imwrite(save_path,image_aug)
print(img.shape)
