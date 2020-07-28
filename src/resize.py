from PIL import Image
import os, glob
# 가위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서


image_folder = ['rock','scissor', 'paper']

# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size=(28,28)
image_dir_path  = os.getenv("HOME")+ "/PycharmProjects/gawibawibo/datasets/"

for i in image_folder:
    images = glob.glob(image_dir_path + i + "/*.jpg")

    for img in images:
        #print(img)
        old_img=Image.open(img)
        new_img=old_img.resize(target_size,Image.ANTIALIAS)
        new_img.save(img,"JPEG")

print("resize 완료!")