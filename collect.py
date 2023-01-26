import os
import errno
import cv2

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


main_dir = './paper_segmentation_2'
create_folder(main_dir)

# list_1 = ['./seg_final_1/text_type_4/check_2/new_img_0_2.png',
#           './seg_final_1/text_type_3/check_2/new_img_0_6.png',
#           './seg_final_1/text_type_11/check_2/new_img_0_2.png',
#           './seg_final_1/text_type_2/check_2/new_img_0_1.png']
#
# label = './seg_final_1/text_type_1/check_1/label_0.png'
# all = cv2.imread(label)
# cv2.imwrite(f'{main_dir}/seg_label_0.png', all)
#
# index = 0
# tt = [4, 3, 11, 2]
# cnt=0
# for img in list_1:
#     all = cv2.imread(img)
#     cv2.imwrite(f'{main_dir}/seg_{index}_type_{tt[cnt]}.png', all)
#     cnt+=1
#
#
# list_1 = ['./seg_final_1/text_type_4/check_1/new_img_4_5.png',
#           './seg_final_1/text_type_3/check_1/new_img_4_2.png',
#           './seg_final_1/text_type_11/check_1/new_img_4_3.png']
#
# label = './seg_final_1/text_type_1/check_1/label_4.png'
# all = cv2.imread(label)
# cv2.imwrite(f'{main_dir}/seg_label_4.png', all)
#
# index = 4
# tt = [4, 3, 11]
# cnt=0
# for img in list_1:
#     all = cv2.imread(img)
#     cv2.imwrite(f'{main_dir}/seg_{index}_type_{tt[cnt]}.png', all)
#     cnt+=1


list_1 = ['./seg_final_2/text_type_4/check_2/new_img_15_0.png',
          './seg_final_2/text_type_3/check_2/new_img_15_6.png',
          './seg_final_2/text_type_11/check_1/new_img_15_0.png']

label = './seg_final_2/text_type_4/check_1/label_15.png'
all = cv2.imread(label)
cv2.imwrite(f'{main_dir}/seg_label_15.png', all)

index = 15
tt = [4, 3, 11]
cnt=0
for img in list_1:
    all = cv2.imread(img)
    cv2.imwrite(f'{main_dir}/seg_{index}_type_{tt[cnt]}.png', all)
    cnt+=1

list_1 = ['./seg_final_2/text_type_4/check_1/new_img_18_5.png',
          './seg_final_2/text_type_3/check_2/new_img_18_5.png',
          './seg_final_2/text_type_11/check_1/new_img_18_3.png']

label = './seg_final_2/text_type_4/check_1/label_18.png'
all = cv2.imread(label)
cv2.imwrite(f'{main_dir}/seg_label_18.png', all)

index = 18
tt = [4, 3, 11]
cnt=0
for img in list_1:
    all = cv2.imread(img)
    cv2.imwrite(f'{main_dir}/seg_{index}_type_{tt[cnt]}.png', all)
    cnt+=1
