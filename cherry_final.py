import cv2

# og_img = './face_final_original_1/text_type_1/check_1/og_img_4.png'
#
# imgs = ['./face_final_original_1/text_type_1/check_2/new_img_5_0.png',
#         './face_final_original_1/text_type_3/check_1/new_img_5_9.png',
#         './face_final_original_1/text_type_4/check_2/new_img_5_23.png',
#         './face_final_original_1/text_type_5/check_2/new_img_5_23.png',
#         './face_final_original_1/text_type_8/check_2/new_img_5_22.png',
#         './face_final_original_1/text_type_9/check_1/new_img_5_3.png',
#         './face_final_original_1/text_type_10/check_2/new_img_5_8.png',
#         './face_final_original_1/text_type_11/check_2/new_img_5_21.png',
#         './face_final_original_1/text_type_12/check_1/new_img_5_22.png',
#         './face_final_original_1/text_type_15/check_2/new_img_5_21.png',
#         './face_final_original_1/text_type_17/check_2/new_img_5_5.png',
#         './face_final_original_1/text_type_18/check_2/new_img_5_18.png',
#         './face_final_original_1/text_type_19/check_2/new_img_5_19.png']
#
# cherry = []
# img = cv2.imread(og_img)
# img = cv2.resize(img, (512, 512))
# cherry.append(img)
#
# for img in imgs:
#     print(img)
#     cherry.append(cv2.imread(img))
#
# all = cv2.hconcat(cherry)
# cv2.imwrite(f'./face_final_original_1/cherry_face_final_original_1.png', all)


# og_img = './face_final_original_1/text_type_1/check_1/og_img_4.png'
#
# # 5, 3, 8, 10, 11, 15, 17, 18, 19
#
# imgs = ['./face_final_original_1/text_type_5/check_2/new_img_5_23.png',
#         './face_final_original_1/text_type_3/check_1/new_img_5_9.png',
#         './face_final_original_1/text_type_8/check_2/new_img_5_22.png',
#         './face_final_original_1/text_type_10/check_2/new_img_5_8.png',
#         './face_final_original_1/text_type_11/check_2/new_img_5_21.png',
#         './face_final_original_1/text_type_15/check_2/new_img_5_21.png',
#         './face_final_original_1/text_type_17/check_2/new_img_5_5.png',
#         './face_final_original_1/text_type_18/check_2/new_img_5_18.png',
#         './face_final_original_1/text_type_19/check_2/new_img_5_19.png']
#
# cherry = []
# img = cv2.imread(og_img)
# img = cv2.resize(img, (512, 512))
# cherry.append(img)
#
# for img in imgs:
#     print(img)
#     cherry.append(cv2.imread(img))
#
# all = cv2.hconcat(cherry)
# cv2.imwrite(f'./face_final_original_1/cherry_face_final_original_1.png', all)


# og_img = './face_final_original_2/text_type_1/check_1/og_img_6.png'
#
# # 5, 3, 8, 10, 11, 15, 17, 18, 19
#
# imgs = ['./face_final_original_2/text_type_5/check_2/new_img_7_25.png',
#         './face_final_original_2/text_type_3/check_1/new_img_7_27.png',
#         './face_final_original_2/text_type_8/check_2/new_img_7_25.png',
#         './face_final_original_2/text_type_10/check_2/new_img_7_27.png',
#         './face_final_original_2/text_type_11/check_2/new_img_7_9.png',
#         './face_final_original_2/text_type_15/check_1/new_img_7_9.png',
#         './face_final_original_2/text_type_17/check_2/new_img_7_21.png',
#         './face_final_original_2/text_type_18/check_2/new_img_7_11.png',
#         './face_final_original_2/text_type_19/check_2/new_img_7_12.png']
#
# cherry = []
# img = cv2.imread(og_img)
# img = cv2.resize(img, (512, 512))
# cherry.append(img)
#
# for img in imgs:
#     print(img)
#     cherry.append(cv2.imread(img))
#
# all = cv2.hconcat(cherry)
# cv2.imwrite(f'./face_final_original_2/cherry_face_final_original_2.png', all)




og_img = './face_final_original_3/text_type_1/check_1/og_img_8.png'

# 5, 3, 8, 10, 11, 15, 17, 18, 19

imgs = ['./face_final_original_3/text_type_5/check_2/new_img_9_3.png',
        './face_final_original_3/text_type_3/check_1/new_img_9_20.png',
        './face_final_original_3/text_type_8/check_1/new_img_9_11.png',
        './face_final_original_3/text_type_10/check_2/new_img_9_6.png',
        './face_final_original_3/text_type_11/check_2/new_img_9_9.png',
        './face_final_original_3/text_type_15/check_1/new_img_9_27.png',
        './face_final_original_3/text_type_17/check_2/new_img_9_25.png',
        './face_final_original_3/text_type_18/check_2/new_img_9_16.png',
        './face_final_original_3/text_type_19/check_2/new_img_9_0.png']

cherry = []
img = cv2.imread(og_img)
img = cv2.resize(img, (512, 512))
cherry.append(img)

for img in imgs:
    print(img)
    cherry.append(cv2.imread(img))

all = cv2.hconcat(cherry)
cv2.imwrite(f'./face_final_original_3/cherry_face_final_original_3.png', all)





# og_img = 'seg_final_1/text_type_1/check_1/label_0.png'
#
# imgs = ['./seg_final_1/text_type_1/check_2/new_img_0_0.png',
#         './seg_final_1/text_type_2/check_2/new_img_0_1.png',
#         './seg_final_1/text_type_3/check_2/new_img_0_6.png',
#         './seg_final_1/text_type_4/check_2/new_img_0_2.png',
#         './seg_final_1/text_type_5/check_1/new_img_0_0.png',
#         './seg_final_1/text_type_6/check_1/new_img_0_5.png',
#         './seg_final_1/text_type_8/check_1/new_img_0_6.png',
#         './seg_final_1/text_type_10/check_2/new_img_0_3.png',
#         './seg_final_1/text_type_11/check_2/new_img_0_2.png']
#
# cherry = []
# img = cv2.imread(og_img)
# img = cv2.resize(img, (512, 512))
# cherry.append(img)
#
# for img in imgs:
#     print(img)
#     cherry.append(cv2.imread(img))
#
# all = cv2.hconcat(cherry)
# cv2.imwrite(f'./seg_final_1/cherry_seg_final_1_0.png', all)