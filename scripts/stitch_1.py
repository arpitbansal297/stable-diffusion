import cv2

img_ind = 0
folder = './face_g3_fast_2_text_type_5'
cnt = 1
all = None
for i in range(1, 17):
    row = None
    for j in range(5):
        path = folder + f'/check_{i}/new_img_{img_ind}_{j}.png'
        print(path)
        img = cv2.imread(path)

        if row is None:
            row = img
        else:
            row = cv2.hconcat([row, img])



    if all is None:
        all = row
    else:
        all = cv2.vconcat([all, row])

f = folder.split('/')[1]
cv2.imwrite(f'{folder}/all_{f}_{img_ind}.png', all)

