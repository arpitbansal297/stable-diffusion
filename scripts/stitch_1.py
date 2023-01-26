import cv2

img_ind = 4
folder = './face_detect_original/text_type_3'
cnt = 1
all = None
for i in range(1, 14):
    row = None
    for j in range(5):
        path = folder + f'/check_{i}/new_img_{img_ind}_{j}.png'
        print(path)
        img = cv2.imread(path)
        print(len(img))

        if row is None:
            row = img
        else:
            row = cv2.hconcat([row, img])



    if all is None:
        all = row
    else:
        all = cv2.vconcat([all, row])

f = folder.split('/')[1]
cv2.imwrite(f'{folder}/all_{f}_{img_ind}_see.png', all)

