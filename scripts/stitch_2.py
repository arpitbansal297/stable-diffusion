import cv2

img_ind = 15
folder = './seg_final_2'
cnt = 1
all = None

tt = [2, 3, 4, 10, 11]

for t in tt:
    row = None
    for i in range(1, 3):
        for j in range(10):
            path = folder + f'/text_type_{t}/check_{i}/new_img_{img_ind}_{j}.png'
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

