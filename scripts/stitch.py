import cv2

folder = './ablations_2'
cnt = 1
all = None
for i in range(6):
    row = None
    for j in range(5):
        path = folder + f'/check_{cnt}/out_img_4.png'
        print(path)
        img = cv2.imread(path)

        if row is None:
            row = img
        else:
            row = cv2.hconcat([row, img])

        cnt+=1


    if all is None:
        all = row
    else:
        all = cv2.vconcat([all, row])


cv2.imwrite(f'{folder}/all.png', all)

