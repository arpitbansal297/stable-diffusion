from deepface import DeepFace
import cv2
import os

img_ind = 0
folder = './face_detect_original/text_type_7'

file_name = folder + '/res.txt'

if not os.path.exists(file_name):
    open(file_name, "w").close()

class node:
    def __init__(self, sim, result, img):
        self.sim = sim
        self.result = result
        self.img = img

with open(file_name, "w") as file:
    for img_ind in range(10):
        print("Analysis for img_ind : ", img_ind)

        file.write(f"Analysis for img_ind : {img_ind}")
        file.write("\n")

        All = []
        for i in range(1, 17):

            for j in range(5):
                path_1 = folder + f'/check_{i}/new_img_{img_ind}_{j}.png'
                path_2 = folder + f'/check_{i}/og_img_{img_ind}.png'
                # print(path_1)
                #
                try:
                    result = DeepFace.verify(img1_path=path_1,
                                             img2_path=path_2,
                                             model_name="VGG-Face")
                    # print(result)
                    All.append(node(sim=result['distance'], result=result, img=path_1))

                except:
                    pass

        All = sorted(All, key=lambda x: x.sim)

        og_img = folder + f'/check_1/og_img_{img_ind}.png'
        og_img = cv2.imread(og_img)
        og_img = cv2.resize(og_img, (512, 512))
        best_imgs = [og_img]

        for i in range(min(len(All), 10)):
            print(All[i].img)
            print(All[i].result)

            file.write(All[i].img)
            file.write("\n")

            file.write(str(All[i].result))
            file.write("\n")

            img = cv2.imread(All[i].img)
            best_imgs.append(img)

        best_imgs = cv2.hconcat(best_imgs)
        uq = folder.split("/")[1]
        cv2.imwrite(f'{folder}/all_{uq}_{img_ind}.png', best_imgs)
