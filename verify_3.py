from deepface import DeepFace
import cv2
import os
from scripts.helper import get_face_text
import clip
import torch
from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("RN50", device=device)

class node:
    def __init__(self, sim, result, img):
        self.sim = sim
        self.result = result
        self.img = img

img_ind = 0
# main_folder = './face_final_original_1'
main_folder = './face_final_ts_single_1'
all_tt = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

file_name = main_folder + '/res.txt'
if not os.path.exists(file_name):
    open(file_name, "w").close()

with open(file_name, "w") as file:
    All_tt_imgs = []

    for tt in all_tt:
        folder = main_folder + f'/text_type_{tt}'
        print("Analysis for text_type : ", tt)

        file.write(f"Analysis for text type : {tt}")
        file.write("\n")

        prompt = get_face_text(tt)
        print(prompt)
        file.write(f"The text is : {prompt}")
        file.write("\n")

        # text = clip.tokenize([prompt]).to(device)

        All = []
        for i in range(1, 3):

            for j in range(30):
                path_1 = folder + f'/check_{i}/new_img_{img_ind}_{j}.png'
                path_2 = folder + f'/check_{i}/og_img_{img_ind}_0.png'
                # print(path_1)

                try:
                    result = DeepFace.verify(img1_path=path_1,
                                             img2_path=path_2,
                                             model_name="VGG-Face")
                    # All.append(node(sim=result['distance'], result=result, img=path_1))
                    nn = node(sim=0, result=[], img=path_1)
                    cnt = 0

                    for k in range(1):
                        path_2 = folder + f'/check_{i}/og_img_{img_ind}_{k}.png'
                        try:
                            result = DeepFace.verify(img1_path=path_1,
                                                     img2_path=path_2,
                                                     model_name="VGG-Face")

                            nn.result.append(result)
                            nn.sim += result['distance']
                            # print(path_2)
                            # print(result['distance'])
                            cnt += 1
                        except:
                            pass

                    nn.sim /= cnt
                    # print(cnt, nn.sim)
                    All.append(nn)
                    # image = preprocess(Image.open(path_1)).unsqueeze(0).to(device)
                    # with torch.no_grad():
                    #     image_features = model.encode_image(image)
                    #     text_features = model.encode_text(text)
                    #
                    #     logits_per_image, logits_per_text = model(image, text)
                    # print(logits_per_image)

                except:
                    pass

        All = sorted(All, key=lambda x: x.sim)

        og_img = folder + f'/check_1/og_img_{img_ind}_0.png'
        # print(og_img)
        og_img = cv2.imread(og_img)
        og_img = cv2.resize(og_img, (512, 512))
        best_imgs = [og_img]

        save_num = 15

        for i in range(min(len(All), save_num)):
            # print(All[i].img)
            # print(All[i].result)

            file.write(All[i].img)
            file.write("\n")

            file.write(str(All[i].sim))
            file.write("\n")
            file.write("\n")

            img = cv2.imread(All[i].img)
            best_imgs.append(img)

        # print(len(best_imgs))

        while(len(best_imgs) < save_num + 1):
            best_imgs.append(img * 0)

        # print(len(best_imgs))

        best_imgs = cv2.hconcat(best_imgs)
        All_tt_imgs.append(best_imgs)


    All_tt_imgs = cv2.vconcat(All_tt_imgs)

    uq = main_folder.split("/")[1]
    cv2.imwrite(f'{main_folder}/all_{uq}_{img_ind}.png', All_tt_imgs)
