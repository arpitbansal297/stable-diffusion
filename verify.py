from deepface import DeepFace
from PIL import Image



result = DeepFace.verify(img1_path=f"./face_g3/check_9/og_img.png", img2_path=f"./face_g3/check_3/out_img_0.png",
                         model_name="VGG-Face")
print(result)


