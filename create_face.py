import os
import errno
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

main_dir = './face_final_original_2'
create_folder(main_dir)

all_tt = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

for tt in all_tt:
    tt_folder = f'{main_dir}/text_type_{tt}'
    create_folder(tt_folder)

    p = f"python scripts/fast_2_detect_original.py --image_index 6 --text_type {tt} --n_iter 1 --optim_guidance_3 --fr_crop --optim_num_steps 2 --optim_guidance_3_wt 20000 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_1/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
    print(p)

for tt in all_tt:
    tt_folder = f'{main_dir}/text_type_{tt}'
    create_folder(tt_folder)

    p = f"python scripts/fast_2_detect_original.py --image_index 6 --scale 3 --text_type {tt} --n_iter 1 --optim_guidance_3 --fr_crop --optim_num_steps 2 --optim_guidance_3_wt 20000 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_2/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
    print(p)


main_dir = './face_final_original_3'
create_folder(main_dir)

all_tt = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

for tt in all_tt:
    tt_folder = f'{main_dir}/text_type_{tt}'
    create_folder(tt_folder)

    p = f"python scripts/fast_2_detect_original.py --image_index 8 --text_type {tt} --n_iter 1 --optim_guidance_3 --fr_crop --optim_num_steps 2 --optim_guidance_3_wt 20000 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_1/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
    print(p)

for tt in all_tt:
    tt_folder = f'{main_dir}/text_type_{tt}'
    create_folder(tt_folder)

    p = f"python scripts/fast_2_detect_original.py --image_index 8 --scale 3 --text_type {tt} --n_iter 1 --optim_guidance_3 --fr_crop --optim_num_steps 2 --optim_guidance_3_wt 20000 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_2/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
    print(p)




# main_dir = './face_final_ts_single_1'
# create_folder(main_dir)
#
# for tt in all_tt:
#     tt_folder = f'{main_dir}/text_type_{tt}'
#     create_folder(tt_folder)
#
#     p = f"python scripts/fast_2_detect_original_multiple_pictures.py --image_mean 1 --text_type {tt} --n_iter 1 --optim_guidance_3 --fr_crop --optim_num_steps 2 --optim_guidance_3_wt 20000 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_1/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
#     print(p)
#
# for tt in all_tt:
#     tt_folder = f'{main_dir}/text_type_{tt}'
#     create_folder(tt_folder)
#
#     p = f"python scripts/fast_2_detect_original_multiple_pictures.py --image_mean 1 --scale 3 --text_type {tt} --n_iter 1 --optim_guidance_3 --fr_crop --optim_num_steps 2 --optim_guidance_3_wt 20000 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_2/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
#     print(p)
#
#
#
#
# main_dir = './face_final_ts_multiple_1'
# create_folder(main_dir)
#
# for tt in all_tt:
#     tt_folder = f'{main_dir}/text_type_{tt}'
#     create_folder(tt_folder)
#
#     p = f"python scripts/fast_2_detect_original_multiple_pictures.py --image_mean 5 --text_type {tt} --n_iter 1 --optim_guidance_3 --fr_crop --optim_num_steps 2 --optim_guidance_3_wt 20000 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_1/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
#     print(p)
#
# for tt in all_tt:
#     tt_folder = f'{main_dir}/text_type_{tt}'
#     create_folder(tt_folder)
#
#     p = f"python scripts/fast_2_detect_original_multiple_pictures.py --image_mean 5 --scale 3 --text_type {tt} --n_iter 1 --optim_guidance_3 --fr_crop --optim_num_steps 2 --optim_guidance_3_wt 20000 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_2/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
#     print(p)


