import os
import errno
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

# main_dir = './seg_final_1'
# create_folder(main_dir)
#
# for tt in range(1, 12):
#     tt_folder = f'{main_dir}/text_type_{tt}'
#     create_folder(tt_folder)
#
#     p = f"python scripts/fast_segmentation.py --text_type {tt} --scale 1.5 --n_iter 5 --optim_guidance_3 --optim_num_steps 10 --optim_guidance_3_wt 400 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_1/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
#     print(p)
#
# for tt in range(1, 12):
#     tt_folder = f'{main_dir}/text_type_{tt}'
#     create_folder(tt_folder)
#
#     p = f"python scripts/fast_segmentation.py --text_type {tt} --scale 2.0 --n_iter 5 --optim_guidance_3 --optim_num_steps 10 --optim_guidance_3_wt 400 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_2/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
#     print(p)

main_dir = './seg_final_2'
create_folder(main_dir)

tt_all = [2, 3, 4, 10, 11]

for tt in tt_all:
    tt_folder = f'{main_dir}/text_type_{tt}'
    create_folder(tt_folder)

    p = f"python scripts/fast_segmentation_166.py --to_take 15 --text_type {tt} --scale 1.5 --n_iter 16 --optim_guidance_3 --optim_num_steps 10 --optim_guidance_3_wt 400 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_1/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
    print(p)

for tt in tt_all:
    tt_folder = f'{main_dir}/text_type_{tt}'
    create_folder(tt_folder)

    p = f"python scripts/fast_segmentation_166.py --to_take 15 --text_type {tt} --scale 2.0 --n_iter 16 --optim_guidance_3 --optim_num_steps 10 --optim_guidance_3_wt 400 --optim_original_guidance --ddim_steps 500 --optim_folder {main_dir}/text_type_{tt}/check_2/ --ckpt /cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --n_samples 1 --H 512 --W 512"
    print(p)
