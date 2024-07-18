import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import os

from pytorch_fid import fid_score

# 计算两张图像的psnr和ssim
def psnr_ssim(ori_img, test_img):
    psnr = peak_signal_noise_ratio(ori_img, test_img)
    ssim = structural_similarity(ori_img, test_img, channel_axis=True)
    mse = mean_squared_error(ori_img, test_img)

    return psnr, ssim, mse

if __name__ == "__main__":
    paths = ['./results/pix2pix/test_latest/FID/real/', './results/pix2pix/test_latest/FID/fake/']

    real_imgs = os.listdir(paths[0])
    fake_imgs = os.listdir(paths[1])

    f = open('./results/pix2pix/test_latest/evaluate.txt', 'w')

    psnr_sum = 0
    ssim_sum = 0
    mse_sum = 0

    for i in range(len(fake_imgs)):
        psnr_bands = 0
        ssim_bands = 0
        mse_bands = 0
        for j in range(3):
            real_img = cv2.imread(paths[0] + os.sep + real_imgs[i])[j]
            fake_img = cv2.imread(paths[1] + os.sep + fake_imgs[i])[j]

            psnr, ssim, mse = psnr_ssim(real_img, fake_img)   # 每个通道的psnr, ssim, mse
            psnr_bands += psnr
            ssim_bands += ssim
            mse_bands += mse

        psnr_sum += (psnr_bands / 3)
        ssim_sum += (ssim_bands / 3)
        mse_sum += (mse_bands / 3)
        f.write(fake_imgs[i] + ':\t' + 'psnr: ' + str(psnr_bands / 3) + '\t' + 'ssim: ' + str(ssim_bands / 3) + '\t' + 'mse: ' + str(mse_bands / 3) + '\n')

        print(paths[1] + os.sep + fake_imgs[i])

    psnr_mean = psnr_sum / len(fake_imgs)
    ssim_mean = ssim_sum / len(fake_imgs)
    mse_mean = mse_sum / len(fake_imgs)

    # FID计算
    fid_value = fid_score.calculate_fid_given_paths(paths, batch_size=32, device='cuda:0', dims=2048, num_workers=0)

    f.write('psnr_mean: ' + str(psnr_mean) + '\n' + 'ssim_mean: ' + str(ssim_mean)
            + '\n' + 'mse_mean: ' + str(mse_mean) + '\n' + 'fid: ' + str(fid_value) + '\n')

    # f.write('psnr_mean: ' + str(psnr_mean) + '\n' + 'ssim_mean: ' + str(ssim_mean)
    #         + '\n' + 'mse_mean: ' + str(mse_mean) + '\n' + '\n')
