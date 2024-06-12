import numpy as np
import matplotlib.pyplot as plt
import os


def zip_img_by_svd(img, plotId, imgfile, imgfile_size, rate=0.8):
    zip_img = np.zeros(img.shape)

    zip_rate_singular = 0

    for chanel in range(3):
        u, sigma, v = np.linalg.svd(img[:, :, chanel])
        sigma_i = 0
        temp = 0

        while (temp / np.sum(sigma)) < rate:
            temp += sigma[sigma_i]
            sigma_i += 1

        zip_rate_singular += temp / np.sum(sigma)

        SigmaMat = np.zeros((sigma_i, sigma_i))
        for i in range(sigma_i):
            SigmaMat[i][i] = sigma[i]
        zip_img[:, :, chanel] = u[:, 0:sigma_i].dot(SigmaMat).dot(v[0:sigma_i, :])

    zip_rate_singular /= 3
    zip_rate_singular = 1 - zip_rate_singular

    for i in range(3):
        MAX = np.max(zip_img[:, :, i])
        MIN = np.min(zip_img[:, :, i])
        zip_img[:, :, i] = (zip_img[:, :, i] - MIN) / (MAX - MIN)

    zip_img = np.round(zip_img * 255).astype("uint8")
    plt.imsave("zip_" + imgfile + str(plotId - 1) + ".jpg", zip_img)
    size = os.path.getsize("zip_" + imgfile + str(plotId - 1) + ".jpg")

    size_rate = 1 - size / imgfile_size
    print(plotId - 1)
    print(str(round(zip_rate_singular * 100, 2)) + "% сжатия по сингулярным числам")
    print(str(round(size_rate * 100, 2)) + "% сжатия по памяти")
    print("Размер " + str(round(size / 1024)) + " КБ")
    print()

    f = plt.subplot(3, 3, plotId)
    plt.subplots_adjust(hspace=0.5, wspace=0.9)
    f.imshow(zip_img)
    f.axis('off')
    f.set_title(str(round(zip_rate_singular * 100, 2)) + "% сжатия по сингулярным числам\n" + str(round(size_rate * 100, 2)) + "% сжатия по памяти\nРазмер " + str(round(size / 1024)) + " КБ")


if __name__ == '__main__':
    imgfile = "img1.jpg"
    imgfile_size = os.path.getsize(imgfile)
    plt.figure(figsize=(12, 12))
    plt.rcParams['font.sans-serif'] = 'SimHei'
    img = plt.imread(imgfile)
    f1 = plt.subplot(331)
    plt.subplots_adjust(hspace=0.5, wspace=0.9)
    f1.imshow(img)
    f1.axis('off')
    f1.set_title("Исходное изображение\nРазмер " + str(round(imgfile_size / 1024)) + " КБ")
    for i in range(8):
        rate = (i + 1) / 10.0
        zip_img_by_svd(img, i + 2, imgfile[:4], imgfile_size, rate)
    plt.show()
