import io
import random
import time

import PIL
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

import torchvision.transforms as T
from matplotlib.testing.jpl_units import rad
from scipy import ndimage
from scipy.interpolate import UnivariateSpline, interpolate
from scipy.ndimage import filters
from skimage import color
from sklearn.preprocessing import normalize


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# gauss模糊
def imblurgauss(im, level):
    # Takes in PIL Image and returns Gaussian Blurred PIL Image
    levels = [0.1, 0.5, 1, 2, 5, 7]
    sigma = levels[level]
    im_dist = im.filter(ImageFilter.GaussianBlur(radius=sigma))
    return im_dist


# 镜头模糊(报警告)
def imblurlens(im, level):
    # Takes PIL Image and returns lens blurred image

    # MATLAB version https://github.com/alexandrovteam/IMS_quality/blob/master/codebase/fspecialIM.m
    levels = [1, 2, 4, 6, 8]
    radius = levels[level]

    im = np.array(im)
    crad = int(np.ceil(radius - 0.5))
    [x, y] = np.meshgrid(np.arange(-crad, crad + 1, 1), np.arange(-crad, crad + 1, 1), indexing='xy')
    maxxy = np.maximum(abs(x), abs(y))
    minxy = np.minimum(abs(x), abs(y))
    m1 = np.multiply((radius ** 2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2), (minxy - 0.5)) + np.nan_to_num(
        np.multiply((radius ** 2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2),
                    np.sqrt(radius ** 2 - (maxxy + 0.5) ** 2)), nan=0)
    m2 = np.multiply((radius ** 2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2), (minxy + 0.5)) + np.nan_to_num(
        np.multiply((radius ** 2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2),
                    np.sqrt(radius ** 2 - (maxxy - 0.5) ** 2)), nan=0)
    sgrid = np.multiply((radius ** 2 * (0.5 * (np.arcsin(m2 / radius) - np.arcsin(m1 / radius)) + 0.25 * (
            np.sin(2 * np.arcsin(m2 / radius)) - np.sin(2 * np.arcsin(m1 / radius)))) - np.multiply((maxxy - 0.5),
                                                                                                    (m2 - m1)) + (
                                 m1 - minxy + 0.5)), ((((radius ** 2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2) & (
            radius ** 2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)) | ((minxy == 0) & (maxxy - 0.5 < radius) & (
            maxxy + 0.5 >= radius)))))
    sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < radius ** 2)
    sgrid[crad, crad] = min(np.pi * radius ** 2, np.pi / 2)
    if ((crad > 0) and (radius > crad - 0.5) and (radius ** 2 < (crad - 0.5) ** 2 + 0.25)):
        m1 = np.sqrt(rad ** 2 - (crad - 0.5) ** 2)
        m1n = m1 / radius
        sg0 = 2 * (radius ** 2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))
        sgrid[2 * crad + 1, crad + 1] = sg0
        sgrid[crad + 1, 2 * crad + 1] = sg0
        sgrid[crad + 1, 1] = sg0
        sgrid[1, crad + 1] = sg0
        sgrid[2 * crad, crad + 1] = sgrid[2 * crad, crad + 1] - sg0
        sgrid[crad + 1, 2 * crad] = sgrid[crad + 1, 2 * crad] - sg0
        sgrid[crad + 1, 2] = sgrid[crad + 1, 2] - sg0
        sgrid[2, crad + 1] = sgrid[2, crad + 1] - sg0
    sgrid[crad, crad] = min(sgrid[crad, crad], 1)
    h = sgrid / sgrid.sum()
    ndimage.convolve(im[:, :, 0], h, output=im[:, :, 0], mode='nearest')
    ndimage.convolve(im[:, :, 1], h, output=im[:, :, 1], mode='nearest')
    ndimage.convolve(im[:, :, 2], h, output=im[:, :, 2], mode='nearest')
    im = Image.fromarray(im)
    return im


# 运动模糊
def imblurmotion(im, level):
    levels = [12, 16, 20, 24, 28, 32]

    kernel_size = levels[level]
    phi = random.choice([0, 90]) / (180) * np.pi
    kernel = np.zeros((kernel_size, kernel_size))

    im = np.array(im)

    if phi == 0:
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    else:
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    kernel /= kernel_size

    ndimage.convolve(im[:, :, 0], kernel, output=im[:, :, 0], mode='nearest')
    ndimage.convolve(im[:, :, 1], kernel, output=im[:, :, 1], mode='nearest')
    ndimage.convolve(im[:, :, 2], kernel, output=im[:, :, 2], mode='nearest')

    im = Image.fromarray(im)

    return im


# 颜色扩散
def imcolordiffuse(im, level):
    levels = [1, 3, 6, 8, 12]

    amount = levels[level]
    im = np.array(im)

    sigma = 1.5 * amount + 2
    scaling = amount

    lab = color.rgb2lab(im)
    l = lab[:, :, 0]

    lab = filters.gaussian(lab, sigma=sigma, channel_axis=-1) * scaling
    lab[:, :, 0] = l
    im = 255 * color.lab2rgb(lab)
    im = np.clip(im, 0, 255)
    im = Image.fromarray(np.uint8(im))
    return im


# 归一化方法将数组的值线性映射到指定范围
def mapmm(e):
    mina = 0.0
    maxa = 1.0
    minx = np.min(e)
    maxx = np.max(e)
    if minx < maxx:
        e = (e - minx) / (maxx - minx) * (maxa - mina) + mina
    return e


# 颜色偏移:对图像的某个颜色通道进行随机位移，并根据图像的梯度进行加权，以在梯度较大的区域更强烈地表现颜色变化
def imcolorshift(im, level):
    levels = [1, 3, 6, 8, 12]
    amount = levels[level]

    im = np.float32(np.array(im) / 255.0)
    # RGB to Gray
    x = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]

    dx = np.gradient(x, axis=0)
    dy = np.gradient(x, axis=1)
    e = np.hypot(dx, dy)  # magnitude

    e = filters.gaussian(e, sigma=4)

    e = mapmm(e)
    e = np.clip(e, 0.1, 1)
    e = mapmm(e)

    percdev = [1, 1]

    valuehi = np.percentile(e, 100 - percdev[1])
    valuelo = 1 - np.percentile(1 - e, 100 - percdev[0])

    e = np.clip(e, valuelo, valuehi)
    e = mapmm(e)

    channel = 1
    g = im[:, :, channel]
    amt_shift = np.uint8(np.round((normalize(np.random.random([1, 2]), norm='l2', axis=1) * amount)))

    padding = np.multiply(int(np.max(amt_shift)), [1, 1])

    y = np.pad(g, padding, 'symmetric')
    y = np.roll(y, amt_shift.reshape(-1))

    sl = padding[0]

    g = y[sl:-sl, sl:-sl]

    J = im
    J[:, :, channel] = np.multiply(g, e) + np.multiply(J[:, :, channel], (1 - e))
    J = J * 255.0

    im = Image.fromarray(np.uint8(J))
    return im


# 颜色饱和度调整处理
def imcolorsaturate(im, level):
    levels = [0.4, 0.2, 0.1, 0, -0.4]
    amount = levels[level]

    im = np.array(im)
    hsvIm = color.rgb2hsv(im)
    hsvIm[:, :, 1] = hsvIm[:, :, 1] * amount
    im = color.hsv2rgb(hsvIm) * 255.0
    im = np.clip(im, 0, 255)
    im = Image.fromarray(np.uint8(im))

    return im


# 调整Lab色彩空间中的a和b通道来实现色彩饱和度的调整
def imsaturate(im, level):
    levels = [1, 2, 3, 6, 9, 12]
    amount = levels[level]

    lab = color.rgb2lab(im)
    lab[:, :, 1:] = lab[:, :, 1:] * amount
    im = color.lab2rgb(lab) * 255.0
    im = np.clip(im, 0, 255)
    im = Image.fromarray(np.uint8(im))

    return im


# jpeg压缩
def imcompressjpeg(im, level):
    levels = [70, 43, 36, 24, 4]
    amount = levels[level]

    imgByteArr = io.BytesIO()
    im.save(imgByteArr, format='JPEG', quality=amount)
    im1 = Image.open(imgByteArr)

    return im1


# gauss噪声
def imnoisegauss(im, level):
    levels = [0.001, 0.002, 0.003, 0.05, 0.1, 0.25]

    im = np.float32(np.array(im) / 255.0)

    row, col, ch = im.shape

    var = levels[level]
    mean = 0
    sigma = var ** 0.5
    gauss = np.array(im.shape)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = im + gauss
    noisy = noisy * 255.0
    noisy = np.clip(noisy, 0, 255)

    return Image.fromarray(noisy.astype('uint8'))


# 噪声调色噪声处理
def imnoisecolormap(im, level):
    levels = [0.0001, 0.0005, 0.001, 0.008, 0.05]
    var = levels[level]
    mean = 0

    im = np.array(im)
    ycbcr = color.rgb2ycbcr(im)
    ycbcr = ycbcr / 255.0

    row, col, ch = ycbcr.shape
    sigma = var ** 0.5
    gauss = np.array(ycbcr.shape)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = ycbcr + gauss

    im_dist = color.ycbcr2rgb(noisy * 255.0) * 255.0
    im_dist = np.clip(im_dist, 0, 255)
    return Image.fromarray(im_dist.astype('uint8'))


# 脉冲噪声处理:将一部分像素置为黑色，将一部分像素置为白色
def imnoiseimpulse(im, level):
    levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.15]
    prob = levels[level]

    im = np.array(im)
    output = im

    black = np.array([0, 0, 0], dtype='uint8')
    white = np.array([255, 255, 255], dtype='uint8')

    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white

    return Image.fromarray(output.astype('uint8'))


# 乘法噪声，将像素值按照一定的方差进行缩放
def imnoisemultiplicative(im, level):
    levels = [0.001, 0.005, 0.01, 0.08, 0.15]

    im = np.float32(np.array(im) / 255.0)

    row, col, ch = im.shape

    var = levels[level]
    mean = 0
    sigma = var ** 0.5
    gauss = np.array(im.shape)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = im + im * gauss
    noisy = noisy * 255.0
    noisy = np.clip(noisy, 0, 255)

    return Image.fromarray(noisy.astype('uint8'))


# 去噪处理：在图像中引入高斯噪声，然后随机选择一个去噪滤波器类型（高斯模糊或方框模糊）来去除噪声
def imdenoise(im, level):
    levels = [0.001, 0.002, 0.003, 0.005, 0.01]

    im = np.float32(np.array(im) / 255.0)

    row, col, ch = im.shape

    var = levels[level]
    mean = 0
    sigma = var ** 0.5
    gauss = np.array(im.shape)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = im + gauss
    noisy = noisy * 255.0
    noisy = np.clip(noisy, 0, 255)

    im = Image.fromarray(noisy.astype('uint8'))

    filt_type = np.random.randint(0, 2)
    if filt_type == 0:
        denoised = im.filter(ImageFilter.GaussianBlur(radius=3))
    elif filt_type == 1:
        denoised = im.filter(ImageFilter.BoxBlur(radius=2))

    return denoised


# 亮度增强处理
def imbrighten(im, level):
    levels = [0.1, 0.2, 0.4, 0.7, 1.1]

    amount = levels[level]
    im = np.float32(np.array(im) / 255.0)

    lab = color.rgb2lab(im)
    L = lab[:, :, 0] / 100.0
    L_ = curvefit(L, 0.5 + 0.5 * amount)
    lab[:, :, 0] = L_ * 100.0

    J = curvefit(im, 0.5 + 0.5 * amount)

    J = (2 * J + np.clip(color.lab2rgb(lab), 0, 1)) / 3.0
    J = np.clip(J * 255.0, 0, 255)

    return Image.fromarray(J.astype('uint8'))


# 样条插值
def curvefit(xx, coef):
    x = np.array([0, 0.5, 1])
    y = np.array([0, coef, 1])

    tck = UnivariateSpline(x, y, k=2)
    return np.clip(tck(xx), 0, 1)


# 亮度减弱函数
def imdarken(im, level):
    levels = [0.05, 0.1, 0.2, 0.4, 0.8]
    param = levels[level]
    # convert 0-1
    im = np.array(im).astype(np.float32) / 255.0

    # generate curve to fit based on amount
    x = [0, 0.5, 1]
    y = [0, 0.5, 1]
    y[1] = 0.5 - param / 2

    # generate interpolating function and interpolate input
    cs = UnivariateSpline(x, y, k=2)
    yy = cs(im)

    # convert back to PIL image
    im_out = np.clip(yy, 0, 1)
    im_out = (im_out * 255).astype(np.uint8)
    im_out = Image.fromarray(im_out)
    return im_out


# 对图像进行平移
def immeanshift(im, level):
    levels = [0.15, 0.08, 0, -0.08, -0.15, -0.5]
    amount = levels[level]

    im = np.float32(np.array(im) / 255.0)

    im = im + amount
    im = im * 255.0

    im = np.clip(im, 0, 255)
    return Image.fromarray(im.astype('uint8'))


# 对图像进行缩放和再放大来引入失真
def imresizedist(im, level):
    levels = [2, 3, 4, 8, 16, 24]
    amount = levels[level]

    size = im.size

    w = size[0]
    h = size[1]

    scaled_w = int(w / amount)
    scaled_h = int(h / amount)

    resized_image = im.resize((scaled_w, scaled_h))

    im = resized_image.resize((w, h))
    return im


# 双线性插值的图像缩放函数
def imresizedist_bilinear(im, level):
    levels = [2, 3, 4, 8, 16, 24]
    amount = levels[level]

    size = im.size

    w = size[0]
    h = size[1]

    scaled_w = int(w / amount)
    scaled_h = int(h / amount)

    resized_image = im.resize((scaled_w, scaled_h), Image.BILINEAR)

    im = resized_image.resize((w, h), Image.BILINEAR)
    return im


# 使用最近邻插值的图像缩放函数
def imresizedist_nearest(im, level):
    levels = [2, 3, 4, 5, 6, 8]
    amount = levels[level]

    size = im.size

    w = size[0]
    h = size[1]

    scaled_w = int(w / amount)
    scaled_h = int(h / amount)

    resized_image = im.resize((scaled_w, scaled_h), Image.NEAREST)

    im = resized_image.resize((w, h), Image.NEAREST)
    return im


# 使用Lanczos插值的图像缩放函数
def imresizedist_lanczos(im, level):
    levels = [2, 3, 4, 8, 16]
    amount = levels[level]

    size = im.size

    w = size[0]
    h = size[1]

    scaled_w = int(w / amount)
    scaled_h = int(h / amount)

    resized_image = im.resize((scaled_w, scaled_h), Image.LANCZOS)

    im = resized_image.resize((w, h), Image.LANCZOS)
    return im


# 在L通道上应用高通滤波器来增强图像的细节
def imsharpenHi(im, level):
    levels = [1, 2, 3, 6, 12]
    param = levels[level]
    ## param range to be use -> double from matlab
    ## convert PIL-RGB to LAB for operation in L space
    im = np.array(im).astype(np.float32)
    LAB = color.rgb2lab(im)
    im_L = LAB[:, :, 0]

    ## compute laplacians
    gy = np.gradient(im_L, axis=0)
    gx = np.gradient(im_L, axis=1)
    ggy = np.gradient(gy, axis=0)
    ggx = np.gradient(gx, axis=1)
    laplacian = ggx + ggy

    ## subtract blurred version from image to sharpen
    im_out = im_L - param * laplacian

    ## clip L space in 0-100
    im_out = np.clip(im_out, 0, 100)

    ## convert LAB to PIL-RGB
    LAB[:, :, 0] = im_out
    im_out = 255 * color.lab2rgb(LAB)
    im_out = im_out.astype(np.uint8)
    im_out = Image.fromarray(im_out)
    return im_out


# 对比度调整函数，通过曲线拟合对比度
def imcontrastc(im, level):
    levels = [0.3, 0.15, 0, -0.4, -0.6]
    param = levels[level]
    ## convert 0-1
    im = np.array(im).astype(np.float32) / 255.0

    ## generate curve to fit based on amount->param
    coef = [[0.3, 0.5, 0.7], [0.25 - param / 4, 0.5, 0.75 + param / 4]]
    defa = 0
    x = [0, 0, 0, 0, 1]
    x[1:-1] = coef[0]
    y = [0, 0, 0, 0, 1]
    y[1:-1] = coef[1]

    ## generate interpolating function and interpolate input
    cs = UnivariateSpline(x, y)
    yy = cs(im)

    ## convert back to PIL image
    im_out = np.clip(yy, 0, 1)
    im_out = (im_out * 255).astype(np.uint8)
    im_out = Image.fromarray(im_out)
    return im_out


# 彩色块添加函数，随机在图像中添加彩色块
def imcolorblock(im, level):
    levels = [2, 4, 6, 8, 10]
    param = levels[level]
    ## convert 0-1
    im = np.array(im).astype(np.float32)
    h, w, _ = im.shape

    ## define patchsize
    patch_size = [32, 32]

    h_max = h - patch_size[0]
    w_max = w - patch_size[1]

    block = np.ones((patch_size[0], patch_size[1], 3))

    ## place the color blocks at random
    for i in range(0, param):
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        x = int(random.uniform(0, 1) * w_max)
        y = int(random.uniform(0, 1) * h_max)
        im[y:y + patch_size[0], x:x + patch_size[1], :] = color * block

    ## convert back to PIL image
    im_out = (im).astype(np.uint8)
    im_out = Image.fromarray(im_out)
    return im_out


# 通过调整像素大小引入图像像素化效果
def impixelate(im, level):
    levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    param = levels[level]
    z = 0.95 - param ** (0.6)
    size_z_1 = int(z * im.width)
    size_z_2 = int(z * im.height)

    im_out = im.resize((size_z_1, size_z_2), resample=PIL.Image.NEAREST)
    im_out = im_out.resize((im.width, im.height), resample=PIL.Image.NEAREST)
    return im_out


# 随机在图像中添加非同心圆
def imnoneccentricity(im, level):
    levels = [20, 40, 60, 80, 100]
    param = levels[level]
    ## convert 0-1
    im = np.array(im).astype(np.float32)
    h, w, _ = im.shape

    ## define patchsize
    patch_size = [16, 16]

    radius = 16

    h_min = radius
    w_min = radius

    h_max = h - patch_size[0] - radius
    w_max = w - patch_size[1] - radius

    block = np.ones((patch_size[0], patch_size[1], 3))

    ## place the color blocks at random
    for i in range(0, param):
        w_start = int(random.uniform(0, 1) * (w_max - w_min)) + w_min
        h_start = int(random.uniform(0, 1) * (h_max - h_min)) + h_min
        patch = im[h_start:h_start + patch_size[0], w_start:w_start + patch_size[1], :]

        rand_w_start = int((random.uniform(0, 1) - 0.5) * radius + w_start)
        rand_h_start = int((random.uniform(0, 1) - 0.5) * radius + h_start)
        im[rand_h_start:rand_h_start + patch_size[0], rand_w_start:rand_w_start + patch_size[1], :] = patch

    ## convert back to PIL image
    im_out = (im).astype(np.uint8)
    im_out = Image.fromarray(im_out)
    return im_out


# 对图像应用位移引入图像失真
def imwarpmap(im, shifts):
    sy, sx = shifts[:, :, 0], shifts[:, :, 1]
    # create mesh-grid for image shape
    [xx, yy] = np.meshgrid(range(0, shifts.shape[1]), range(0, shifts.shape[0]))

    # check whether grey image or RGB
    shape = im.shape
    im_out = im
    if len(shape) > 2:
        ch = shape[-1]
    else:
        ch = 1

    # iterate function over each channel
    for i in range(ch):
        im_out[:, :, i] = ndimage.map_coordinates(im[:, :, i], [(yy - sy).ravel(), (xx - sx).ravel()], order=3,
                                                  mode='reflect').reshape(im.shape[:2])

    # clip image between 0-255
    im_out = np.clip(im_out, 0, 255)

    return im_out


# 图像抖动：多次应用随机位移引入 1.2787 秒
def imjitter(im, level):
    levels = [0.05, 0.1, 0.2, 0.5, 1]
    param = levels[level]
    ## convert 0-1
    im = np.array(im).astype(np.float32)
    h, w, _ = im.shape

    sz = [h, w, 2]

    ## iterate image-warp over for 5 times
    J = im
    for i in range(0, 5):
        ## generate random shift map
        shifts = np.random.randn(h, w, 2) * param
        J = imwarpmap(J, shifts)

    ## convert back to PIL image
    im_out = (J).astype(np.uint8)
    im_out = Image.fromarray(im_out)
    return im_out


# TODO 更换失真策略
# 固定叠加22种最大等级失真
def apply_classical_distortions1(image):
    # 定义失真函数列表
    distortions = [
        imblurgauss,
        imblurlens,
        imblurmotion,
        imcolorsaturate,
        imsaturate,
        imcompressjpeg,
        imnoisegauss,
        imnoisecolormap,
        imnoiseimpulse,
        imnoisemultiplicative,
        imdenoise,
        imbrighten,
        imdarken,
        immeanshift,
        imresizedist,
        imresizedist_bilinear,
        imresizedist_nearest,
        imresizedist_lanczos,
        imcontrastc,
        impixelate,
        imnoneccentricity,
        imjitter
    ]

    # 依次应用选定的失真函数
    distorted_image = image
    i = 0
    for distortion in distortions:
        i = i + 1
        # 随机选择失真的强度或级别
        # distortion_level = random.randint(3, 4)
        distortion_level = 4
        # print('使用了', distortion.__name__, distortion_level)
        # t1 = time.time()
        distorted_image = distortion(distorted_image, distortion_level)
        # t2 = time.time()
        # run_time = t2 - t1
        # print(f"{i}:使用{distortion.__name__}运行时间为 {run_time:.4f} 秒")

    return distorted_image

# 固定叠加10种最大等级失真
def apply_classical_distortions(image):
    # 定义失真函数列表
    distortions = [
        imblurgauss,
        imblurmotion,
        imcompressjpeg,
        # imsaturate,

        imnoisegauss,
        imnoiseimpulse,

        imresizedist,
        imresizedist_bilinear,
        imresizedist_nearest,
        impixelate
    ]
    # random.shuffle(distortions)  # 随机打乱失真函数顺序
    # 依次应用选定的失真函数
    distorted_image = image
    # i = 0
    for distortion in distortions:
        # i = i + 1
        # 随机选择失真的强度或级别
        # distortion_level = random.randint(3, 4)
        distortion_level = 3  # 使用最大失真等级
        # print('使用了', distortion.__name__, distortion_level)
        # t1 = time.time()
        distorted_image = distortion(distorted_image, distortion_level)
        # t2 = time.time()
        # run_time = t2 - t1
        # print(f"{i}:使用{distortion.__name__}运行时间为 {run_time:.4f} 秒")

    return distorted_image


# 随机叠加几种失真
def apply_random_distortions(image):
    # 定义失真函数列表
    distortions = [
        imblurgauss,
        imblurlens,
        imblurmotion,
        # imcolordiffuse,   # error
        # imcolorshift,  # error
        imcolorsaturate,
        imsaturate,
        imcompressjpeg,
        imnoisegauss,
        imnoisecolormap,
        imnoiseimpulse,
        imnoisemultiplicative,
        imdenoise,
        imbrighten,
        imdarken,
        immeanshift,
        imresizedist,
        imresizedist_bilinear,
        imresizedist_nearest,
        imresizedist_lanczos,
        # imsharpenHi,
        imcontrastc,
        # imcolorblock,
        impixelate,
        imnoneccentricity,
        # imwarpmap,  # error
        imjitter
    ]

    # 从失真函数列表中随机选择5个函数
    selected_distortions = random.sample(distortions, 5)
    # 依次应用选定的失真函数
    distorted_image = image
    for distortion in selected_distortions:
        # 随机选择失真的强度或级别
        # distortion_level = random.randint(0, 4)
        distortion_level = 4
        # print('使用了', distortion.__name__, distortion_level)
        distorted_image = distortion(distorted_image, distortion_level)

    return distorted_image

# # 创建一个图像合成器
# fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
# # LQ = pil_loader('/home/dataset/LIVE/gblur/img1.bmp')
# # LQ = pil_loader('/home/dataset/LIVE/jpeg/img1.bmp')
# # LQ = pil_loader('/home/dataset/LIVE/fastfading/img10.bmp')
# # LQ = pil_loader('/home/dataset/LIVE/jp2k/img1.bmp')
# # LQ = pil_loader('/home/dataset/LIVE/refimgs/bikes.bmp')
# LQ = pil_loader('/data/user/cwz/DRIQA/框图/I13_01_02.png')
# # LQ2 = pil_loader('../imgs/i09_21_2.bmp')
# # LQ3 = pil_loader('../imgs/i09_21_3.bmp')
# # LQ4 = pil_loader('../imgs/i09_21_4.bmp')
# # LQ5 = pil_loader('../imgs/i09_21_5.bmp')
# # LQ.show()
# # 在第一个子图中显示多个图像
# # axes[0].imshow(LQ)
# # axes[0].imshow(LQ2)
# # axes[0].imshow(LQ3)
# # axes[0].imshow(LQ4)
# # axes[0].imshow(LQ5)
# axes[0].axis('off')
# t1 = time.time()
# ZLQ = apply_classical_distortions(LQ)
# # ZLQ2 = apply_classical_distortions(LQ2)
# # ZLQ3 = apply_classical_distortions(LQ3)
# # ZLQ4 = apply_classical_distortions(LQ4)
# # ZLQ5 = apply_classical_distortions(LQ5)
# ZLQ.save('/data/user/cwz/DRIQA/框图/I13_addDist.png')
# # ZLQ2.save('../imgs/i09_21_2_dis.bmp')
# # ZLQ3.save('../imgs/i09_21_3_dis.bmp')
# # ZLQ4.save('../imgs/i09_21_4_dis.bmp')
# # ZLQ5.save('../imgs/i09_21_5_dis.bmp')
# t2 = time.time()
# run_time = t2 - t1
# print(f"共计为 {run_time:.4f} 秒")
# axes[1].imshow(ZLQ)
# # axes[1].imshow(ZLQ2)
# # axes[1].imshow(ZLQ3)
# # axes[1].imshow(ZLQ4)
# # axes[1].imshow(ZLQ5)
# axes[1].axis('off')
# plt.show()
