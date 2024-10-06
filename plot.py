import matplotlib.pyplot as plt
from matplotlib.image import imread

# 读取五张图片
image_paths = ["./imgs/I09/I09.BMP", "./imgs/I09/i09_19_1.bmp", "./imgs/I09/i09_19_2.bmp", "./imgs/I09/i09_19_3.bmp", "./imgs/I09/i09_19_4.bmp", "./imgs/I09/i09_19_5.bmp"]
images = [imread(path) for path in image_paths]

# 创建子图
fig, axs = plt.subplots(1, 6, figsize=(20, 5))

title = [
    "MOS\nHiFFTiq\nMFca-IQA",
    "5.5000\n5.1298\n5.3901",
    "4.6774\n4.7942\n4.7732",
    "4.5161\n4.4851\n4.4068",
    "4.0625\n3.4913\n4.0004",
    "3.4849\n3.9914\n3.3842",
         ]
# 绘制每张图片
for i in range(6):
    axs[i].imshow(images[i])
    axs[i].axis('off')
    axs[i].set_title(title[i], fontdict={'family' : 'Times New Roman', 'size'   : 18}, y=-0.4)

# 调整布局
plt.tight_layout()

# 保存图形到文件
plt.savefig('./imgs/VisualExperiment.png')  # 保存为PNG格式

# 显示图形
plt.show()
