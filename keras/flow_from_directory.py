import matplotlib.pyplot as  plt
import glob
from PIL import Image
from keras.preprocessing import image

path = 'flow_from_directory_train/'
gen_path = 'flow_from_directory_result/'




def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure()
    for i in range(9):
        img = Image.open(name_list[i])
        # add_subplot(331) 参数一：子图总行数，参数二：子图总列数，参数三：子图位置
        sub_img = fig.add_subplot(331 + i)
        sub_img.imshow(img)
    plt.show()
    return fig


# 打印图片列表
name_list = glob.glob(path + '*/*')
print(name_list)
# ['train\\00a366d4b4a9bbb6c8a63126697b7656.jpg', 'train\\00f34ac0a16ef43e6fd1de49a26081ce.jpg', 'train\\0a5f744c5077ad8f8d580081ba599ff5.jpg', 'train\\0a70f64352edfef4c82c22015f0e3a20.jpg', 'train\\0a783538d5f3aaf017b435ddf14cc5c2.jpg', 'train\\0a896d2b3af617df543787b571e439d8.jpg', 'train\\0abdda879bb143b19e3c480279541915.jpg', 'train\\0ac12f840df2b15d46622e244501a88c.jpg', 'train\\0b6c5bc46b7a0e29cddfa45b0b786d09.jpg']

# 打印图片
fig = print_result(path + '*/*')

# 保存图片
fig.savefig(gen_path + '/original_0.png', dpi=200, papertype='a5')

# 原图
datagen = image.ImageDataGenerator()
gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path,
                                       save_prefix='dog_gen', target_size=(224, 224))
for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '/*')
fig.savefig(gen_path + '/original_1.png', dpi=200, papertype='a5')