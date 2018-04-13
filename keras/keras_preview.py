from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.6,
        zoom_range=0.2,
       horizontal_flip=True,
#        fill_mode='constant')
       fill_mode='nearest')

img = load_img('E:/study/ML/tensorflow/cv/Object Detection/Mask_RCNN/kaggle/train/cat.3.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='E:/study/ML/tensorflow/cv/Object Detection/Mask_RCNN/kaggle/preview', save_prefix='cat', save_format='jpg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely


gen_data = datagen.flow_from_directory("E:/study/ML/tensorflow/cv/Object Detection/Mask_RCNN/kaggle/flow_from_directory", batch_size=2, shuffle=False, save_to_dir='E:/study/ML/tensorflow/cv/Object Detection/Mask_RCNN/kaggle/preview',
                                       save_prefix='dog_gen', target_size=(224, 224))

#gen_data = datagen.flow(x, batch_size=1,save_to_dir='E:/study/ML/tensorflow/cv/Object Detection/Mask_RCNN/kaggle/preview', save_prefix='cat', save_format='jpg')
for i in range(1):
    gen_data.next()