from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model, load_model
from cnn_utils import get_img_namelist, load_dataset, convert_to_one_hot
from keras.preprocessing import image
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
import scipy.misc
import h5py
K.set_image_data_format("channels_last")

def main():
    _, _, X_test_orig, Y_test_orig, classes = load_dataset()

    #X_test = preprocess_input(X_test_orig)
    X_test = X_test_orig / 255
    Y_test = convert_to_one_hot(Y_test_orig, len(classes)).T

    model = load_model("resnets_signs_model.h5")

    preds = model.evaluate(x=X_test, y=Y_test)
    print("Loss=" + str(preds[0]))
    print("Test Accuracy=" + str(preds[1]))

    img_names = get_img_namelist()
    
    for img_name in img_names:
        img_path = "images/" + img_name + ".jpg"
        img = image.load_img(img_path, target_size=(64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        x /= 255
        #print("Input image shape: ", x.shape)
        result = np.argmax(np.squeeze(model.predict(x)))
        print("prediction for image " + img_name + " is " + str(result))
        my_image = scipy.misc.imread(img_path)
        plt.imshow(my_image)
        plt.show()
       

if __name__ == '__main__':
    main()
    
