#=============================================================================================
#                                        Import Packages
#=============================================================================================
import os
import numpy as np
from PIL import Image, ImageOps
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
import random
import tensorflow as tf
from keras.initializers import glorot_uniform, he_normal
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, f1_score

from matplotlib import pyplot

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

#os.system("sudo pip install PIL")

#---------------------------------------Import Data-------------------------------------
import predict_whu369

#=============================================================================================
#                                          Load Data
#=============================================================================================
def get_images_paths(path):

    images_paths = []
    labels_paths = []

    for file in os.listdir(path):
        single_file_path = os.path.join(path, file)
        if(single_file_path.endswith("png")):
            images_paths.append(single_file_path)
            labels_paths.append(single_file_path.replace(".png", ".txt"))
    return images_paths, labels_paths

#=============================================================================================
#                                        Data Pre-processing
#=============================================================================================

# -----------------------------------------Label Targets------------------------------------
def get_target(labels_paths):
    labels = []

    for label_path in labels_paths:
        label_file = open(label_path)
        label = label_file.readline()
        label_file.close()
        labels.append(label)
    return labels

def label_encoding(t):
    label = {'red blood cell': 0, 'ring': 1, 'schizont': 2, 'trophozoite': 3}
    t = np.array([label[item] for item in t])
    return t

#-----------------------------------------Resize Inputs---------------------------------------

def get_input(images_paths):
    desired_size = 50
    imgs_array = []
    for image_path in images_paths:
        img = Image.open(image_path)
        old_size = img.size

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_img = Image.new('RGB', (desired_size, desired_size))
        new_img.paste(img, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

        new_img_array = np.asarray(new_img.convert('RGB'))

        imgs_array.append(new_img_array)

    return np.array(imgs_array)

# reference: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/

#------------------------------------Image Augmentation--------------------------------------
def targets_distribution(y):
    values, counts = np.unique(y, return_counts=True)
    values = list(values)
    counts = list(counts)
    rbc_num = counts[values.index(0)]
    r_num = counts[values.index(1)]
    s_num = counts[values.index(2)]
    t_num = counts[values.index(3)]
    return rbc_num, r_num, s_num, t_num

def sub_img_paths(targets, inputs):
    targets = list(targets)
    ring_imgs = []
    schizont_imgs = []
    trophozoite_imgs = []

    for index in range(0, len(targets)):
        t = targets[index]
        if t == 1:
            ring_imgs.append(inputs[index])
        if t == 2:
            schizont_imgs.append(inputs[index])
        if t == 3:
            trophozoite_imgs.append(inputs[index])
    return np.array(ring_imgs), np.array(schizont_imgs), np.array(trophozoite_imgs)

def img_aug(rbc_num, img_arr, current_img_num):
    aug_imgs = []
    for img in img_arr:

        # expand dimension to one sample
        samples = expand_dims(img, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=180, width_shift_range=0.2, height_shift_range=0.2)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)

        # Increase to approximately the same number of the red blood cell images
        n = (rbc_num // current_img_num - 1)
        for i in range(n):
            batch = it.next()
            #batch size: (1, 100, 100, 3)
            aug_img = np.squeeze(batch, axis=0)
            #aug_img size: (100, 100, 3)
            aug_imgs.append(aug_img)

    return np.array(aug_imgs)

# reference: https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
# reference: https://www.freecodecamp.org/news/classify-butterfly-images-deep-learning-keras/
# reference: https://keras.io/preprocessing/image/
# reference: https://towardsdatascience.com/image-augmentation-for-deep-learning-histogram-equalization-a71387f609b2

def display_aug_img(aug_imgs, a, b):
    selection = aug_imgs[a:b]
    n = b-a
    for i in range(n):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(selection[i])
    pyplot.show()

#-----------------------------------Generate A Balanced Dataset-----------------------------------
def balanced_dataset(original_x, original_y, aug_ring_img, aug_sch_img, aug_tro_img):
    args_x = (original_x, aug_ring_img, aug_sch_img, aug_tro_img)
    inputs = np.concatenate(args_x)

    r_num = aug_ring_img.shape[0]
    s_num = aug_sch_img.shape[0]
    t_num = aug_tro_img.shape[0]
    new_ring_label = np.array(r_num * [1])
    new_sch_label = np.array(s_num * [2])
    new_tro_label = np.array(t_num * [3])
    args_y = (original_y, new_ring_label, new_sch_label, new_tro_label)
    targets = np.concatenate(args_y)

    return inputs, targets

# reference: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/

#=============================================================================================
#                                         Modeling
#=============================================================================================

# Currently the model is tested using the original unbalanced dataset, when the private held-out test set is available
# These two variables (original_x and original_int_y) should be replaced with the actual x_test and y_test
# Therefore, given the constraint of the data
# The accuracy rate and mean score of Cohen_Kappa and F1_score printed from the following function are slightly skewed

#------------------------------------Set Up Cross Validation----------------------------------

def cross_validation(inputs, targets, original_x, original_int_y):

    inputs = inputs.reshape(len(inputs), -1)
    inputs = inputs / 255

    skf = StratifiedKFold(n_splits=10, shuffle=True)

    avg_acc = []

    for train_index, test_index in skf.split(inputs, targets):
        x_train, x_test, y_train, y_test = inputs[train_index], inputs[test_index], targets[train_index], targets[test_index]
        Y_train = to_categorical(y_train, num_classes=4)
        Y_test = to_categorical(y_test, num_classes=4)

        accuracy, cohen_kappa, f1_value = train_model3(x_train, x_test, Y_train, Y_test, original_x, original_int_y)
        avg_acc.append(accuracy)

    print("Mean Accuracy: " + str(np.mean(avg_acc)))
    print("Mean Cohen_Kappa: " + str(np.mean(cohen_kappa)))
    print("Mean F1_score: " + str(np.mean(f1_value)))

# reference: https://datascience.stackexchange.com/questions/11747/cross-validation-in-keras


#----------------------------------------Train Model-----------------------------------------------
# As mentioned above, the original_x and original_int_y variables should be replaced with the actual x_test and y_test
def train_model3(x_train, x_test, y_train, y_test, original_x, original_int_y):

    # Set-Up
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    weight_init = glorot_uniform(seed=SEED)

    # Hyper Parameters
    LR = 0.001
    N_NEURONS = (128, 32, 32)
    N_EPOCHS = 150  # Either 100 or 150 is a good option. This model has not been tested before during the 7-Day competition.
    BATCH_SIZE = 64
    DROPOUT = 0.1

    # Build Model
    model = Sequential([
        Dense(N_NEURONS[0], input_dim=7500, kernel_initializer=weight_init),
        Activation("relu"),
        Dropout(DROPOUT),
        BatchNormalization()
    ])
    # Loops over the hidden dims to add more layers
    for n_neurons in N_NEURONS[1:]:
        model.add(Dense(n_neurons, activation="relu", kernel_initializer=weight_init))
        model.add(Dropout(DROPOUT, seed=SEED))
        model.add(BatchNormalization())

    # Adds a final output layer with softmax to map to the 4 classes
    model.add(Dense(4, activation="softmax", kernel_initializer=weight_init))
    model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

    # Fit Model
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
              callbacks=[ModelCheckpoint("mlp_whu369.hdf5", monitor="val_loss", save_best_only=True)])

    # Evaluate Model
    original_y = to_categorical(original_int_y, num_classes=4)
    original_x = original_x.reshape(len(original_x), -1)
    original_x = original_x / 255
    accuracy = 100 * model.evaluate(original_x, original_y)[1]
    cohen_kappa = cohen_kappa_score(np.argmax(model.predict(original_x), axis=1), np.argmax(original_y, axis=1))
    f1_value = f1_score(np.argmax(model.predict(original_x), axis=1), np.argmax(original_y, axis=1), average='macro', labels=np.unique(y_train))

    print("Final accuracy on validations set:", accuracy, "%")
    print("Cohen Kappa", cohen_kappa)
    print("F1 score", f1_value)

    return accuracy, cohen_kappa, f1_value

#=============================================================================================
#                                   Test predict_whu369.py
#=============================================================================================

def perform_test():
    path = "./train_full/"
    x_test, y = get_images_paths(path)
    y_test_pred, *models = predict_whu369.predict(x_test)

    # %% -------------------------------------------------------------------------------------------------------------------
    assert isinstance(y_test_pred, type(np.array([1])))  # Checks if your returned y_test_pred is a NumPy array
    assert y_test_pred.shape == (len(x_test),)  # Checks if its shape is this one (one label per image path)
    # Checks whether the range of your predicted labels is correct
    assert np.unique(y_test_pred).max() <= 3 and np.unique(y_test_pred).min() >= 0
    print("test finished")


if __name__ == '__main__':
    #------------------------------Load Data-----------------------------------------
    path = "./train_full/"
    x, y = get_images_paths(path)
    #----------------------------Data Preprocessing----------------------------------
    # Enlabel Targets
    target = get_target(y)
    int_target = label_encoding(target)

    # Image Augmentation
    input = get_input(x)

    red_blood_cell_num, ring_num, sch_num, tro_num = targets_distribution(int_target)
    ring_img, schizont_img, trophozoite_img = sub_img_paths(int_target, input)

    ring_aug_img = img_aug(red_blood_cell_num, ring_img, ring_num)
    sch_aug_img = img_aug(red_blood_cell_num, schizont_img, sch_num)
    tro_aug_img = img_aug(red_blood_cell_num, trophozoite_img, tro_num)

    #display_aug_img(ring_aug_img, 20, 29)

    #------------------Generate Balanced Dataset Using Augmented Images----------------
    new_inputs, new_targets = balanced_dataset(input, int_target, ring_aug_img, sch_aug_img, tro_aug_img)

    #----------------------------------Modeling----------------------------------------
    cross_validation(new_inputs, new_targets, input, int_target)

    #-------------------------------Perform Test---------------------------------------
    perform_test()



