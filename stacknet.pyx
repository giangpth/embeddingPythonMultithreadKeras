'''VGGFace model for Keras.
# Reference:
- [Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
'''
# utils
from __future__ import print_function
from __future__ import division
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from keras import backend as K
from keras.utils.data_utils import get_file

""" functions are mostly taken and modified from keras/applications (https://github.com/fchollet/keras/tree/master/keras/applications) """

WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_v2.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_v2.h5'
LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels.npy'

LABELS = None


def preprocess_input(x, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        x = x[:, ::-1, ...]
        x[:, 0, :, :] -= 93.5940
        x[:, 1, :, :] -= 104.7624
        x[:, 2, :, :] -= 129.1863
    else:
        x = x[..., ::-1]
        x[..., 0] -= 93.5940
        x[..., 1] -= 104.7624
        x[..., 2] -= 129.1863
    return x


def decode_predictions(preds, top=5):
    global LABELS
    if len(preds.shape) != 2 or preds.shape[1] != 2622:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 2622)). '
                         'Found array with shape: ' + str(preds.shape))
    if LABELS is None:
        fpath = get_file('rcmalli_vggface_labels.json',
                         LABELS_PATH,
                         cache_subdir='models')
        LABELS = np.load(fpath)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [[str(LABELS[i]), pred[i]] for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results
#############################################################################
def extendFeature(filename, feature, limit):
    npyfile = filename + ".npy"
    if not os.path.isfile(npyfile) or feature.shape[0] >= limit:
        np.save(filename, feature)
    else:
        oldarr = np.load(npyfile)
        # print(oldarr)
        if oldarr.shape[0] > limit:
            n = feature.shape[0]
            oldarr = oldarr[n:, :]
            # print (oldarr)
            oldarr = np.append(feature, oldarr, axis=0)
        else:
            oldarr = np.append(feature, oldarr, axis=0)
        # print(oldarr)
        np.save(filename, oldarr)


########################################
def moveLastChangedFile(foldername, newfilename):
    lastchange = 0
    for filename in os.listdir(foldername):
        if filename.endswith('.h5') or filename.endswith('hdf5'):
            # print(os.path.join(foldername, filename))
            statbuf = os.stat(os.path.join(foldername, filename))
            # print("Modification time: {}".format(statbuf.st_mtime))
            if statbuf.st_mtime > lastchange:
                lastchange = statbuf.st_mtime
                lastfile = os.path.join(foldername, filename)

    print(lastfile)
    os.rename(lastfile, newfilename)
# VGG for face recognition


import warnings

from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.models import Model
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
import PIL
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import clone_model
from keras.models import load_model
import os
import tensorflow as tf
import keras
from keras.preprocessing import image
from sklearn import svm
import pickle
import shutil
import h5py

def VGGFace(include_top=True, weights='vggface',
            input_tensor=None, input_shape=None,
            pooling=None,
            classes=2622):
    """Instantiates the VGGFace architecture.
    Optionally loads weights pre-trained
    on VGGFace dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if weights not in {'vggface', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `vggface` '
                         '(pre-training on VGGFace Dataset).')

    if weights == 'vggface' and include_top and classes != 2622:
        raise ValueError('If using `weights` as vggface original with `include_top`'
                         ' as true, `classes` should be 2622')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, name='fc6')(x)
        x = Activation('relu', name='fc6/relu')(x)
        x = Dense(4096, name='fc7')(x)
        x = Activation('relu', name='fc7/relu')(x)
        x = Dense(2622, name='fc8')(x)
        x = Activation('softmax', name='fc8/softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

            # Ensure that the model takes into account
            # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        # Create model.
    model = Model(inputs, x, name='VGGFace')  # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('/home/workstation/Workspace/FaceID_v2/model/rcmalli_vggface_tf_v2.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
#            weights_path = get_file('rcmalli_vggface_tf_notop_v2.h5',
#                                   utils.WEIGHTS_PATH_NO_TOP,
#                                    cache_subdir='models')
            weights_path = '/home/workstation/Workspace/FaceID_v2/model/rcmalli_vggface_tf_notop_v2.h5'
        model.load_weights(weights_path, by_name=True)
        # if K.backend() == 'theano':
        #     layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='pool5')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc6')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

# VGG finetune


class FaceRecognizer():
    MAX_SAVE_IMAGE = 2000
    # khoi tao, dau vao:
    # nb_class: so luong class dua vao finetune
    # width, height: kich co rong cao cua anh
    def __init__(self, nb_class, width, height):
        self.nb_class = nb_class
        self.img_w = width
        self.img_h = height
        self.vgg_base_model = VGGFace(include_top=False, input_shape=(self.img_h, self.img_w, 3))
        self.datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.25,
                                          height_shift_range=0.25, shear_range=0.15, horizontal_flip=True)
        print("create FaceRecognizer instance")

    # fine tune pretrained model
    # train_dir: duong dan den folder chua anh train, chia subfolder con theo class
    # batch_size: batch size, set so phu hop voi kich co ram
    def finetune(self, train_dir, epoch, batch_size):
        # chekch existance of folder ./model/finetune
        if not os.path.isdir('/home/workstation/Workspace/FaceID_v2/model/finetune'):
            os.mkdir('/home/workstation/Workspace/FaceID_v2/model/finetune')
            print('create finetune folder')
        # freeze some layers dont want to be trained
        for layer in self.vgg_base_model.layers:
            layer.trainable = False
        self.vgg_base_model.summary()
        vgg_base_out = self.vgg_base_model.output
        # add fully connected layer
        x = Flatten(name='flatten')(vgg_base_out)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dense(self.nb_class, activation='softmax', name='fc3')(x)
        self.model_finetune = Model(self.vgg_base_model.input, x)
        self.model_finetune.summary()
        self.model_finetune.compile(optimizer='adam', metrics=['accuracy'],
                                    loss='mean_squared_error')

        # prepare data to finetune
        train_data = self.datagen.flow_from_directory(train_dir, target_size=(self.img_w, self.img_h),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True)
        file_path = '/home/workstation/Workspace/FaceID_v2/model/finetune/Face_weight-{epoch:02d}-{acc:.2f}.h5'
        checkpoint = ModelCheckpoint(filepath=file_path, monitor='acc', verbose=1,
                                     save_best_only=True, mode='max')
        callback_list=[checkpoint]
        self.model_finetune.fit_generator(train_data, steps_per_epoch=train_data.samples/batch_size,
                                          epochs=epoch, callbacks=callback_list)
#        self.model_finetune.save('./model/face_finetune_model.h5')
#        self.vgg_base_model.save('./model/face_vgg_base_model.h5')


        print('finetune file to use')
        # print(file_path)
        moveLastChangedFile('/home/workstation/Workspace/FaceID_v2/model/finetune', '/home/workstation/Workspace/FaceID_v2/model/final_finetune.h5')

#        self.model_finetune.load_weights('./model/final_finetune.h5')
#         self.model_finetune.load_weights('./model/finetune/Face_weight-17-1.00.h5')
        self.model_finetune.save('/home/workstation/Workspace/FaceID_v2/model/model/face_finetune_model.h5')

        shutil.rmtree('/home/workstation/Workspace/FaceID_v2/model/finetune')
        print("finetune done")


    # load model
    def load_finetune_models(self):
        self.model_finetune = load_model(filepath='/home/workstation/Workspace/FaceID_v2/model/face_finetune_model.h5')
        self.extractor = Model(self.model_finetune.input, self.model_finetune.get_layer('fc2').output)


    # extract feature to svm classify
    # img_folder: duong dan den folder chua subfolder anh, moi subfolder chua 250 anh, 200 anh cho train, 50 anh cho val
    # feature_folder: duong dan den feature folder, trong folder feature can co 2 folder, 1 folder ten train, 1 folder ten val
    # filename: ten doi tuong, cung la ten cua subfolder trong img_folder va extracted folder
    # extracted_folder: anh trong folder img_folder sau khi extract feature xong duoc chuyen qua folder extracted image
    def extract_feature(self, filename, img_folder, extracted_folder, val_split=0.2):
        print("Extract feature")
        feature_folder = '/home/workstation/Workspace/FaceID_v2/data/features'
        if not os.path.isdir(feature_folder):
            print('make new feature folder')
            print(feature_folder)
            os.makedirs(feature_folder)
        data = []
        val = []
        counter = 0

        for root, dirs, files in os.walk(img_folder):
            for subdir in dirs:
                if subdir == filename:
                    numimg = len([name for name in os.listdir(os.path.join(img_folder, subdir)) if os.path.isfile(os.path.join(img_folder, subdir, name))])
                    print(numimg)
                    numtrain = numimg*(1-val_split)
                    print('numtrain')
                    print(numtrain)
                    for imgname in os.listdir(os.path.join(img_folder, subdir)):
                        # print((os.path.join(root, subdir)))
                        if imgname.endswith(".jpg") or imgname.endswith(".png") or imgname.endswith(".jpeg") \
                                or imgname.endswith(".bmp") or imgname.endswith(".ppm"):
                            img = image.load_img(os.path.join(img_folder, subdir, imgname), target_size=(self.img_w, self.img_h))
                            img = image.img_to_array(img)
                            img = img/255.
                            if counter < numtrain:
                                data.append(img)
                            else:
                                val.append(img)
                            counter = counter + 1
        #                     move extracted image to extracted folder.
                            if not os.path.isdir(os.path.join(extracted_folder, filename)):
                                os.mkdir(os.path.join(extracted_folder, filename))
                            numfile = len([name for name in os.listdir(os.path.join(extracted_folder, filename))
                                           if os.path.isfile(os.path.join(extracted_folder, filename, name))])
                            ext = os.path.splitext(imgname)[1]
                            # print('number file extracted directory', filename)
                            # print(numfile)
                            if numfile >= self.MAX_SAVE_IMAGE: #if number file in extracted folder excel maximun number file
                                from random import randint
                                numname = randint(0, self.MAX_SAVE_IMAGE)
                            else:
                                numname = numfile
                            newname = os.path.join(extracted_folder, filename, str(numname) + ext)
                            os.rename(os.path.join(root, subdir, imgname), newname)
        data = np.asarray(data)
        val = np.asarray(val)
        print('data shape ')
        print(data.shape)
        print('val shape')
        print(val.shape)

        if data.shape[0] > 0:
            # feature = self.extractor.predict(data)
            feature = self.extractor.predict_generator(self.datagen.flow(data), steps=100)
            print('feature data shape')
            print(feature.shape)
            print(os.path.join(feature_folder, 'train'))
            if not os.path.isdir(os.path.join(feature_folder, 'train')):
                print('make new train folder')
                os.makedirs(os.path.join(feature_folder, 'train'))
###########################################
            # np.save(os.path.join(feature_folder, 'train', filename), feature)
            extendFeature(os.path.join(feature_folder, "train", filename), feature, self.MAX_SAVE_IMAGE)

        if val.shape[0] > 0:
            # val_feat = self.extractor.predict(val)
            val_feat = self.extractor.predict_generator(self.datagen.flow(val))
            print('feature val shape')
            print(val_feat.shape)
            print(os.path.join(feature_folder, 'val', filename))
            if not os.path.isdir(os.path.join(feature_folder, 'val')):
                print('make new val folder')
                os.makedirs(os.path.join(feature_folder, 'val'))
###############################################
            # np.save(os.path.join(feature_folder, 'val', filename), val_feat)
            extendFeature(os.path.join(feature_folder, "val", filename), val_feat, self.MAX_SAVE_IMAGE)

    # train svm model
    # feature folder: trong do chua folder train va val (co the co val hoac khong)
    # train folder: folder trong do chua file train feature
    # val folder: folder trong do chua file val feature
    def train_svm(self, feature_folder):
        # check existance of ./model/svm_model folder
        if not os.path.isdir('/home/workstation/Workspace/FaceID_v2/model/svm_model'):
            print('create svm_model folder')
            os.mkdir('/home/workstation/Workspace/FaceID_v2/model/svm_model')
        train_folder = os.path.join(feature_folder, 'train')
        val_folder = os.path.join(feature_folder, 'val')
        # prepare train data
        alldata = np.ndarray([0, 256])
        self.classes = []
        self.classes_index = []
        trainlabel = np.array([])
        count_class = 0
        for filename in os.listdir(train_folder):
            tem = np.load(os.path.join(train_folder, filename))
            print(tem.shape)
            alldata = np.append(alldata, tem, axis=0)
            self.classes.append(filename)
            self.classes_index.append(count_class)
            lb = np.full((tem.shape[0], 1), count_class)
            trainlabel = np.append(trainlabel, lb)
            count_class = count_class + 1
        self.classes_dict = dict(zip(self.classes, self.classes_index))
        # print(alldata.shape)
        print(self.classes_dict)
        # print(trainlabel)
        # print(trainlabel.shape)
        pickle.dump(self.classes_dict, open("/home/workstation/Workspace/FaceID_v2/model/dict.pkl", "wb"), protocol=2)

        # prepare validation data
        valdata = np.ndarray([0, 256])
        vallabel = []
        if os.path.isdir(val_folder):
            for valfilename in os.listdir(val_folder):
                tem = np.load(os.path.join(val_folder, valfilename))
                print(tem.shape)
                valdata = np.append(valdata, tem, axis=0)
                for classes, idx in self.classes_dict.items():
                    if valfilename == classes:
                        onelb = idx
                lb = np.full((tem.shape[0], 1), onelb)
                vallabel = np.append(vallabel, lb)
            # print(vallabel)
            # print(valdata.shape)

        gamma = [1e-8, 3e-8, 1e-7, 3e-7, 0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001,
                 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        accs = []
        valaccs = []
        for i in gamma:
            svm_model = svm.SVC(gamma=i, kernel='rbf', probability=True)
            svm_model.fit(alldata, trainlabel)
            check = svm_model.predict(alldata)
            check_zo = check == trainlabel
            acc = (np.count_nonzero(check_zo)) / check_zo.size*100
            print('train acc')
            print(acc)
            accs.append(acc)
            svm_file_mode = '/home/workstation/Workspace/FaceID_v2/model/svm_model/SVM_model-gamma-acc-'
            svm_file_mode += str(acc)
            if valdata.shape[0] != 0:
                valcheck = svm_model.predict(valdata)
                valcheck_zo = valcheck == vallabel
                valacc = (np.count_nonzero(valcheck_zo))/valcheck_zo.size*100
                print(valacc)
                valaccs.append(valacc)
                svm_file_mode += '-val-acc-'
                svm_file_mode += str(valacc)

            svm_file_mode += '.sav'
            pickle.dump(svm_model, open(svm_file_mode, 'wb'), protocol=2)

        final_svm_file = '/home/workstation/Workspace/FaceID_v2/model/final_svm.sav'

        maxtol = 0
        maxtolidx = 0

        maxval = 0
        maxvalidx = 0

        maxacc = 0
        maxaccidx = 0
        for i in range(len(accs)):
            if valdata.shape[0] != 0:
                if accs[i] + valaccs[i] > maxtol:
                    maxtol = accs[i] + valaccs[i]
                    maxtolidx = i
                if valaccs[i] > maxval:
                    maxval = valaccs[i]
                    maxvalidx = i
            if accs[i] > maxacc:
                maxacc = accs[i]
                maxaccidx = i

        max_res_svm = '/home/workstation/Workspace/FaceID_v2/model/svm_model/SVM_model-gamma-acc-'
        if valaccs[maxtolidx] > 0.7:
            max_res_svm += str(accs[maxtolidx])
            max_res_svm += '-val-acc-'
            max_res_svm += str(valaccs[maxtolidx])
        elif maxval > 0.7:
            max_res_svm += str(accs[maxvalidx])
            max_res_svm += '-val-acc-'
            max_res_svm += str(valaccs[maxvalidx])
        else:
            max_res_svm += str(accs[maxaccidx])
            if valdata.shape[0] > 0:
                max_res_svm += '-val-acc-'
                max_res_svm += str(valaccs[maxaccidx])
        max_res_svm += '.sav'

        print(max_res_svm)
        os.rename(max_res_svm, final_svm_file)
        shutil.rmtree('/home/workstation/Workspace/FaceID_v2/model/svm_model')


    # predict image, can goi load_finetune_model 1 lan truoc khi goi ham nay,
    # img_path = duong dan cua anh muon test
    
    def predict_image(self, img_path):
        # load classes va classes index
        dt = pickle.load(open('/home/workstation/Workspace/FaceID_v2/model/dict.pkl', 'rb'))
        print(dt)
        
        # prepare data
        img = image.load_img(img_path, target_size=(self.img_w, self.img_h))
        img = image.img_to_array(img)

        # img = img/255.
        # img = np.expand_dims(img, axis=0)
        # print(img.shape)
        img_feat = self.extractor.predict_generator(self.datagen.flow(img))
        svm_cls = pickle.load(open('/home/workstation/Workspace/FaceID_v2/model/final_svm.sav', 'rb'))
        label = svm_cls.predict(img_feat)
        print(label)
        print(svm_cls.predict_proba(img_feat))
        for name, key in dt.items():
            if label == key:
                print(name)

        # print(tensorflow.__version__)

    def test_predict_img(self, img_path):
        print(img_path)
        
    def predict_image_raw(self, img):
        dt = pickle.load(open('/home/workstation/Workspace/FaceID_v2/model/dict.pkl', 'rb'))
        print(dt)
        # prepare data
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        print(img.shape)
        img_feat = self.extractor.predict_generator(self.datagen.flow(img))
        svm_cls = pickle.load(open('/home/workstation/Workspace/FaceID_v2/model/final_svm.sav', 'rb'))
        label = svm_cls.predict(img_feat)
        print(label)
        print(svm_cls.predict_proba(img_feat))
        for name, key in dt.items():
            if label == key:
                print(name)


cdef public object createInstanceStacknet():
    print('create stack net instance')
    return FaceRecognizer(37, 224, 224)
cdef public void loadFinetuneModelForIns(object ins):
    ins.load_finetune_models()
    print('load finetune model done')
cdef public void insExtractFeature(object ins, folName, imgfol, exfol):
    ins.extract_feature(folName, imgfol, exfol, 0.2)
cdef public void predict_path_img(object ins, img_path):
    ins.test_predict_img(img_path)
