from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout, Conv3D, MaxPooling3D, BatchNormalization
from keras.optimizers import Adam
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import numpy as np
import os,sys
import nibabel as nib
from tensorflow.keras.backend import eval
import argparse
from keras.models import model_from_json
from urllib.request import urlopen

def pairwise_distance(feature, squared=False):

    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)

    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]

    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def create_base_network(image_input_shape, embedding_size):

    input_image = Input(shape=image_input_shape)

    x=Conv3D(16,(3,3,3), activation='relu',padding="same")(input_image)
    x=MaxPooling3D((3,3,3),strides=(1,1,1),padding='same')(x)
    x = Dropout(0.1)(x)
    x=Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
    x=MaxPooling3D((3,3,3),strides=(1,1,1),padding='same')(x)
    x = Dropout(0.1)(x)
    x=Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
    x=MaxPooling3D((3,3,3),strides=(1,1,1),padding='same')(x)
    x = Dropout(0.1)(x)
    x=Flatten()(x)
    x = Dense(embedding_size)(x)

    base_network = Model(inputs=input_image, outputs=x)

    return base_network


def normalize(img,thr):
    ind = np.where(img <thr)
    img[ind]=0
    # normalize inputs to 0-1
    max_v = np.percentile(img,99.5)
    img[img>max_v]=max_v
    min_v = 0
    img = (img - min_v)  / (max_v - min_v)

    return img

def dist(model,embedding_size,imgA,imgB):
    input_image_shape = (92, 108, 92, 1)

    # creating an empty network
    testing_embeddings = create_base_network(input_image_shape,
                                             embedding_size=embedding_size)

    # Grabbing the weights from the trained network
    for layer_target, layer_source in zip(testing_embeddings.layers, model.layers[2].layers):
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        del weights

    test=np.zeros((2, 92,108,92))

    img0=nib.load(imgA)
    img0_data=normalize(img0.get_data(),0)
    img1=nib.load(imgB)
    img1_data=normalize(img1.get_data(),0)

    test[0,:,:,:]=img0_data
    test[1,:,:,:]=img1_data

    x_embeddings = testing_embeddings.predict(np.reshape(test, (len(test), 92, 108, 92, 1)))
 
    pdist = pairwise_distance(x_embeddings, squared=True)
    distM=eval(pdist)
 
    dist=(distM[:,-1])
    print("img1:{}".format(imgA))
    print("img2:{}".format(imgB))    
    print("Distance:{}".format(dist[:-1]))

# classification of the MR image 
def classify(my_model,embedding_size,img):
 
    input_image_shape = (92, 108, 92, 1)

    # creating an empty network
    testing_embeddings = create_base_network(input_image_shape,
                                             embedding_size=embedding_size)
    # Grabbing the weights from the trained network
    for layer_target, layer_source in zip(testing_embeddings.layers, my_model.layers[2].layers):
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        del weights

    test=np.zeros((6, 92,108,92))
    img0=nib.load('template/t1_template.nii')
    img0_data=normalize(img0.get_data(),0)
    img1=nib.load('template/t1c_template.nii')
    img1_data=normalize(img1.get_data(),0)
    img2=nib.load('template/t2_template.nii')
    img2_data=normalize(img2.get_data(),0)
    img3=nib.load('template/fl_template.nii')
    img3_data=normalize(img3.get_data(),0)
    img4=nib.load('template/flc_template.nii')
    img4_data=normalize(img4.get_data(),0)

    test[0,:,:,:]=img0_data
    test[1,:,:,:]=img1_data
    test[2,:,:,:]=img2_data
    test[3,:,:,:]=img3_data
    test[4,:,:,:]=img4_data
   
    img5=nib.load(img)
    img5_data=normalize(img5.get_data(),0)

    test[5,:,:,:]=img5_data

    x_embeddings = testing_embeddings.predict(np.reshape(test, (len(test), 92, 108, 92, 1)))
 
    pdist = pairwise_distance(x_embeddings, squared=True)
    distM=eval(pdist)
    dist=(distM[:,-1])
 
    index=np.argmin(dist[:-1])
    if index==0:
        contrast='T1'
    elif index==1:
        contrast='post contrast T1'
    elif index==2:
        contrast='T2'
    elif index==3:
        contrast='FLAIR'
    elif index==4:
        contrast='post contrast FLAIR'

    print("Image:{} -- {}".format(img,contrast))        
       
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode",type=str, default='dist')
	parser.add_argument("--img1", type=str)
	parser.add_argument("--img2", type=str)
	parser.add_argument("--gpu", type=str,default='-1') # default is for running on CPU mode; enter 0,1,2,..... for GPU mode

	args=parser.parse_args()

	return args

if __name__ == "__main__":
	args=get_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	# load json and create model
	json_file = open('model/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	if not os.path.isfile('model/weight_001.h5'):
		url = "https://zenodo.org/record/4402306/files/weight_001.h5?download=1"
		print("Downloading", url, "...")
		data = urlopen(url).read()
		with open('model/weight_001.h5', 'wb') as f:
			f.write(data)

	# load weights into new model
	loaded_model.load_weights("model/weight_001.h5")
	print("Loaded model from disk")

	embedding_size =32
	
	if args.mode=='dist':
		# compute the distance between img1 and img2
		dist(loaded_model,embedding_size,args.img1,args.img2)

	elif args.mode=='classify':
		# classify the MR image contrast
		classify(loaded_model,embedding_size,args.img1)
