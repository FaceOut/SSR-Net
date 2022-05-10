import cv2
import numpy as np

def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def augment_data(images):
    for i in range(0,images.shape[0]):
        if np.random.random() > 0.5:
            images[i] = images[i][:,::-1]
        if np.random.random() > 0.8:
            images[i] = grayscale(images[i])
    return images

def data_generator(X,Y,batch_size):
    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        p,q = [],[]
        for i in range(len(X)):
            p.append(X[i])
            q.append(Y[i])
            if len(p) == batch_size:
                yield augment_data(np.array(p)),np.array(q)
                p,q = [],[]
        if p:
            yield augment_data(np.array(p)),np.array(q)
            p,q = [],[]
