import csv
import cv2
import numpy as np

path = "./data"
folders = ["/dust", "/dust_road", "/left", "/rec_right", "/rec_r2c", "/left_full_2", "/right_full_2", "/sharp_turns"]


images = []
measurements = []
for folder in folders:
    root = path + folder
    lines = []
    print(root)
    with open(root + "/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

        for line in lines[1:]:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = root + "/IMG/" + filename
            measurement = float(line[3])
            if (measurement != 0.0):
                measurements.append(measurement)
                measurements.append(measurement*(-1))
                image = cv2.imread(current_path)
                image_inv = np.fliplr(image)
                images.append(image)
                images.append(image_inv)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, validation_split=0.2, nb_epoch=5, shuffle=True)
model.save('model.h5')
