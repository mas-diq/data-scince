import tensorflow as tf

#M Shiddiq F
#14118957

#menginput data
#28x28 dataset angka dari 0-9
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

#normalisasi
x_train = tf.keras.utils.normalize(x_train, axis=1) 
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(28, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(28, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy'
  ,metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)
