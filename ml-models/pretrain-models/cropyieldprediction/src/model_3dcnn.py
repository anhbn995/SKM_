import tensorflow as tf
import tensorflow.keras as keras

def build_model(input_shape):
    ts=input_shape[2]
    model = keras.Sequential(
    [
        # Conv_1
        keras.layers.Conv3D(64, (3,3,ts), padding='same', strides=(1, 1, 1), input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        # Pool_1
        keras.layers.MaxPooling3D((2,2,1), strides=(1,1,1)), # we do not reduce the size in the 1st pooling
        
        # Conv_2
        keras.layers.Conv3D(128, (3,3,ts), padding='same', strides=(1, 1, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        # Pool_2
        keras.layers.MaxPooling3D((2,2,1), strides=(2,2,1)), # we only reduce the spatial size in the 2nd pooling
        
        # Conv_3a
        keras.layers.Conv3D(256, (3,3,ts), padding='same', strides=(1, 1, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),        

        # Conv_3b
        keras.layers.Conv3D(256, (3,3,ts), padding='same', strides=(1, 1, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),        

        # Pool_3
        keras.layers.MaxPooling3D((2,2,1), strides=(2,2,1)), # we only reduce the spatial size in the 3rd pooling

        # Conv_4a
        keras.layers.Conv3D(512, (3,3,ts), padding='same', strides=(1, 1, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),        

        # Conv_4b
        keras.layers.Conv3D(512, (3,3,ts), padding='same', strides=(1, 1, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),        

        # Pool_4
        keras.layers.MaxPooling3D((2,2,ts), strides=(2,2,ts)), # we use the original pooling for the 4rd maxPooling (timestaps) in this layer
        
        # Flatten
        keras.layers.Flatten(),
        # Fully connected layers
        # Fc_5
        keras.layers.Dense(1024),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),          
        
        # Fc_6
        keras.layers.Dense(1),   
    ])
    
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError(), 'mae']
    )
    
    return model

if __name__=='__main__':
  input_shape=(33,33,3,10)
  model=build_model(input_shape)
  model.summary()