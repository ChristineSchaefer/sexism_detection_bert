import tensorflow as tf


def train(model, x_train_ids, x_train_attention, y_train, epochs, batch_size, num_steps, x_valid_ids, x_valid_attention, y_valid):
    path = 'training_model/model/'

    try:
        loaded_model = tf.keras.models.load_model(path + '{model.name}')
        print('Load trained model')
        return loaded_model

    except IOError:
        print('No trained model found in the directory. Start training.')
        # start training
        trained_model = model.fit(
            x=[x_train_ids, x_train_attention],
            y=y_train.to_numpy(),
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=num_steps,
            validation_data=([x_valid_ids, x_valid_attention], y_valid.to_numpy()),
            verbose=2
        )
        # TODO: save model
        trained_model.save(path + '{model.name}')
        print(trained_model.history)

        return trained_model


    


