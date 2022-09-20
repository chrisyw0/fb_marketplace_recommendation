import tensorflow as tf
from fbRecommendation.dl.tensorflow.model.tf_model_util import TFModelUtil


class TFBaseModel:
    @classmethod
    def get_model(cls, **kwargs):
        pass

    @classmethod
    def load_model(cls, model, model_file_path):
        model.load_weights(model_file_path)


class TFImageModel(TFBaseModel):
    @classmethod
    def get_model(cls, **kwargs):
        num_class = kwargs.get("num_class")
        model_name = kwargs.get("model_name")
        dropout_conv = kwargs.get("dropout_conv")
        dropout_pred = kwargs.get("dropout_pred")
        image_shape = kwargs.get("image_shape")
        image_base_model = kwargs.get("image_base_model")

        inputs = tf.keras.layers.Input(shape=image_shape)

        img_augmentation = tf.keras.models.Sequential(
            [
                tf.keras.layers.RandomFlip('horizontal'),
                tf.keras.layers.RandomRotation(0.2)
            ],
            name="img_augmentation"
        )

        if image_base_model == "RestNet50":
            preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        elif image_base_model.startswith("EfficientNet"):
            preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

        tf_image_base_model = TFModelUtil.prepare_image_base_model(
            image_base_model,
            image_shape
        )

        image_global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name="pooling")
        image_dropout_0 = tf.keras.layers.Dropout(dropout_conv, name="dropout_0")
        image_dropout_1 = tf.keras.layers.Dropout(dropout_pred, name="dropout_1")
        image_dense_layer_0 = tf.keras.layers.Dense(1024, activation="relu", name="dense_0")
        image_dense_layer_1 = tf.keras.layers.Dense(256, activation="relu", name="dense_1")
        prediction = tf.keras.layers.Dense(num_class, name="prediction")

        layers = [
            tf_image_base_model,
            image_global_average_layer,
            image_dropout_0,
            image_dense_layer_0,
            image_dense_layer_1,
            image_dropout_1,
        ]

        image_seq_layers = tf.keras.Sequential(
            layers=layers, name="image_sequential"
        )

        x = img_augmentation(inputs)
        x = preprocess_input(x)
        x = image_seq_layers(x)
        outputs = prediction(x)

        return tf.keras.Model(inputs, outputs, name=model_name), tf_image_base_model, image_seq_layers


class TFTextModel(TFBaseModel):
    @classmethod
    def get_model(cls, **kwargs):
        num_class = kwargs.get("num_class")
        model_name = kwargs.get("model_name")
        dropout_conv = kwargs.get("dropout_conv")
        dropout_pred = kwargs.get("dropout_pred")
        num_max_tokens = kwargs.get("num_max_tokens")
        embedding_model = kwargs.get("embedding_model")

        embedding_layer = TFModelUtil.gensim_to_keras_embedding(
            embedding_model,
            train_embeddings=False,
            input_shape=(num_max_tokens,))

        text_conv_layer_1 = tf.keras.layers.Conv1D(48, 3, activation="relu", name="text_conv_1")
        text_pooling_layer_1 = tf.keras.layers.AveragePooling1D(2, name="text_avg_pool_1")
        text_dropout_1 = tf.keras.layers.Dropout(dropout_conv, name="text_dropout_conv_1")
        text_conv_layer_2 = tf.keras.layers.Conv1D(24, 3, activation="relu", name="text_conv_2")
        text_pooling_layer_2 = tf.keras.layers.AveragePooling1D(2, name="text_avg_pool_2")
        text_flatten = tf.keras.layers.Flatten(name="text_flatten")
        text_dropout_2 = tf.keras.layers.Dropout(dropout_conv, name="text_dropout_conv_2")
        text_dense = tf.keras.layers.Dense(256, activation='relu', name="text_dense_1")
        text_dropout_pred = tf.keras.layers.Dropout(dropout_pred, name="text_dropout_pred_1")
        prediction = tf.keras.layers.Dense(num_class, name="prediction")

        inputs = tf.keras.layers.Input(shape=(num_max_tokens,), name="input")

        layers = [
            embedding_layer,
            text_conv_layer_1,
            text_pooling_layer_1,
            text_dropout_1,
            text_conv_layer_2,
            text_pooling_layer_2,
            text_flatten,
            text_dropout_2,
            text_dense,
            text_dropout_pred
        ]

        text_seq_layers = tf.keras.models.Sequential(layers=layers,
                                                     name="text_seq_layers")

        x = text_seq_layers(inputs)
        outputs = prediction(x)

        model = tf.keras.Model(inputs, outputs, name=model_name)

        return model, text_seq_layers, embedding_layer


class TFTextTransformerModel(TFBaseModel):
    @classmethod
    def get_model(cls, **kwargs):
        num_class = kwargs.get("num_class")
        model_name = kwargs.get("model_name")
        embedding = kwargs.get("embedding")
        embedding_dim = kwargs.get("embedding_dim")
        embedding_pretrain_model = kwargs.get("embedding_pretrain_model")
        dropout_pred = kwargs.get("dropout_pred")

        embedding_model, embedding_preprocess = TFModelUtil.prepare_embedding_model(
            embedding=embedding,
            embedding_dim=embedding_dim,
            pretrain_model=embedding_pretrain_model,
            trainable=False
        )

        TFModelUtil.set_base_model_trainable(embedding_model, 1)

        inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input")

        text_dense_layer_1 = tf.keras.layers.Dense(256, activation='relu', name="dense_1")
        text_dropout_layer_1 = tf.keras.layers.Dropout(dropout_pred, name="dropout")
        prediction_layer = tf.keras.layers.Dense(num_class, name="prediction")

        x = embedding_preprocess(inputs)
        x = embedding_model(x, training=False)["pooled_output"]
        x = text_dropout_layer_1(x)
        text_model_output = text_dense_layer_1(x)

        text_seq_layer = tf.keras.Model(inputs, text_model_output, name="text_seq_layer_transformer")

        x = text_seq_layer(inputs)
        outputs = prediction_layer(x)

        model = tf.keras.Model(inputs, outputs, name=model_name)

        return model, text_seq_layer, embedding_model


class TFCombineModel(TFBaseModel):
    @classmethod
    def get_model(cls, **kwargs):
        num_class = kwargs.get("num_class")
        model_name = kwargs.get("model_name")
        is_transformer_based_text_model = kwargs.get("is_transformer_based_text_model")
        num_max_tokens = kwargs.get("num_max_tokens")
        image_shape = kwargs.get("image_shape")
        text_seq_layers = kwargs.get("text_seq_layers")
        image_seq_layers = kwargs.get("image_seq_layers")
        image_base_model = kwargs.get("image_base_model")

        img_augmentation = tf.keras.models.Sequential(
            [
                tf.keras.layers.RandomFlip('horizontal'),
                tf.keras.layers.RandomRotation(0.2)
            ],
            name="img_augmentation"
        )

        prediction = tf.keras.layers.Dense(num_class, name="prediction")

        if is_transformer_based_text_model:
            text_inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input")
        else:
            text_inputs = tf.keras.layers.Input(shape=(num_max_tokens,), dtype=tf.int32, name="input")

        image_inputs = tf.keras.layers.Input(shape=image_shape, dtype=tf.float32)

        x_text = text_seq_layers(text_inputs)
        x_img = img_augmentation(image_inputs)

        if image_base_model == "RestNet50":
            preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
            x_img = preprocess_input(x_img)
        elif image_base_model.startswith("EfficientNet"):
            preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
            x_img = preprocess_input(x_img)

        x_img = image_seq_layers(x_img)
        x = tf.concat([x_text, x_img], 1)

        outputs = prediction(x)
        model = tf.keras.Model(
            {
                "token": text_inputs,
                "image": image_inputs
            },
            outputs, name=model_name
        )

        return model
