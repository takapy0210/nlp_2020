import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, SimpleRNN, LSTM, Conv1D, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.models import Model
from transformers import TFBertModel, TFRobertaModel


class RNNModel:

    def __init__(self, input_dim, output_dim,
                 emb_dim=300, hid_dim=100,
                 embeddings=None, trainable=True):
        self.input = Input(shape=(None,), name='input')
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       mask_zero=True,
                                       trainable=trainable,
                                       name='embedding')
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       mask_zero=True,
                                       trainable=trainable,
                                       embeddings_initializer=tf.keras.initializers.Constant(embeddings),
                                       name='embedding')
        self.rnn = SimpleRNN(hid_dim, name='rnn')
        self.fc = Dense(output_dim, activation='softmax')

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        output = self.rnn(embedding)
        y = self.fc(output)
        return Model(inputs=x, outputs=y)


class BiRNNModel:

    def __init__(self, input_dim, output_dim,
                 emb_dim=300, hid_dim=100,
                 embeddings=None, trainable=True):
        self.input = Input(shape=(None,), name='input')
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       mask_zero=True,
                                       trainable=trainable,
                                       name='embedding')
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       mask_zero=True,
                                       trainable=trainable,
                                       embeddings_initializer=tf.keras.initializers.Constant(embeddings),
                                       name='embedding')
        self.rnn = Bidirectional(SimpleRNN(hid_dim, name='rnn'))
        self.fc = Dense(output_dim, activation='softmax')

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        output = self.rnn(embedding)
        y = self.fc(output)
        return Model(inputs=x, outputs=y)


class BiRNNModel_2L:

    def __init__(self, input_dim, output_dim,
                 emb_dim=300, hid_dim=100,
                 embeddings=None, trainable=True):
        self.input = Input(shape=(None,), name='input')
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       mask_zero=True,
                                       trainable=trainable,
                                       name='embedding')
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       mask_zero=True,
                                       trainable=trainable,
                                       embeddings_initializer=tf.keras.initializers.Constant(embeddings),
                                       name='embedding')
        self.rnn1 = Bidirectional(SimpleRNN(hid_dim, return_sequences=True, name='rnn1'))
        self.rnn2 = Bidirectional(SimpleRNN(hid_dim, name='rnn2'))
        self.fc = Dense(output_dim, activation='softmax')

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        rnn1 = self.rnn1(embedding)
        output = self.rnn2(rnn1)
        y = self.fc(output)
        return Model(inputs=x, outputs=y)


class LSTMModel:

    def __init__(self, input_dim, output_dim,
                 emb_dim=300, hid_dim=100,
                 embeddings=None, trainable=True):
        self.input = Input(shape=(None,), name='input')
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       mask_zero=True,
                                       trainable=trainable,
                                       name='embedding')
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       mask_zero=True,
                                       trainable=trainable,
                                       embeddings_initializer=tf.keras.initializers.Constant(embeddings),
                                       name='embedding')
        self.lstm = LSTM(hid_dim, name='lstm')
        self.fc = Dense(output_dim, activation='softmax')

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        output = self.lstm(embedding)
        y = self.fc(output)
        return Model(inputs=x, outputs=y)


class CNNModel:

    def __init__(self, input_dim, output_dim,
                 filters=250, kernel_size=3,
                 emb_dim=300, embeddings=None, trainable=True):
        self.input = Input(shape=(None,), name='input')
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       trainable=trainable,
                                       name='embedding')
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       trainable=trainable,
                                       embeddings_initializer=tf.keras.initializers.Constant(embeddings),
                                       name='embedding')
        self.conv = Conv1D(filters,
                           kernel_size,
                           padding='valid',
                           activation='relu',
                           strides=1)
        self.pool = GlobalMaxPooling1D()
        self.fc = Dense(output_dim, activation='softmax')

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        conv = self.conv(embedding)
        pool = self.pool(conv)
        y = self.fc(pool)
        return Model(inputs=x, outputs=y)


class RobertaModel:

    def __init__(self, model_name, max_len, output_dim=1):
        """モデルを構築する
        """
        self.transformer = TFRobertaModel.from_pretrained(model_name)
        self.input = tf.keras.layers.Input(shape=(max_len, ), dtype=tf.int32, name='input_word_ids')
        self.fc = tf.keras.layers.Dense(output_dim, activation='softmax', name='output')

    def build(self):
        input_layer = self.input
        sequence_output = self.transformer(input_layer)[0]
        cls_token = sequence_output[:, 0, :]  # We only need the cls_token, resulting in a 2d array
        output_layer = self.fc(cls_token)
        model = Model(inputs=[input_layer], outputs=[output_layer])
        return model


class BertModel:

    def __init__(self, model_name, max_len, output_dim=1):
        """モデルを構築する
        """
        self.transformer = TFBertModel.from_pretrained(model_name)
        self.input = tf.keras.layers.Input(shape=(max_len, ), dtype=tf.int32, name='input_word_ids')
        self.fc = tf.keras.layers.Dense(output_dim, activation='softmax', name='output')

    def build(self):
        input_layer = self.input
        sequence_output = self.transformer(input_layer)[0]
        cls_token = sequence_output[:, 0, :]  # We only need the cls_token, resulting in a 2d array
        output_layer = self.fc(cls_token)
        model = Model(inputs=[input_layer], outputs=[output_layer])
        return model
