import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from transformers import *
from transformers import TFDistilBertModel,DistilBertTokenizer,DistilBertConfig

max_len, num_classes = 32, 2
# get tokenizer
dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# get model
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


def bert_preproc (sentences, labels):
    # Prepare the model input
    input_ids = []
    attention_masks = []

    for sent in sentences:
        dbert_inps = dbert_tokenizer.encode_plus(sent, add_special_tokens=True, max_length=max_len,
                                                 pad_to_max_length=True, return_attention_mask=True, truncation=True)
        input_ids.append(dbert_inps['input_ids'])
        attention_masks.append(dbert_inps['attention_mask'])

    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)
    labels = np.array(labels)
    return input_ids, attention_masks, labels


def bert_model ():
    inps = Input(shape=(max_len,), dtype='int64')
    masks = Input(shape=(max_len,), dtype='int64')
    dbert_layer = dbert_model(inps, attention_mask=masks)[0][:, 0, :]
    dense = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dbert_layer)
    dropout = Dropout(0.5)(dense)
    pred = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(dropout)
    model = tf.keras.Model(inputs=[inps, masks], outputs=pred)
    print(model.summary())
    return model



class BERT_Classification:
    def __init__(self):

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        self.model = bert_model ()


    def bert_train (self, workspace, train_inp, train_mask, train_label, val_inp, val_mask, val_label,num_epochs):
        log_dir = workspace + os.sep + 'log'
        os.makedirs(log_dir, exist_ok=True)
        model_save_path = workspace + os.sep +  'dbert_model.h5'
        chkpt = ModelCheckpoint(filepath = model_save_path, save_weights_only=True, monitor='val_loss',
                                               mode='min', save_best_only=True)
        tboard = TensorBoard(log_dir=log_dir)
        callbacks = [chkpt, tboard]

        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
        history = self.model.fit([train_inp, train_mask], train_label, batch_size=16, epochs=num_epochs,
                            validation_data=([val_inp, val_mask], val_label), callbacks=callbacks)
        return history


    def bert_predict (self, workspace, val_inp, val_mask):
        model_save_path = workspace + os.sep + 'dbert_model.h5'
        trained_model = bert_model()
        trained_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
        trained_model.load_weights(model_save_path)
        preds = trained_model.predict([val_inp, val_mask], batch_size=16)
        pred_labels = preds.argmax(axis=1)
        return pred_labels, preds.max(axis=1)
