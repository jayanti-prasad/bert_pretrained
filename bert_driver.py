import os
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from text_cleaning import preprocess_sentence
from dbert_model import bert_preproc, BERT_Classification

def get_data (input_file, column_label):
    # read input data
    if column_label == 'GOOD_BAD':
        column_name = 'Good Comment'
    elif column_label == 'WHAT':
        column_name = 'WHAT Happened?'
    elif column_label == 'WHY':
        column_name = 'WHY it Happened?'
    elif column_label == 'HOW':
        column_name = 'HOW it was Fixed?'
    else:
        print("Model  name not valid")
        sys.exit()

    prefix = input_file.split(".")[1]
    if prefix == 'xlsx':
        df = pd.read_excel(input_file, engine="openpyxl")
    else:
        df = pd.read_csv(input_file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.dropna()
    df = df.reset_index(drop=True)
    mapping = {'n': 0, 'y': 1}
    df = df.replace({column_name: mapping})
    df.rename(columns={column_name: 'label'}, inplace=True)
    df = shuffle(df)
    return df



def plot_history(history):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    axs[1].plot(history.history['accuracy'], '-o', label="Training")
    axs[1].plot(history.history['val_accuracy'], '-o', label='Validation')
    axs[0].plot(history.history['loss'], '-o', label='Training')
    axs[0].plot(history.history['val_loss'], '-o', label='Validation')

    axs[0].legend()
    axs[1].legend()
    plt.legend()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-file',help='Input File')
    parser.add_argument('-c','--column-name',help='Column Name')
    parser.add_argument('-o','--output-dir',help='Output dir')
    parser.add_argument('-n','--num_epochs',type=int,help='Number of epochs')

    args = parser.parse_args()

    data = get_data(args.input_file, args.column_name)

    data.rename(columns={'COMPLEATION_NOTES': 'text'}, inplace=True)

    workspace = args.output_dir

    os.makedirs (workspace, exist_ok=True)

    #data['gt'] = data['label'].map({'n': 0, 'y': 1})
    #print('Available labels: ', data.label.unique())
    data['text'] = data['text'].map(preprocess_sentence)
    num_classes = len(data.label.unique())

    # prepare the data
    max_len = 32
    sentences = data['text']
    labels = data['label'].to_list()
    labels = np.array(labels)
    print(len(sentences), len(labels))
    print("labels=",labels)
    #sys.exit()

    input_ids, attention_masks  = bert_preproc (sentences)

    label_class_dict = {0: 'n', 1: 'y'}
    target_names = label_class_dict.values()

    train_inp, val_inp, train_label, val_label, train_mask, val_mask \
        = train_test_split(input_ids, labels,attention_masks, test_size=0.2)

    M = BERT_Classification()
    history = M.bert_train(workspace, train_inp, train_mask, train_label, val_inp, val_mask, val_label,args.num_epochs)
    plot_history(history)

    lab_p, prob_p = M.bert_predict(workspace, val_inp, val_mask)

    print(lab_p)
    print(prob_p)
