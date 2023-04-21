import argparse
import pandas as pd
from text_cleaning import preprocess_sentence
from dbert_model import bert_preproc, BERT_Classification


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-file',help='Input File')
    parser.add_argument('-m','--model-path',help='Trained Model path')
    parser.add_argument('-c','--column-name',help='Column Name')

    args = parser.parse_args()

    df = pd.read_csv (args.input_file,nrows=100)

    y = df[args.column_name].to_list()
    mask = [ i for i in range (0, len (y)) if  isinstance(y[i], str)]
    df = df.iloc[mask]


    df.dropna()
    df[args.column_name] = df[args.column_name].map(preprocess_sentence)

    # prepare the data
    max_len = 32
    sentences = df[args.column_name]
    input_ids, attention_masks  = bert_preproc (sentences)

    M = BERT_Classification()
    lab_p, prob_p = M.bert_predict(args.model_path, input_ids, attention_masks)

    print(lab_p)
    print(prob_p)
