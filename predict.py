
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.train import make_predictions

if __name__ == '__main__':
    args = parse_train_args()
    # args.checkpoint_dir = './ckpt'
    modify_train_args(args)

    df_smis = pd.read_csv(args.data_path)
    df_features = pd.read_csv(args.features_path[0])

    df = pd.concat([df_smis, df_features], axis=1)

    pred, smiles = make_predictions(args)
    for i in range(len(pred[0])):
        df[f'pred_{i}'] = [item[i] for item in pred]

    print(f"Saving to file {args.save_dir + '/Predict.csv'}")
    df.to_csv(args.save_dir + '/Predict.csv', index=False)