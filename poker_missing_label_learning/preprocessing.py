import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


selected_columns = [
    "level",
    "stack",
    "pot_pre",
    "pot_flop",
    "blinds",
    "bet_pre",
    "bet_flop",
    "result"
]

def load_data(filepath: str) -> pd.DataFrame:
    try:
        df_raw = pd.read_csv(filepath)
        df = df_raw[selected_columns]
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f'File {filepath} not found.')
    except KeyError as e:
        raise KeyError(f'Wanted features are not in the {filepath}.')


def binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Convert result column to binary
    mapping = {
        'gave up': 0,
        'lost': 0,
        'won': 1,
        'took chips': 1
    }
    df['result'] = df['result'].map(mapping)

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    #Feature Engineering
    df = df.copy()
    # Avoid division by zero
    epsilon = 1e-6
    df['stack_to_pot'] = df['stack'] / (df['pot_pre'] + epsilon)
    df['total_bet'] = df['bet_pre'] + df['bet_flop']
    df['pot_growth'] = df['pot_flop'] / (df['pot_pre'] + epsilon)
    df['stack_depth'] = df['stack'] / (df['blinds'] + 1)
    df['total_pot'] = df['pot_pre'] + df['pot_flop']

    return df

def scale_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    log_features = [
        'stack', 'pot_pre', 'pot_flop', 'bet_pre', 'bet_flop',
        'total_bet', 'stack_depth', 'total_pot'
    ]

    # Log1p (log(1+x))
    for col in log_features:
        df[col] = np.log1p(df[col])

    scaler = MinMaxScaler()

    # Min-Max
    minmax_features = [
        'stack', 'pot_pre', 'pot_flop', 'bet_pre', 'bet_flop',
        'total_bet', 'stack_depth', 'total_pot', 'stack_to_pot',
        'pot_growth', 'blinds', 'level'
    ]

    df[minmax_features] = scaler.fit_transform(df[minmax_features])

    return df

def run_pipeline(input_path: str, output_path: str):
    df_raw = load_data(input_path)
    df_labeled = binarize_labels(df_raw)
    df_engineered = engineer_features(df_labeled)
    df_final = scale_and_transform(df_engineered)
    df_final.to_csv(output_path, index=False)
    pass

if __name__ == "__main__":
    INPUT_FILE = "one_dollar_spin_and_go.csv"
    OUTPUT_FILE = "poker_data_preprocessed.csv"

    run_pipeline(INPUT_FILE, OUTPUT_FILE)
