import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from constants import LABEL_COLORS


def read_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# Work with this function whilst integrating MLFlow
def get_artifacts(df: pd.DataFrame):
    # Class Distribution
    sns.countplot('Class', data=df, palette=LABEL_COLORS)
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

    fig, ax = plt.subplots(1, 2, figsize=(18,4))

    amount_val = df['Amount'].values
    time_val = df['Time'].values

    sns.distplot(amount_val, ax=ax[0], color='r') #TODO Deal with the deprecation
    ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
    ax[0].set_xlim([min(amount_val), max(amount_val)])

    sns.distplot(time_val, ax=ax[1], color='b') #TODO Deal with the deprecation
    ax[1].set_title('Distribution of Transaction Time', fontsize=14)
    ax[1].set_xlim([min(time_val), max(time_val)])
    pass