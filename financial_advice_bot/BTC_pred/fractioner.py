import pandas as pd
import numpy as np

def df_frac(df):
# making a Dataframe sample to reduce the weight of the processing data
# 'frac' is the % of the sample we take (float values 0 to 1)
# 'replace' when we sample with replacement, the two sample values are independent
# weights used to ensure that metrics derived from a data set are representative of the population (the set of observations)
    yes_no = {"yes":['y', 'yes'], "no": ['n', 'no']}

    frac_q = input(f'The file has {len(df)} lines. Do you want to fraction the Dataframe? ' ).lower()
    while frac_q not in yes_no['yes'] and frac_q not in yes_no['no']:
        frac_q = input(f'The file has {len(df)} lines. Do you want  to fraction the Dataframe? Type y/n ' )

    if frac_q in yes_no['yes']:
        frac = input('What fraction do you want to take? \nDecimal values between 0 and 1. ')
        while float(frac) not in np.arange(0, 1, 0.1):
            frac = input('What fraction do you want to make? \nDecimal values between 0 and 1. For example, 0.1 ')
        frac = float(frac)
        df = df.sample(frac=frac, replace=True, random_state=123)
        print(f'After removing a {(1-frac)*100} %, the Dataframe has {len(df)} lines.')

    df = df.sort_index()
    return df