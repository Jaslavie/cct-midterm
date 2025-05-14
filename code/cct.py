import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath="data/plant_knowledge.csv"):
    """
    Load and return plant data as a numpy array of informants x items
    Informants are unique rows of data / observations
    items are the columns of the data
    Drop the ID column because we are only interested in the data
    """
    df = pd.read_csv(filepath)
    informants_ids = df['Informant ID'].unique() # extract unique observations
    X = df.drop(columns=['Informant ID']).values # drop ID column and convert everything to a numpy array
    
    return X, informants_ids