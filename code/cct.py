import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
# N = number of informants
# M = number of items (questions)
# Xij = informant i's response to item j (0 or 1)
# Zj = latent "consensus" or "correct" answer for item j (0 or 1)
# Di = latent "competence" of informant i (probability of knowing the correct answer), where 0.5 ≤ Di ≤ 1.

def load_data(filepath="data/plant_knowledge.csv"):
    """
    Load and return plant data as a numpy array of informants x items
    Informants are unique rows of data / observations (i.e. participants in the study)
    items are the columns of the data (Yes/No responses if plants can be used for medicine)
    Drop the ID column because we are only interested in the data
    """
    df = pd.read_csv(filepath)
    informants_ids = df['Informant ID'].unique() # extract unique observations
    X = df.drop(columns=['Informant ID']).values # drop ID column and convert everything to a numpy array
    
    return X, informants_ids

def build_model(X):
    """
    Build the CCT model 
    - Input: binary response data matrix
    - Output: PyMC model (i.e. a probabilistic model that represents the likelihood of certain outcomes)
    We will use this 
    """
    N, M = X.shape # extracts the number of informants and items

    # creates a model object
    with pm.Model() as model:
        # define the prior distribution of the informant competence (D)
        # Beta dist. scaled to [0.5, 1] range (assumption that most have moderate competence)
        # alpha=2 and beta=2 are default values for the distribution
        # Distribution over the number of informants (N)
        D_raw = pm.Beta('D_raw', alpha=2, beta=2, shape=N)
        D = pm.Deterministic("D", 0.5 + 0.5 * D_raw)

        # Define the prior distribution of the consensus answer (Z)
        # Bernoulli dist. with prior probability of 0.5 (uncertainty before data is observed)
        # Distribution over the number of items (M)
        Z = pm.Bernoulli('Z', p=0.5, shape=M)

        # Turn D into a column vector (currently a row vector)
        # Easier to perform matrix multiplication with Z
        D_reshaped = D[:, None]

        # Calculate probability matrix of p_ij (prob of informant i knowing the correct answer to item j)
        
        # p_ij = Z_j * D_i + (1 - Z_j) * (1 - D_i)
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)

        # Define the likelihood of the observed data (X) after observing the data
        # X is pulled from the data matrix in the provided data file
        X_obs = pm.Bernoulli('X_obs', p=p, observed=X)

    return model
        