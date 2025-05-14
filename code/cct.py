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

def sample_posterior(model, draws=2000, chains=4, tune=1000):
    """
    Performs Bayesian inference on the CCT model using Markov Chain Monte Carlo (MCMC) sampling
    MCMC samples from the posterior distribution of the model parameters (i.e. samples collected after the model has run and observed data)
    The sampler explores different possible values for unkown parameters by taking small steps toward the answer using the data
    Chaining allows us to check multiple potential answers and determine the most likely answer
    - 2000 draws represents the number of posterior samples drawn per chain
    - 4 chains represents the number of parallel MCMC chains
    - 1000 tune represents the number of samples discarded before sampling. 
    Returns: a trace of PyMC inference data containing posterior samples
    """
    with model:
        trace = pm.sample(
            draws=draws,
            chains=chains,
            tune=tune,
            # tells the sampler to return an ArviZ InferenceData object
            # The ArviZ is a container for posterior samples, diagnostics, and other information
            return_inferencedata=True
        )

    return trace

def analyze_model(trace, X, informant_ids=None):
    """
    Analyzes the results of hte model
    Returns
    - competence_df: the posterior competence of each informant
    - consensus_df: the posterior consensus of each item
    """
    # --- check for convergence from MCMC sampling -----
    # arviz is a library that analyzes the results of MCMC sampling
    summary = az.summary(trace, var_names=['D', 'Z'])
    print("Convergence Diagnostics:")
    print(f"- R-hat values should be close to 1.0 (< 1.05 is good)")
    print(f"- Min R-hat: {summary['r_hat'].min():.4f}, Max R-hat: {summary['r_hat'].max():.4f}")
    print(f"- Effective sample sizes (ESS) should be large relative to actual samples")
    print(f"- Min ESS: {summary['ess_bulk'].min():.1f}")
    # model does not converge if the R-hat value (if all chains agreed with each other) is greater than 1.05 (exploring options)
    # otherwise the model has converged
    if summary['r_hat'].max() > 1.05:
        print("Model may not have converged.")
    else:
        print("Model had converged.")
    
    # ----- estimate informant competence -----
    # extract competence from the trace
    d_samples = trace.posterior["D"].values
    d_mean = d_samples.mean(axis=0)
    d_hdi = az.hdi(d_samples.reshape(-1, len(d_mean))) # the most likely range of competence (appears the most)

    # create a df with informant ids and their associated competence
    # sort in descending order of competence
    if informant_ids is None:
        informant_ids = [f"Informant{i+1}" for i in range(len(d_mean))]
    competence_df = pd.DataFrame({
        "Informant": informant_ids,
        "Competence": d_mean,
        "HDI_low": d_hdi[:, 0], # lower bound of most likely guess
        "HDI_high": d_hdi[:, 1] # upper bound of most likely guess
    })
    competence_df = competence_df.sort_values("Competence", ascending=False)
    
    # print the results
    print("\nInformant Competence Estimates:")
    print(f"- Most competent: {competence_df.iloc[0]['Informant']} ({competence_df.iloc[0]['Competence']:.3f})")
    print(f"- Least competent: {competence_df.iloc[-1]['Informant']} ({competence_df.iloc[-1]['Competence']:.3f})")
    print(f"- Average competence: {competence_df['Competence'].mean():.3f}")

    # ----- estimate consensus answers -----
    z_samples = trace.posterior["Z"].values
    z_mean = z_samples.mean(axis=0)
    z_consensus = (z_mean > 0.5).astype(int) # finds the value with a probability over 50%
    majority_vote = (X.mean(axis=0) > 0.5).astype(int) # finds the value with a percentage over 50%
    
    # Create DataFrame with consensus estimates
    consensus_df = pd.DataFrame({
        "Item": [f"Item {i+1}" for i in range(len(z_mean))],
        "P(Z=1)": z_mean,
        "CCT_Consensus": z_consensus,
        "Majority_Vote": majority_vote,
        "Match": z_consensus == majority_vote
    })
    consensus_df = consensus_df.sort_values("Consensus", ascending=False)
    print("\nConsensus Answer Estimates:")
    print(f"- Items where CCT and majority vote agree: {consensus_df['Match'].sum()} of {len(consensus_df)}")

    # find points where CCT and majority vote disagree
    # compares the majority vote (naive aggregation) of each question with the consensus answer form the CCT model
    disagreements = consensus_df[~consensus_df["Match"]]
    if len(disagreements) > 0:
        print("Items where CCT and majority vote disagree:")
        for _, row in disagreements.iterrows():
            print(f"  * {row['Item']}: CCT says {'Yes' if row['CCT_Consensus']==1 else 'No'} " +
                f"(p={row['P(Z=1)']:.2f}), Majority says {'Yes' if row['Majority_Vote']==1 else 'No'}")
            
    # ----- visualize competence and consensus -----
    # generated by claude 3.7
    # competence
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        competence_df["Competence"],
        np.arange(len(competence_df)),
        xerr=np.vstack([
            competence_df["Competence"] - competence_df["HDI_low"],
            competence_df["HDI_high"] - competence_df["Competence"]
        ]),
        fmt='o',
        capsize=5
    )
    plt.yticks(np.arange(len(competence_df)), competence_df["Informant"])
    plt.xlabel("Competence (Probability of Correct Answer)")
    plt.ylabel("Informant")
    plt.title("Estimated Informant Competence with 95% Credible Intervals")
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # consensus
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(z_mean)), z_mean, alpha=0.7)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    
    # Add markers for where CCT and majority vote differ
    if len(disagreements) > 0:
        disagreement_indices = [int(item.split()[1])-1 for item in disagreements["Item"]]
        plt.scatter(disagreement_indices, z_mean[disagreement_indices], 
                   color='red', s=100, zorder=3, label='Disagrees with majority')
    
    plt.xlabel("Item Number")
    plt.ylabel("P(Consensus Answer = Yes)")
    plt.title("Estimated Consensus Answers")
    plt.xticks(np.arange(len(z_mean)), [f"Item {i+1}" for i in range(len(z_mean))])
    plt.ylim(0, 1)
    if len(disagreements) > 0:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
