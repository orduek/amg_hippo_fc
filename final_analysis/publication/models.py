# Utility functions for the Bayes analysis
import numpy as np
import pymc3 as pm

def conditionModel(n_sub, n_cond, subIDX, condIDX, trials, df, observed):
    with pm.Model() as model:
    
        mu_a = pm.Normal('mu_a', mu=0, sigma=.5)
        sigma_a = pm.HalfCauchy('sigma_a', .5)
        a_matt = pm.Normal('a_matt', mu=0, sd=1, shape=n_sub)
        a = pm.Deterministic('a', mu_a + sigma_a*a_matt)

        mu_b = pm.Normal('mu_b', mu=0, sigma=.5)
        sigma_b = pm.HalfCauchy('sigma_b', .5)
        b_matt = pm.Normal('b_matt', mu=0, sigma=1, shape=n_cond)
        b_cond = pm.Deterministic('b_cond', mu_b + sigma_b*b_matt)

        # add trials and amygdala activation
        b_trials = pm.Normal('b_trials', mu=0, sigma=.5)
        b_amg = pm.Normal('b_amg', mu=0, sigma=.5)



        eps = pm.HalfCauchy('eps', .5)

        # make mu a determenistic distribution so we can test later vs. actual data

        mu = pm.Deterministic('mu',a[subIDX] + b_cond[condIDX] + b_trials*trials + b_amg*df.amg)

        y_hat = pm.Normal('y_hat', mu=mu, sigma=eps, observed=df[str(observed)])

        trace = pm.sample(target_accept=.95, chains=4, cores=4, return_inferencedata=True, draws=2000, tune=2000, random_seed=123)    
    return trace

def scrModel(n_sub, n_cond, subIDX, condIDX, trials, df, observed, coupling):
    
    with pm.Model() as model1_scr:

        mu_a = pm.Normal('mu_a', mu=0, sigma=.5)
        sigma_a = pm.HalfCauchy('sigma_a', .5)
        a_matt = pm.Normal('a_matt', mu=0, sd=1, shape=n_sub)
        a = pm.Deterministic('a', mu_a + sigma_a*a_matt)

        mu_b = pm.Normal('mu_b', mu=0, sigma=.5)
        sigma_b = pm.HalfCauchy('sigma_b', .5)
        b_matt = pm.Normal('b_matt', mu=0, sigma=1, shape=n_cond)
        b_cond = pm.Deterministic('b_cond', mu_b + sigma_b*b_matt)

        # add trials and amygdala activation
        b_trials = pm.Normal('b_trials', mu=0, sigma=.5)
        b_amg = pm.Normal('b_amg', mu=0, sigma=.5)

        # add coupling
        b_coup = pm.Normal('b_coup', mu=0, sigma=.5)



        eps = pm.HalfCauchy('eps', .5)

        # make mu a determenistic distribution so we can test later vs. actual data

        mu = pm.Deterministic('mu',a[subIDX] + b_cond[condIDX] + b_trials*trials + b_amg*df.amg + b_coup*df[str(coupling)])

        y_hat = pm.Normal('y_hat', mu=mu, sigma=eps, observed=df[str(observed)])


        trace_scr = pm.sample(target_accept=.95, chains=4, cores=10, return_inferencedata=True, draws=2000, tune=2000)    
    return trace_scr


def peModel(n_sub, n_cond, subIDX, condIDX, trials, df, coupling):
    with pm.Model() as model_pe:
        # adding intercept
        intercept = pm.Normal('intercept', mu=0, sigma=1)

        mu_a = pm.Normal('mu_a', mu=0, sigma=.5)
        sigma_a = pm.HalfCauchy('sigma_a', .5)
        a_matt = pm.Normal('a_matt', mu=0, sd=1, shape=n_sub)
        a = pm.Deterministic('a', mu_a + sigma_a*a_matt)


        # add trials and amygdala activation
        b_trials = pm.Normal('b_trials', mu=0, sigma=.5)
        b_amg = pm.Normal('b_amg', mu=0, sigma=.5)

        # add coupling
        #b_coup = pm.Normal('b_coup', mu=0, sigma=.5)
        mu_coup = pm.Normal('mu_coup', mu=0, sigma=.5)
        sigma_coup = pm.HalfCauchy('sigma_coup', .5)
        coup_matt = pm.Normal('coup_matt', mu=0, sd=1, shape=n_sub)
        b_coup = pm.Deterministic('b_coup', mu_coup + sigma_coup*coup_matt)


        eps = pm.HalfCauchy('eps', .5)

        # make mu a determenistic distribution so we can test later vs. actual data

        mu = pm.Deterministic('mu',intercept + a[subIDX] + b_trials*trials + b_amg*df.amg + b_coup[subIDX]*df[str(coupling)])

        y_hat = pm.Normal('y_hat', mu=mu, sigma=eps, observed=df.pe)


        trace_pe = pm.sample(target_accept=.95, chains=4, cores=10, return_inferencedata=True, draws=2000, tune=2000)
    return trace_pe