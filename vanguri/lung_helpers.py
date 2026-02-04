import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import copy
import sys


import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.colors as mcolors

from tqdm import tqdm

from lifelines.plotting import add_at_risk_counts


from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn import manifold
from scipy.interpolate import make_interp_spline
from scipy.stats.mstats import ttest_ind


from scipy.stats import spearmanr, pearsonr, kendalltau, zscore, maxwell, lognorm
import seaborn as sns
import matplotlib.pyplot as plt
import timeit

import lifelines
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import seaborn as sns
# sns.set_theme(style="whitegrid")
sns.set_style("ticks", {'axes.grid' : False, 'xtick.direction':'out', 'ytick.direction':'in'})

import numpy as np
import scipy.stats
from scipy import stats

from statannotations.Annotator import Annotator


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


import pyarrow.parquet as pq

FONT_BASE = {'family' : 'sans-serif',
        'sans-serif':'helvetica',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **FONT_BASE)
plt.rc('axes', unicode_minus=False)

CSS4_COLORS = mcolors.CSS4_COLORS

RAD_JOB_TAG = 'filtered-radiomics'


"""
Created on Tue Nov  6 10:06:52 2018

@author: yandexdataschool

Original Code found in:
https://github.com/yandexdataschool/roc_comparison

updated: Raul Sanchez-Vazquez
"""
# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2

def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)

def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)

def compute_ground_truth_statistics(ground_truth, sample_weight):
    ground_truth = np.array(ground_truth)
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight

def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def auc_roc_ci(y_true, y_pred, alpha):
    """
    Return AUC and CL at alpha for vectors y_true, y_pred
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    auc, auc_cov = delong_roc_variance(
        y_true,
        y_pred)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1

    return auc, ci

def find_optimal_cutoff(target, predicted):
    """ 
    Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def compute_jk(n, q, z):
    j = int (n*q - np.sqrt(n*q*(1-q)))
    k = int (n*q + np.sqrt(n*q*(1-q)))
    i = int (n*q)
    
    return [j,i,k]
    
def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def get_clinical_table(path, main_index_col):
    df = pd.read_csv(path)
    df['main_index'] = df[main_index_col].astype(str)
    df = df.set_index('main_index')

    df['halo_tumor_quality'] = df['halo_tumor_quality'].astype(float)

    df.loc  [ df['BOR'] == 'SD', 'label' ] = 1
    df.loc  [ df['BOR'] == 'SD (pseudoprogression)', 'label' ] = 1
    df.loc  [ df['BOR'] == 'POD','label' ] = 1

    df.loc  [ df['BOR'] == 'POD/brain',    'label' ] = 1
    df.loc  [ df['BOR'] == 'POD/bone',     'label' ] = 1
    df.loc  [ df['BOR'] == 'POD/death',    'label' ] = 1
    df.loc  [ df['BOR'] == 'POD/clinical', 'label' ] = 1

    df.loc  [ df['BOR'] == 'PR', 'label' ] = 0
    df.loc  [ df['BOR'] == 'CR', 'label' ] = 0

    print (len(df), sum(df['label'] == 0 ))
    return df

def get_clinical_table_v2(path, main_index_col, cohort):
    df = pd.read_csv(path, converters = {'did_acc':str})
    df['main_index'] = df[main_index_col].astype(str)
    df = df.set_index('main_index')
    df = df.loc[cohort.index.dropna()]

    df['halo_tumor_quality'] = df['halo_tumor_quality'].astype(float, errors='ignore')

    df.loc [~df['pack_years'].str.isnumeric(), 'pack_years'] = 0.0

    df.loc  [ :, 'label' ] = 1
    df.loc  [ df['bor'] == 1, 'label' ] = 0
    df.loc  [ df['bor'] == 2, 'label' ] = 0
   
    print (len(df), sum(df['label'] == 0 ))
    return df

def get_id_table():
    df = pd.read_csv("/gpfs/mskmind_emc/data_user/aukermaa/LUNG_DATA/id_inventory_v3.csv")
    df.set_index("hobbit_id")
    print (len(df))
    return df

def get_pdl1_table(id_table):
    df = \
    pd.read_csv("/gpfs/mskmind_emc/data_user/aukermaa/LUNG_DATA/slide_inventory_v3.csv")

    df.loc[~df['Sauter PD-L1 Score'].str.isnumeric(), 'Sauter PD-L1 Score'] = np.nan
    df = df.set_index("Hobbit ID").join(id_table.set_index("hobbit_id"))
    df['main_index'] = df['dmp_pt_id']
    df = df.set_index("main_index")[['Sauter PD-L1 Score']].dropna()
    print (len(df))
    return df

def get_omnibus():
    omnibus = pd.read_csv('/home/aukermaa/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540.csv', dtype={'did_acc': str, 'pdl1_image_id': str, 'record_id':int})
    omnibus['main_index'] = omnibus['record_id'].apply(lambda x: f"R-{x}")

    df = omnibus.set_index('main_index')

    # df = df.loc[cohort.index]

    df.loc [~df['pack_years'].str.isnumeric(), 'pack_years'] = 0.0

    df.loc  [ :, 'label' ] = 1
    df.loc  [ df['bor'] == 1, 'label' ] = 0
    df.loc  [ df['bor'] == 2, 'label' ] = 0
   
    print (len(df), sum(df['label'] == 0 ))
    return df

def get_genomic_table():
    df = pd.read_csv("/gpfs/mskmind_emc/data_user/aukermaa/LUNG_DATA/genomic_inventory_v3.csv")
    df['main_index'] = df['Patient ID']
    df = df.set_index('main_index')
    df = df[df.columns[df.columns.str.contains("bin|TMB")]].astype(int)
    print (len(df))
    return df


def get_textures_table(table_names=[]):
    tables = []

    for name in table_names:
        tables.append (
            pq.ParquetDataset(f"/gpfs/mskmind_emc/data_user/aukermaa/data_dev/{name}/STAIN_GLCM_DS/") \
                .read().to_pandas().reset_index().set_index('main_index')
        )

    df = pd.concat(tables, axis=1)
    print (len(df))
    return df    

def get_textures_table(table_names=[]):
    tables = []

    for name in table_names:
        tables.append (
            pq.ParquetDataset(f"/gpfs/mskmind_emc/data_user/aukermaa/data_dev/{name}/STAIN_GLCM_DS/") \
                .read().to_pandas().reset_index().set_index('main_index')
        )


    df = pd.concat(tables, axis=1)
    print (len(df))
    return df

    
def get_radiology_table(table_name):
    df = \
    pq.ParquetDataset(f"/gpfs/mskmindhdp_emc/data_dev/{table_name}/RECIST_RADIOMICS_DS") \
    .read().to_pandas().reset_index().set_index('main_index')
    print (len(df))
    return df


def decorate_with_site_index(df):
    df = df.reset_index().set_index(['main_index', 'job_tag', 'lesion_index'])
    df_main_index = df.reset_index()
    data_table_RF_PA = df_main_index[ (df_main_index['lesion_index'] == 1) | (df_main_index['lesion_index'] == 2)]
    data_table_RF_PL = df_main_index[ (df_main_index['lesion_index'] == 3) | (df_main_index['lesion_index'] == 4)]
    data_table_RF_LN = df_main_index[ (df_main_index['lesion_index'] == 5) | (df_main_index['lesion_index'] == 6)]

    data_table_RF_PA.loc[:,'site'] = 'PC'
    data_table_RF_PL.loc[:,'site'] = 'PL'
    data_table_RF_LN.loc[:,'site'] = 'LN'

    df_sites = pd.concat([data_table_RF_PA, data_table_RF_PL, data_table_RF_LN]).reset_index().set_index(['main_index', 'job_tag', 'site'])
    df_sites = df_sites.drop(columns="index")
    return df_sites


def prepare_rad_modality(df_dict, df, modality_mask, sites, l_idx, name):

    df_site  = df[df.index.isin(sites, level='site')]
    modality_site_full = df_site.copy(deep=True).reset_index()

    modality_site = modality_site_full[(modality_site_full['job_tag']==RAD_JOB_TAG) & (modality_site_full['lesion_index']==l_idx)]
    modality_mask.loc[modality_site.set_index('main_index').index, name] = True
    
    modality_site = modality_site \
        .set_index(['main_index', "lesion_index"])\
        .groupby(level=[0]).agg(np.mean) \
        .join(modality_mask[name], how='right').drop(columns=name)

    modality_site_full = modality_site_full.set_index('main_index')
    df_dict[name] = modality_site

    return modality_site_full


def prepare_rad_modality_by_size(df_dict, df, modality_mask, sites, name, sort='lesion_index', ascending=True, reduce=False):
    # 1. Converti sites in lista
    if type(sites)==str: sites=[sites]
    
    # 2. Filtra per site anatomico (PC, PL, LN)
    df_site = df[df.index.isin(sites, level='site')]
    modality_site_full = df_site.copy(deep=True).reset_index()
    
    # 3. Filtra solo job_tag radiomico
    modality_site = modality_site_full[(modality_site_full['job_tag']==RAD_JOB_TAG)]
    
    # 4. Marca pazienti con questa modalità disponibile
    modality_mask.loc[modality_site.set_index('main_index').index, name] = True
    
    # 5. Seleziona UNA lesione per paziente (sort + drop_duplicates)
    # poi fa right join per includere TUTTI i pazienti (con NaN se manca modalità)
    modality_site = modality_site \
        .sort_values(sort, ascending=ascending)\
        .drop_duplicates(subset=['main_index'])\
        .set_index('main_index')\
        .join(modality_mask[name], how='right').drop(columns=name)
    
    # 6. Debug: conta lesioni
    print(np.unique(modality_site['lesion_index'].dropna(), return_counts=True))
    modality_site = modality_site.drop(columns='lesion_index')
    
    # 7. Salva in dict
    modality_site_full = modality_site_full.set_index('main_index')
    df_dict[name] = modality_site
    
    return modality_site_full  # versione completa (con tutte le lesioni)


def prepare_other_modalities(df_dict, df, modality_mask, name):
    # Set pathology
    df = df.copy(deep=True).reset_index().set_index('main_index')
    modality_mask.loc[df.index, name] = True
    df = df.astype(float)
    df = df.join(modality_mask[name], how='right').drop(columns=name)
    df_dict[name] = df

# this function is used aionlab      
def select_outlier_stable_features(df, cutoff):
    df = df.reset_index()
    df = df[df['job_tag']==RAD_JOB_TAG]

    x = df.copy(deep=True).drop(columns=["main_index", "job_tag", "site", "index"], errors='ignore')

    x[(np.abs(zscore(x.values.astype(float))) > cutoff)] = np.nan
        
    outlier_counts = x.isna().sum()
        
    selected_features = outlier_counts[outlier_counts == 0].index
    return outlier_counts, selected_features

# outlier_counts, sel_fx_outlier = select_outlier_stable_features(rad_data_table_1mm_OWGL, 8)

def select_by_interlesion_variance(df, cutoff=0.1, plot=False):
    """ Look at perbuations intra-vs-inter lesion varaince, either the mean of stds, or the std of means """
    df = df.reset_index().set_index(['main_index', 'lesion_index'])
    df = df[df['job_tag']=='pertubation-radiomics']
    interlesion_variance = df.groupby(level=[0,1]).agg(np.std).mean() / df.groupby(level=[0,1]).agg(np.mean).std()
    cols = interlesion_variance[interlesion_variance < cutoff].index
    variance_ranks = interlesion_variance.rank(method='min').rename("interlesion_variance_rank")
    if plot:
        plt.hist(interlesion_variance, bins=20)
        plt.xlabel("Mean of intra-lesion variance / variance of intra-lesion means")
        plt.ylabel("Counts")
        # plt.savefig("eda/pertubation-intra-inter-variances.pdf")
    return interlesion_variance, cols

# variance, sel_fx_robust = select_by_interlesion_variance(rad_data_table_1mm_OWGL, cutoff=0.25)

# this function is used aionlab      
def select_radiomics_features_elastic(train_df, outcomes_df, robustness_cutoff=0.15, outlier_cutoff=6, l1_strength=0.1):
    outlier_counts,  sel_fx_outlier = select_outlier_stable_features(train_df, cutoff=outlier_cutoff)
    print(f"Selected {len(sel_fx_outlier)} outlier-stable features")
    # variance_scores, sel_fx_robust  = select_by_interlesion_variance(train_df, cutoff=robustness_cutoff) we don't have multiple lesions per patient
    fx_to_use = sorted(set(sel_fx_outlier))

    fx_to_use_with_index = ['main_index'] + fx_to_use
    train_df = train_df[train_df['job_tag'] == RAD_JOB_TAG]
    train_df = train_df.drop(columns=['job_tag', 'site'])[fx_to_use_with_index]

    train_df = train_df.set_index('main_index')

    Y_train  = outcomes_df.loc[train_df.index, 'label']

    scaler  = PowerTransformer()        
    X_train = scaler.fit_transform(train_df)
    
    clf_lr =  LogisticRegression(penalty='elasticnet', random_state=0, max_iter=2500, solver='saga', l1_ratio=0.5, C=l1_strength, class_weight='balanced')
    clf_lr.fit(X_train, Y_train)
        
    l1_features = train_df.columns[np.where(clf_lr.coef_[0])[0]]

    df_coef = pd.DataFrame(clf_lr.coef_[0], index=train_df.columns, columns=['coef']).abs().sort_values(by='coef', ascending=False)

    return l1_features, df_coef


def select_standard_features_elastic(train_df, outcomes_df):
    
    train_df = train_df.dropna()
    
    Y_train  = outcomes_df.loc[train_df.index, 'label']

    scaler  = StandardScaler()        
    X_train = scaler.fit_transform(train_df)
    
    clf_lr =  LogisticRegression(penalty='elasticnet', max_iter=2500, solver='saga', l1_ratio=0.5, C=0.1, class_weight='balanced')
    clf_lr.fit(X_train, Y_train)
        
    l1_features = train_df.columns[np.where(clf_lr.coef_[0])[0]]
    
    return l1_features


class MultiLesionModel(nn.Module):
    """
    Impliment fit and eval methods for multi
    """
    def __init__(self, input_size, hidden_size = 8, alpha=0.0, beta=0.0, lr=0.01, steps=100, class_weight=None):
        super(MultiLesionModel, self).__init__()
        torch.manual_seed(42)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.class_weight = class_weight
        self.steps = steps
        self.lr = lr

    def response_zscore(self, input, target=None):
        if self.training:
            threshold = find_optimal_cutoff(target.detach(), input.detach())
            threshold = torch.tensor(threshold)
            # print (classification_report(target.long().detach(), (input > threshold).long().detach()))
            
            self.mu  = threshold
            self.std = input.std(dim=0)
        else:
            pass

        return (input - self.mu) / self.std
        
    def fit(self, df_data, df_labels):
        self.train()

        # Fit scaler
        self.scaler  = RobustScaler()
        self.scaler.fit(df_data)

        data_inputs  = {}
        data_labels  = {}

        batch_index = df_labels.index
        n_batch = len(batch_index)

        for px in batch_index:
            data_inputs[px]  = torch.tensor(self.scaler.transform(df_data.loc[[px]])).float()
            data_labels[px]  = torch.tensor(df_labels.loc[px, 'label']).float()

        label_vector = df_labels['label'].values
        pos_class_weight = sum(label_vector==0) / sum(label_vector==1)

        self.milr = MILR(input_size=self.input_size, hidden_size=self.hidden_size)
        self.criterion  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_class_weight))
        self.optimizer  = torch.optim.Adam(self.parameters(), lr=self.lr)   

        batch_index = batch_index.sort_values() # Needed to prevent numerical instability issues

        for i in range(self.steps):

            agg_loss = 0.

            self.optimizer.zero_grad()

            for px in batch_index:
                logit, attn_scores, risk_scores, embedding = self.milr(data_inputs[px])

                loss = self.criterion(logit.view(1,1), data_labels[px].view(1,1)) / n_batch

                loss.backward()
                
                agg_loss += loss.item()

            l2 = self.alpha * self.milr.get_l2_weight_sum()
            l2.backward()       

            self.optimizer.step()

        targets = []
        outputs = []
        for px in batch_index:
            logit, attn_scores, risk_scores, embedding = self.milr(data_inputs[px])
            outputs.append(logit)
            targets.append(data_labels[px])

        return self.response_zscore(torch.stack(outputs), torch.stack(targets)).detach().numpy()

    def predict_proba(self, df_data):
        self.eval()
       
        batch_index = df_data.index.unique()
        outputs = []

        for px in batch_index:
            data_input  = torch.tensor(self.scaler.transform(df_data.loc[[px]])).float()
            logit, attn_scores, risk_scores, embedding = self.milr(data_input)
            outputs.append(logit)

        calib_output = self.response_zscore(torch.stack(outputs)).detach().numpy()
        
        return pd.DataFrame(calib_output, index=batch_index, columns=['score'])


class MILR(nn.Module):
    """
    Impliment fit and 
    """
    def __init__(self, input_size, hidden_size = 8):
        super(MILR, self).__init__()
        torch.manual_seed(42)

        self.L = input_size    # Input features to attention mechanism
        self.H = hidden_size
        
        self.attn = nn.Linear(self.L, 1, bias=False) # Softmax is invarient to a bias term 
        self.rfct = nn.Linear(self.L, self.H, bias=True)
        
        self.intr = nn.Linear(self.H, 1, bias=True)
 
        self.sm   = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

        # self.reset_params()

    @staticmethod
    def weight_init(m, name=''):
        if isinstance(m, nn.Linear) and name in ['attn', 'rfct']:
            # Using tanh with fan in
            print (name + " with init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='sigmoid')")
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='sigmoid')
            if m.bias is not None: init.zeros_(m.bias)
            
    def reset_params(self):
        for name, m in self.named_modules():
            self.weight_init(m, name)

    def get_l2_weight_sum(self):
        return  torch.stack([p.norm(p=2) for n, p in self.named_parameters() if 'weight' in n]).sum() 
        
    def forward(self, input: torch.Tensor):
        "Takes input instances (batch dimension) and returns logit, Aterm, Cterm, and Eterm"
        Cterm = self.rfct(input)
        Aterm = self.attn(input)
        Eterm = torch.mm( self.sm(Aterm).transpose(1,0), self.tanh(Cterm)).view(self.H) 
        logit = self.intr(Eterm)

        # AdjRisk = Aterm.exp() * (Cterm) - self.rfct.bias

                

        return logit, Aterm, Cterm, Eterm


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super (SimpleMLP, self).__init__()
        self.input_dim = input_dim
        
        if self.input_dim > 4:
            self.lin1 = nn.Linear(input_dim, input_dim // 2) 
            self.relu = nn.LeakyReLU()
            self.lin2 = nn.Linear(input_dim // 2, output_dim) 
        else:
            self.lin1 = nn.Linear(input_dim, output_dim) 
            
    def forward(self, input):
        if self.input_dim > 4:
            return self.lin2(self.relu(self.lin1(input)))
        else:
            return self.lin1(input)

class MaskedMultiModalLoader(Dataset):
    def __init__(self, list_X_inputs, mask_tensor, labels_tensor):
        self.list_X_inputs = list_X_inputs
        self.mask_tensor   = mask_tensor
        self.labels_tensor = labels_tensor
        
        assert len(set([x.shape[0] for x in list_X_inputs])) == 1
        assert list_X_inputs[0].shape[0] == mask_tensor.shape[0]
        assert list_X_inputs[0].shape[0] == labels_tensor.shape[0]
        self.nsamples =  list_X_inputs[0].shape[0]
        # print ("Total samples:", self.nsamples)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        list_X_batch = [x[idx] for x in self.list_X_inputs]
        mask_batch   = self.mask_tensor[idx]
        label_batch  = self.labels_tensor[idx]
        return list_X_batch, mask_batch, label_batch
            
        

class AttentionMatrix(nn.Module):
    def __init__(self, cross_modality_enabled=False, attention_gate_enabled=True):
        super(AttentionMatrix, self).__init__()
        torch.manual_seed(42)

        self.input_channel_shapes = []
        self.softplus = nn.Softplus()
        self.sigmoid  = nn.Sigmoid()
        self.tanh     = nn.Tanh()
        self.cross_modality_enabled = cross_modality_enabled
        self.attention_gate_enabled = attention_gate_enabled
        
        self.temp = nn.Parameter(torch.tensor([1.5]))
        
    def add_channel(self, channel_template):
        """ 
        Add a channel of shape channel_template
        """
        self.input_channel_shapes.append (channel_template.shape[1])
        self.n_input_channels = len(self.input_channel_shapes)
        
    def apply_temperature(self, input):
        return input
        
#         return self.sigmoid(-self.temp) * input + self.sigmoid(self.temp)

    def setup_matrix(self):
        """ 
        Creates two module lists:
            l_risk_linears of length n_channels, which predict the modality-specific risk score
            l_attn_linears of length n_channels^2, which predict the cross modality attention scores
        """
        self.l_attn_linears   = nn.ModuleList()
        self.l_risk_linears   = nn.ModuleList()
        self.l_feature_factor = []
        for channel_shape in self.input_channel_shapes:
            self.l_risk_linears.append   ( nn.Linear(channel_shape, 1) )
            self.l_feature_factor.append ( torch.tensor([channel_shape]) )
            for j in range(self.n_input_channels):
                self.l_attn_linears.append( nn.Linear(channel_shape, 1) )
        
#         Start with zero bias for simplicity
        for m in self.l_attn_linears:  torch.nn.init.zeros_(m.bias)
        for m in self.l_risk_linears:  torch.nn.init.zeros_(m.bias)
            
        # We'll need to reshape our linear list into a square matrix
        self.reshape_tuple = (-1, self.n_input_channels, self.n_input_channels)
    
    def get_l2_weight_sum(self):
        return  torch.stack([p.norm(p=2) for n, p in self.named_parameters() if 'weight' in n]).sum() 
        
    def forward(self, inputs, mask):
        attn_reduced = []
        risk_reduced = []
        
        matrix_index = 0
        
        # Linear mask
        linear_mask  = mask
        
        # Cleverly broadcast linear mask into matrix mask
        matrix_mask  = mask.reshape(-1, self.n_input_channels, 1).expand(-1, -1, self.n_input_channels)

        # Change modal mask depending on cross-modality weighting flag
        if not self.cross_modality_enabled:
            
            # Mask away non-diagnoanl elements i.e., no cross-modality attention weighting
            identity_mask = torch.eye(self.n_input_channels)\
                .reshape(1, self.n_input_channels, self.n_input_channels)\
                .repeat(matrix_mask.shape[0], 1, 1)

            # Multiply masks
            matrix_mask = identity_mask * matrix_mask
        
        # Outer-loop: over input channels/inputs
        for channel_index, input_channel in enumerate(inputs):
            
            # Calculate the risk (once per modality)
            risk_reduced.append ( self.l_risk_linears[channel_index](input_channel) )
            
            # Inner-loop: over input channels/inputs again to predict attention scores for other modalities
            for j in range(self.n_input_channels):
                
                # Calculate the attention weight (once per modality also)
                attn_reduced.append ( self.l_attn_linears[matrix_index](input_channel) / self.l_feature_factor[channel_index] )
                
                # Keep track of our matrix index
                matrix_index += 1
                
        risk_scores  = torch.cat( risk_reduced, axis=1 ) # Turn list of scores into 2D tensor, mask out missing risk scores
        risk_weights = linear_mask * self.tanh ( risk_scores ) # r_i = tanh(R)
        
        attn_matrix = torch.cat( attn_reduced, axis=1 ).reshape(self.reshape_tuple) # Turn list of scores into B x n x n 3D matrix,
        attn_matrix = matrix_mask * self.softplus ( attn_matrix ) # Activate weights, mask away missing modalities so they don't count for other modalities weights
        attn_scores = linear_mask * attn_matrix.sum(dim=1) # Compute the modality specific attention scores, and mask away missing modalities again so they don't contribute to the normalization
        attn_weight = F.normalize(attn_scores, p=1) # Attention a_i = score_i / sum(score_i)
            
#         if not self.training:
#             heatmap = attn_matrix.detach().numpy()[0]
#             heatmap[heatmap==0] = np.nan
#             sns.heatmap(heatmap, cmap='vlag')
#             plt.show()
        
        attn_norm   = attn_matrix.norm(p=2, dim=(1,2)).mean() # Compute the attention weight norm (activation scale)
        risk_norm   = risk_scores.norm(p=2, dim=( 1 )).mean() # Compute the risk weight norm (activation scale)

        if self.attention_gate_enabled: 
            total_risk = torch.sum(risk_weights * attn_weight, dim=1) # Total Risk = sum (r_i * a_i)
        else:
             total_risk = torch.sum(risk_weights, dim=1)

        return total_risk, risk_weights, attn_weight, attn_norm, risk_norm


class MultiModalDynamicModel(nn.Module):
    def __init__(self, epochs=100, alpha=1.0, beta=1.0, lr=0.01, hidden_factor = 2, class_weight='balanced', print_on=50, cross_modality_enabled=False, no_scale=[], attention_gate_enabled=True):
        super(MultiModalDynamicModel, self).__init__()
        self.epochs  = epochs
        self.alpha   = alpha
        self.beta    = beta
        self.lr      = lr
        self.class_weight  = class_weight
        self.print_on = print_on
        self.cross_modality_enabled = cross_modality_enabled
        self.noscale=no_scale
        self.attention_gate_enabled = attention_gate_enabled
    
    def response_zscore(self, input, target=None):

        if self.training:
            threshold = find_optimal_cutoff(target.detach(), input.detach())
            threshold = torch.tensor(threshold)
            
            self.mu  = threshold
            self.std = input.std(dim=0)
            print(f"Training mode - mu: {self.mu.item():.4f}, std: {self.std.item():.4f}")
        else:
            print(f"Eval mode - using mu: {self.mu.item():.4f}, std: {self.std.item():.4f}")
            pass

        return (input - self.mu) / self.std
        
    def fit(self, l_X_INPUTS, arr_MASK, vector_Y):
        self.train()
        
        l_X_INPUTS = copy.deepcopy(l_X_INPUTS)
        arr_MASK   = copy.deepcopy(arr_MASK)
        vector_Y   = copy.deepcopy(vector_Y)
        
        print(f"Fit called with {len(l_X_INPUTS)} modalities, {len(vector_Y)} samples")
                        
        self.dyam = AttentionMatrix(cross_modality_enabled=self.cross_modality_enabled, attention_gate_enabled=self.attention_gate_enabled)
        self.l_scalers = []
        
        for i, X in enumerate(l_X_INPUTS):
            print(f"Modality {i}: shape {X.shape}, noscale={i in self.noscale}")
            
            if i in self.noscale: 
                l_X_INPUTS[i] = torch.tensor(np.nan_to_num(X)).float()
            else:
                scaler  = RobustScaler()
                l_X_INPUTS[i] = torch.tensor(np.nan_to_num(scaler.fit_transform(X))).float()
                self.l_scalers.append ( scaler )
            self.dyam.add_channel (X)

            
        self.dyam.setup_matrix()
                
        mask    = torch.tensor(arr_MASK).float()
        targets = torch.tensor(vector_Y).float()
        
        print(f"Mask shape: {mask.shape}, available modalities per patient: {mask.sum(dim=1).mean().item():.2f}")
        print(f"Class balance: {targets.sum().item()}/{len(targets)} positive")

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=sum(targets==0) / sum(targets==1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        dataset = MaskedMultiModalLoader (l_X_INPUTS, mask, targets)
        loader  = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False )
        n_batch = len(loader)
        
        print(f"Training {self.epochs} epochs with {n_batch} batches")
        
        for i in range(self.epochs + 1):
            for b_inputs, b_mask, b_labels in loader:
                self.optimizer.zero_grad()

                output, risk_scores, mixing_matrix, ar2, rr2 = self.dyam(b_inputs, b_mask)

                loss   = self.criterion(output, b_labels)       
                l2     = self.dyam.get_l2_weight_sum()

                # DEBUG: if (i % self.print_on)==0: print(f"Epoch {i}, Loss: {loss.item():.4f}, L2: {l2.item():.4f}, AR2: {ar2.item():.4f}")

                total_loss = (loss + (self.alpha * l2) + (self.beta * ar2)) / n_batch
                total_loss.backward()

                self.optimizer.step()
            
        # Final fit
        output, risk_scores, mixing_matrix, ar2, rr2 = self.dyam(l_X_INPUTS, mask)
        
        print(f"Final output - min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}")

        return self.response_zscore(output, targets).detach().numpy()

    def get_coefs(self, modality_list, modality_mask):
        coefs = torch.cat([x.weight.flatten() for x in self.dyam.l_risk_linears], dim=0).flatten().detach().numpy()

        columns = []
        for i, df in enumerate(modality_list): columns.extend(df.add_prefix("risk___" + modality_mask.columns[i] + '__').columns)

        df_coef = pd.DataFrame([coefs, np.sign(coefs)], columns=columns, index=['coef', 'sign'])

        return df_coef



    def predict_proba(self, l_X_INPUTS, arr_MASK):
        self.eval()
        
        l_X_INPUTS = copy.deepcopy(l_X_INPUTS)
        arr_MASK   = copy.deepcopy(arr_MASK)
        
        for i, X in enumerate(l_X_INPUTS):
            if i in self.noscale: 
                l_X_INPUTS[i] = torch.tensor(np.nan_to_num(X)).float()
            else:
                l_X_INPUTS[i] = torch.tensor(np.nan_to_num(self.l_scalers[i].transform(X))).float()
                    
        mask = torch.tensor(arr_MASK).float()

        output, risk_scores, mixing_matrix, ar2, rr2 = self.dyam(l_X_INPUTS, mask)
        
        return self.response_zscore(output).detach().numpy()
    
    def get_summary_scores(self, l_X_INPUTS, arr_MASK):
        self.eval()
        
        l_X_INPUTS = copy.deepcopy(l_X_INPUTS)
        arr_MASK   = copy.deepcopy(arr_MASK)
        
        for i, X in enumerate(l_X_INPUTS):
            if i in self.noscale: 
                l_X_INPUTS[i] = torch.tensor(np.nan_to_num(X)).float()
            else:
                l_X_INPUTS[i] = torch.tensor(np.nan_to_num(self.l_scalers[i].transform(X))).float()
                    
        mask = torch.tensor(arr_MASK).float()
        
        share_factor = mask.sum(dim=1, keepdim=True)

        output, risk_scores, mixing_matrix, ar2, rr2 = self.dyam(l_X_INPUTS, mask)
        
#         print (self.dyam.l_risk_linears[5].weight, self.dyam.l_risk_linears[5].bias)
#         print (self.dyam.l_attn_linears[35].weight, self.dyam.l_attn_linears[35].bias)
        
        attention_share = share_factor * mixing_matrix
        
#         print ("Validation risk scores:", risk_scores.detach().numpy())
#         print ("Validation attention:  ", attention_share.detach().numpy())
        
        return output.detach().numpy(), risk_scores.detach().numpy(), mixing_matrix.detach().numpy(), attention_share.detach().numpy()

def get_summary_df(d_summarys):
    df = pd.DataFrame(d_summarys).T.dropna()
    df['score_norm'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
    df['error'] = abs(df['label'] - df['score_norm'])
    df = df.sort_values(['label', 'score'])
    return df


def generate_panel_figure(df):

    print (len(df), df.columns)

    print (classification_report(df['label'], df['score'] > 0))

    for fx in df.columns[df.columns.str.contains('risk')]:

        auc, ci = auc_roc_ci((df['label']), df[fx], 0.95)
        print ("Post-fit [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL".format(fx, auc, (ci[1] - ci[0]) / 2.0) )

    plt.figure(figsize=(15, 20))

    low_err  = df['error'] <  0.5
    high_err = df['error'] >= 0.5

    print (sum(low_err), sum(high_err))

    plt.subplot(1,3,1)

    sns.heatmap(df.loc[:, ['score','label']], cmap="vlag", center=0)
    plt.subplot(1,3,2)
    sns.heatmap(df.loc[:, df.columns.str.contains('risk')], cmap="vlag")
    plt.subplot(1,3,3)

    df[df==0] = np.nan
    sns.heatmap(df.loc[:, df.columns.str.contains('attn')])
    plt.tight_layout()

def generate_lifelines_qcut(df, df_clinical, col='score', custom_bins=None, swap=False, panel=None):
    print ("Running quartile KM fit on", col)

    df_clinical =  df_clinical.loc[df.index]

    plt.figure(figsize=(8,8))

    if custom_bins:
        q1 = df[col].between(custom_bins[0], custom_bins[1])
        q2 = df[col].between(custom_bins[1], custom_bins[2])
        q3 = df[col].between(custom_bins[2], custom_bins[3])
        q4 = df[col].between(custom_bins[3], custom_bins[4])

    else:
        q1 = pd.qcut(df[col], 4, labels=False) == 0
        q2 = pd.qcut(df[col], 4, labels=False) == 1
        q3 = pd.qcut(df[col], 4, labels=False) == 2
        q4 = pd.qcut(df[col], 4, labels=False) == 3

    if swap:
        q1, q2, q3, q4 = q4, q3, q2, q1

    results = logrank_test(
        df_clinical.loc[q1,'pfs'], 
        df_clinical.loc[q4,'pfs'], 
        event_observed_A=df_clinical.loc[q1,'pfs_censor'], 
        event_observed_B=df_clinical.loc[q4,'pfs_censor'])

    pvalue_q1_q4= (np.log10(results.p_value))

    results = logrank_test(
        df_clinical.loc[q1,'pfs'], 
        df_clinical.loc[q2,'pfs'], 
        event_observed_A=df_clinical.loc[q1,'pfs_censor'], 
        event_observed_B=df_clinical.loc[q2,'pfs_censor'])

    pvalue_q1_q2= (np.log10(results.p_value))

    results = logrank_test(
        df_clinical.loc[q3,'pfs'], 
        df_clinical.loc[q4,'pfs'], 
        event_observed_A=df_clinical.loc[q3,'pfs_censor'], 
        event_observed_B=df_clinical.loc[q4,'pfs_censor'])

    pvalue_q3_q4= (np.log10(results.p_value))

    results = logrank_test(
        df_clinical.loc[q2,'pfs'], 
        df_clinical.loc[q3,'pfs'], 
        event_observed_A=df_clinical.loc[q2,'pfs_censor'], 
        event_observed_B=df_clinical.loc[q3,'pfs_censor'])

    pvalue_q2_q3= (np.log10(results.p_value))

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(df_clinical.loc[q1, 'pfs'],  event_observed=df_clinical.loc[q1]['pfs_censor'], label="Q1")
    kmf_low.plot(show_censors=True, c=sns.color_palette("husl", 4)[0], linewidth=2.5, censor_styles={'ms': 5, 'marker': 's'})
    plt.xlim(-1,12)
    apply_style()
    
    kmf_q2 = KaplanMeierFitter()
    kmf_q2.fit(df_clinical.loc[q2, 'pfs'],  event_observed=df_clinical.loc[q2]['pfs_censor'], label="Q2")
    kmf_q2.plot(show_censors=True, c=sns.color_palette("husl", 4)[1], linewidth=2.5, censor_styles={'ms': 5, 'marker': 's'})
    plt.xlim(-1,12)
    apply_style()
        
    kmf_q3 = KaplanMeierFitter()
    kmf_q3.fit(df_clinical.loc[q3, 'pfs'], event_observed=df_clinical.loc[q3]['pfs_censor'], label="Q3")
    kmf_q3.plot(show_censors=True, c=sns.color_palette("husl", 4)[2], linewidth=2.5, censor_styles={'ms': 5, 'marker': 's'})
    plt.xlim(-1,12)
    apply_style()

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(df_clinical.loc[q4, 'pfs'], event_observed=df_clinical.loc[q4]['pfs_censor'], label="Q4")
    ax = kmf_high.plot(show_censors=True, c=sns.color_palette("husl", 4)[3], linewidth=2.5, censor_styles={'ms': 5, 'marker': 's'})
    plt.xlim(-1,12)
    apply_style()
    plt.xlabel('PFS (months)')

    df_summary = pd.concat(
        (kmf_low.event_table.add_prefix('Q1__'), 
        kmf_q2.event_table.add_prefix('Q2__'), 
        kmf_q3.event_table.add_prefix('Q3__'), 
        kmf_high.event_table.add_prefix('Q4__')), axis=1)
    if panel is not None:
        df_summary.to_excel(f'./excel/{panel}.xlsx', sheet_name=panel)

    from matplotlib.legend_handler import HandlerBase

    class AnyObjectHandlerQ1(HandlerBase):
        def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
            l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], linewidth=2.5, color=sns.color_palette("husl", 4)[0])
            l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], linewidth=2.5, color=sns.color_palette("husl", 4)[3])
            return [l1, l2]
    class AnyObjectHandlerQ2(HandlerBase):
        def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
            l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], linewidth=2.5, color=sns.color_palette("husl", 4)[0])
            l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], linewidth=2.5, color=sns.color_palette("husl", 4)[1])
            return [l1, l2]

    class AnyObjectHandlerQ3(HandlerBase):
        def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
            l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], linewidth=2.5, color=sns.color_palette("husl", 4)[1])
            l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], linewidth=2.5, color=sns.color_palette("husl", 4)[2])
            return [l1, l2]

    class AnyObjectHandlerQ4(HandlerBase):
        def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
            l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], linewidth=2.5, color=sns.color_palette("husl", 4)[2])
            l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], linewidth=2.5, color=sns.color_palette("husl", 4)[3])
            return [l1, l2]

    plt.legend([
            AnyObjectHandlerQ1, 
            AnyObjectHandlerQ2,
            AnyObjectHandlerQ3,
            AnyObjectHandlerQ4,
        ], 
        [
            rf'Q1 vs. Q4 log$_{{10}}(p) = {pvalue_q1_q4:.2f}$', 
            rf'Q1 vs. Q2 log$_{{10}}(p) = {pvalue_q1_q2:.2f}$',
            rf'Q2 vs. Q3 log$_{{10}}(p) = {pvalue_q2_q3:.2f}$',
            rf'Q3 vs. Q4 log$_{{10}}(p) = {pvalue_q3_q4:.2f}$',
        ], 
        handler_map={
            AnyObjectHandlerQ1: AnyObjectHandlerQ1(), 
            AnyObjectHandlerQ2: AnyObjectHandlerQ2(),
            AnyObjectHandlerQ3: AnyObjectHandlerQ3(),
            AnyObjectHandlerQ4: AnyObjectHandlerQ4(),
        },
        prop={'size': FONT_BASE['size']*.75},
        frameon=False
    )


    add_at_risk_counts(kmf_low, kmf_high, ax=ax)
    
    print (results)


def generate_lifelines_tcut(df, df_clinical, col='score'):

    df_clinical =  df_clinical.loc[df.index]

    plt.figure(figsize=(8, 8))
    q1 = pd.qcut(df[col], 3, labels=False) == 0
    q2 = pd.qcut(df[col], 3, labels=False) == 1
    q3 = pd.qcut(df[col], 3, labels=False) == 2

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(df_clinical.loc[q1, 'pfs'],  event_observed=df_clinical.loc[q1]['pfs_censor'], label="Q1")
    kmf_low.plot(show_censors=True, c=sns.color_palette("husl", 4)[0], censor_styles={'ms': 5, 'marker': 's'})
    plt.xlim(0,12)
    apply_style()
    
    kmf_q2 = KaplanMeierFitter()
    kmf_q2.fit(df_clinical.loc[q2, 'pfs'],  event_observed=df_clinical.loc[q2]['pfs_censor'], label="Q2")
    kmf_q2.plot(show_censors=True, c=sns.color_palette("husl", 4)[1], censor_styles={'ms': 5, 'marker': 's'})
    plt.xlim(0,12)
    apply_style()
        
    kmf_q3 = KaplanMeierFitter()
    kmf_q3.fit(df_clinical.loc[q3, 'pfs'], event_observed=df_clinical.loc[q3]['pfs_censor'], label="Q3")
    ax = kmf_q3.plot(show_censors=True, c=sns.color_palette("husl", 4)[2], censor_styles={'ms': 5, 'marker': 's'})
    plt.xlim(0,12)
    apply_style()

    plt.xlabel('PFS (months)')




    results = logrank_test(
        df_clinical.loc[q1,'pfs'], 
        df_clinical.loc[q4,'pfs'], 
        event_observed_A=df_clinical.loc[q1,'pfs_censor'], 
        event_observed_B=df_clinical.loc[q4,'pfs_censor'])

    print (results)

def generate_lifelines_binary(df, df_clinical, col='score', threshold=0, panel=None):
    print ("Running binary KM fit on", col)

    df_clinical =  df_clinical.loc[df.index]

    if threshold == 'median':
        threshold = df[col].median()
    if threshold == 'mean':
        threshold = df[col].mean()
    if threshold == 'auc':
         threshold = find_optimal_cutoff(df_clinical['label'], df[col])[0]

    print ("Threshold=", threshold)
    plt.figure(figsize=(8,8))
    q1 = df[col] >  threshold
    q2 = df[col] <= threshold


    kmf_low = KaplanMeierFitter()
    kmf_low.fit(df_clinical.loc[q1, 'pfs'],  event_observed=df_clinical.loc[q1]['pfs_censor'], label="Above Threshold")
    ax = kmf_low.plot(show_censors=True, c='#a6bddb', linewidth=2.5, censor_styles={'ms': 5, 'marker': 's'})
    apply_style()

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(df_clinical.loc[q2, 'pfs'],  event_observed=df_clinical.loc[q2]['pfs_censor'], label="Below Threshold")
    ax = kmf_high.plot(show_censors=True, c='#0570b0', linewidth=2.5, censor_styles={'ms': 5, 'marker': 's'})
    apply_style()

    df_summary = pd.concat((kmf_low.event_table.add_prefix('below_threshold__'), kmf_high.event_table.add_prefix('above_threshold__')), axis=1)
    if panel is not None:
        df_summary.to_excel(f'./excel/{panel}.xlsx', sheet_name=panel)

    plt.xlabel('PFS (months)')

    results = logrank_test(
        df_clinical.loc[q1,'pfs'], 
        df_clinical.loc[q2,'pfs'], 
        event_observed_A=df_clinical.loc[q1,'pfs_censor'], 
        event_observed_B=df_clinical.loc[q2,'pfs_censor'])

    pvalue_bin= (np.log10(results.p_value))


    from matplotlib.legend_handler import HandlerBase

    class AnyObjectHandlerBin(HandlerBase):
        def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
            l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], linewidth=2.5, color='#a6bddb')
            l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], linewidth=2.5, color='#0570b0')
            return [l1, l2]

    plt.legend([
            AnyObjectHandlerBin, 
        ], 
        [
            rf'Above/Below Thershold log$_{{10}}(p) = {pvalue_bin:.2f}$', 
        ], 
        handler_map={
            AnyObjectHandlerBin: AnyObjectHandlerBin(), 
        },
        prop={'size': FONT_BASE['size']*.75},
        frameon=False
    )
    print (results)
    add_at_risk_counts(kmf_low, kmf_high, ax=ax)

    df_clinical['pfs']

        
# this function is used aionlab      
def l1_filter_features_list(modality_list, df_input, df_outcome, exclude_index, position, robustness_cutoff=0.15, outlier_cutoff=6, l1_strength=1.0):
    features, df_coef = select_radiomics_features_elastic(df_input.drop(index=exclude_index, errors='ignore'), df_outcome, robustness_cutoff=robustness_cutoff, outlier_cutoff=outlier_cutoff, l1_strength=l1_strength) 
    modality_list[position] = modality_list[position][features]
    return df_coef

def l1_filter_features_df(df_modality, df_input, df_outcome, exclude_index, robustness_cutoff=0.15, outlier_cutoff=6, l1_strength=1.0):

    features, df_coef = select_radiomics_features_elastic(df_input.drop(index=exclude_index, errors='ignore'), df_outcome, robustness_cutoff=robustness_cutoff, outlier_cutoff=outlier_cutoff, l1_strength=l1_strength) 
    # print (df_coef)
    return df_modality[features]
    return df_coef

def generate_l1_plot(modality_list_in, modality_mask, outcomes, pos, filter, folds=10):

    kf = KFold(n_splits=folds, random_state=0, shuffle=True)
    for fold, (train, test) in enumerate(tqdm(list(kf.split(outcomes.index)), file=sys.stdout)):
        modality_list = [df.copy(deep=True) for df in modality_list_in]

        train_px = outcomes.index[train]
        valid_px  = outcomes.index[test]

        df = l1_filter_features_list(modality_list, filter['l1_selection_df'], outcomes, valid_px, pos, **filter['kwargs'])

# this function is used aionlab      
def train(modality_list_in, modality_mask, outcomes, l1_dfs_filter, model_params, folds=10, predefined_folds=None):
    l_v_scores = []
    l_v_labels = []
    d_summarys_all = {}

    print("\nStarting cross-validation training")
    print("predefined_folds:", predefined_folds is not None)

    if predefined_folds is not None:
        print("Using predefined folds for cross-validation")
        # Usa fold predefiniti
        fold_iterator = []
        for fold_name in predefined_folds['fold'].unique():
            test_mask = predefined_folds['fold'] == fold_name
            train_idx = outcomes.index[~outcomes.index.isin(predefined_folds[test_mask].index)]
            test_idx = outcomes.index[outcomes.index.isin(predefined_folds[test_mask].index)]
            train_folds = predefined_folds.loc[train_idx, 'fold'].unique()
            fold_iterator.append((train_idx, test_idx))
            print(f"Fold {fold_name} (test): {len(test_idx)} samples | Training on folds: {train_folds} ({len(train_idx)} samples)")

    else:
        # Usa KFold standard
        if folds == 'LOO':
            folds = len(outcomes.index)
        kf = KFold(n_splits=folds, random_state=0, shuffle=True)
        fold_iterator = [(outcomes.index[train], outcomes.index[test]) for train, test in kf.split(outcomes.index)]

    df_coef_agg = pd.DataFrame()

    for fold, (train_px, valid_px) in enumerate(tqdm(fold_iterator, file=sys.stdout)):
        print(f"\nProcessing fold {fold+1}: {len(train_px)} train samples, {len(valid_px)} validation samples")
        modality_list = [df.copy(deep=True) for df in modality_list_in]
        

        for pos, filter in l1_dfs_filter.items():
            print(" Applying L1 filter on modality position", pos)
            l1_filter_features_list(modality_list, filter['l1_selection_df'], outcomes, valid_px, pos, **filter['kwargs'])

        train_feature_inputs = [df.loc[train_px].values for df in modality_list]
        valid_feature_inputs = [df.loc[valid_px].values for df in modality_list]

        train_feature_mask   = modality_mask.loc[train_px].astype(int).values
        valid_feature_mask   = modality_mask.loc[valid_px].astype(int).values
        
        train_labels = outcomes.loc[train_px, 'label'].values
        valid_labels = outcomes.loc[valid_px, 'label'].values

        clf    = MultiModalDynamicModel(**model_params)
        scores = clf.fit(train_feature_inputs, train_feature_mask, train_labels)

        df_coef = clf.get_coefs(modality_list, modality_mask)

        df_coef_agg = pd.concat([df_coef_agg, df_coef])

        
        valid_scores = clf.predict_proba(valid_feature_inputs, valid_feature_mask)
            
        l_v_scores.extend(valid_scores)
        l_v_labels.extend(valid_labels)
        
        score, risks, attentions, shares = clf.get_summary_scores(valid_feature_inputs, valid_feature_mask)

        for idx, px in enumerate(valid_px):
            d_summarys_all[px] = {}

            d_summarys_all[px]['label'] = valid_labels[idx]
            d_summarys_all[px]['score'] = valid_scores[idx]
            d_summarys_all[px]['fold'] = fold
        
            for i,risk in enumerate(risks[idx]):
                d_summarys_all[px][f'risk_{modality_mask.columns[i]}'] = risk
            for i,attn in enumerate(attentions[idx]):
                d_summarys_all[px][f'attn_{modality_mask.columns[i]}'] = attn
            for i,share in enumerate(shares[idx]):
                d_summarys_all[px][f'share_{modality_mask.columns[i]}'] = share 
            
    # Calcola AUC con DeLong su TUTTI i fold
    if 1.0 in l_v_labels and 0.0 in l_v_labels:
        y_true = np.array(l_v_labels)
        y_pred = np.array(l_v_scores)
        
        n_total = len(y_true)
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        auc, auc_cov = delong_roc_variance(y_true, y_pred)
        auc_std = np.sqrt(auc_cov)
        ci = (auc - 1.96 * auc_std, auc + 1.96 * auc_std)
        
        print(f"\nOverall AUC calculated on {n_total} samples (pos={n_pos}, neg={n_neg})")
        print(f"AUC = {auc:.3f} (95% CI: {ci[0]:.3f}-{ci[1]:.3f})")

    return get_summary_df(d_summarys_all), df_coef_agg

# added by AI-ON-LAB - this function is used for train/test/ext-val split
def train_standard_split(modality_list_in, modality_mask, outcomes, l1_dfs_filter, model_params, 
                        train_set, test_set, ext_val_set=None):
    """
    Train with standard train/test/ext-val split (no cross-validation)
    
    Args:
        train_set: DataFrame or Index with 'fold'='train'
        test_set: DataFrame or Index with 'fold'='test'
        ext_val_set: Optional DataFrame or Index with 'fold'='ext_val'
    """
    
    # Get indices
    if isinstance(train_set, pd.DataFrame):
        train_idx = train_set.index
    else:
        train_idx = train_set
    
    if isinstance(test_set, pd.DataFrame):
        test_idx = test_set.index
    else:
        test_idx = test_set
        
    if ext_val_set is not None:
        if isinstance(ext_val_set, pd.DataFrame):
            ext_val_idx = ext_val_set.index
        else:
            ext_val_idx = ext_val_set
    
    print(f"Train samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")

    if ext_val_set is not None:
        print(f"External validation samples: {len(ext_val_idx)}")
    
    # Copy modalities
    modality_list = [df.copy(deep=True) for df in modality_list_in]
    
    # Apply L1 filters (exclude test set)
    for pos, filter in l1_dfs_filter.items():
        print(f"Applying L1 filter on modality position {pos}")
        l1_filter_features_list(modality_list, filter['l1_selection_df'], outcomes, test_idx, pos, **filter['kwargs'])
    
    
    # Prepare data
    train_feature_inputs = [df.loc[train_idx].values for df in modality_list]
    test_feature_inputs = [df.loc[test_idx].values for df in modality_list]
    
    train_feature_mask = modality_mask.loc[train_idx].astype(int).values
    test_feature_mask = modality_mask.loc[test_idx].astype(int).values
    
    train_labels = outcomes.loc[train_idx, 'label'].values
    test_labels = outcomes.loc[test_idx, 'label'].values
    
    # Train model
    clf = MultiModalDynamicModel(**model_params)
    clf.fit(train_feature_inputs, train_feature_mask, train_labels)
    
    # Get coefficients
    df_coef = clf.get_coefs(modality_list, modality_mask)
    
    # Predict on test set
    test_scores = clf.predict_proba(test_feature_inputs, test_feature_mask)
    score, risks, attentions, shares = clf.get_summary_scores(test_feature_inputs, test_feature_mask)
    
    # Store results
    d_summarys_all = {}
    for idx, px in enumerate(test_idx):
        d_summarys_all[px] = {
            'label': test_labels[idx],
            'score': test_scores[idx],
            'set': 'test'
        }
        for i, risk in enumerate(risks[idx]):
            d_summarys_all[px][f'risk_{modality_mask.columns[i]}'] = risk
        for i, attn in enumerate(attentions[idx]):
            d_summarys_all[px][f'attn_{modality_mask.columns[i]}'] = attn
        for i, share in enumerate(shares[idx]):
            d_summarys_all[px][f'share_{modality_mask.columns[i]}'] = share
    
    # Calculate test AUC
    if 1.0 in test_labels and 0.0 in test_labels:
        auc, auc_cov = delong_roc_variance(test_labels, test_scores)
        auc_std = np.sqrt(auc_cov)
        ci = (auc - 1.96 * auc_std, auc + 1.96 * auc_std)
        print(f"\nTest AUC: {auc:.3f} (95% CI: {ci[0]:.3f}-{ci[1]:.3f})")
    
    # External validation if provided
    if ext_val_set is not None:
        ext_val_feature_inputs = [df.loc[ext_val_idx].values for df in modality_list]
        ext_val_feature_mask = modality_mask.loc[ext_val_idx].astype(int).values
        ext_val_labels = outcomes.loc[ext_val_idx, 'label'].values
        
        ext_val_scores = clf.predict_proba(ext_val_feature_inputs, ext_val_feature_mask)
        score, risks, attentions, shares = clf.get_summary_scores(ext_val_feature_inputs, ext_val_feature_mask)
        
        for idx, px in enumerate(ext_val_idx):
            d_summarys_all[px] = {
                'label': ext_val_labels[idx],
                'score': ext_val_scores[idx],
                'set': 'ext_val'
            }
            for i, risk in enumerate(risks[idx]):
                d_summarys_all[px][f'risk_{modality_mask.columns[i]}'] = risk
            for i, attn in enumerate(attentions[idx]):
                d_summarys_all[px][f'attn_{modality_mask.columns[i]}'] = attn
            for i, share in enumerate(shares[idx]):
                d_summarys_all[px][f'share_{modality_mask.columns[i]}'] = share
        
        if 1.0 in ext_val_labels and 0.0 in ext_val_labels:
            auc, auc_cov = delong_roc_variance(ext_val_labels, ext_val_scores)
            auc_std = np.sqrt(auc_cov)
            ci = (auc - 1.96 * auc_std, auc + 1.96 * auc_std)
            print(f"External Validation AUC: {auc:.3f} (95% CI: {ci[0]:.3f}-{ci[1]:.3f})\n")
    
    return get_summary_df(d_summarys_all), df_coef

def hyperparameter_search(modality_list, modality_mask, df_out, l1_dfs_filter, hyperparam_grid):
    
    import itertools
    
    param_combinations = list(itertools.product(
        hyperparam_grid['epochs'],
        hyperparam_grid['lr'],
        hyperparam_grid['alpha'],
        hyperparam_grid['beta']
    ))
    
    print(f"Testing {len(param_combinations)} combinations with 10-fold CV\n")
    
    best_fold1_auc = 0
    best_params = None
    results = []
    
    for epochs, lr, alpha, beta in param_combinations:
        params = {'epochs': epochs, 'lr': lr, 'alpha': alpha, 'beta': beta, 
                  'cross_modality_enabled': False}
        
        # 10-fold CV completa
        summary_df, _ = train(
            modality_list_in=modality_list,
            modality_mask=modality_mask,
            outcomes=df_out,
            l1_dfs_filter=l1_dfs_filter,
            model_params=params,
            folds=10,
            predefined_folds=None
        )
        
        # Guarda SOLO fold 0 per scegliere parametri
        fold1_data = summary_df[summary_df['fold'] == 0]
        fold1_auc = roc_auc_score(fold1_data['label'], fold1_data['score'])
        
        results.append({
            'epochs': epochs, 'lr': lr, 'alpha': alpha, 'beta': beta,
            'fold1_auc': fold1_auc
        })
        
        print(f"epochs={epochs}, lr={lr}, alpha={alpha}, beta={beta} -> Fold1 AUC: {fold1_auc:.3f}")
        
        if fold1_auc > best_fold1_auc:
            best_fold1_auc = fold1_auc
            best_params = params
    
    print(f"\nBest params (based on fold 1): {best_params}")
    print(f"Fold 1 AUC: {best_fold1_auc:.3f}\n")
    
    return best_params, pd.DataFrame(results)

def train_subsample(modality_list_in, modality_mask, outcomes, l1_dfs_filter, model_params, folds=10):
    if folds=='LOO':
        folds=len(outcomes.index)

    l_aucs_res = []

    kf = ShuffleSplit(n_splits=20, test_size=0.1, random_state=42)

    for fold, (train, test) in enumerate(tqdm(list(kf.split(outcomes.index)), file=sys.stdout)):
        
        modality_list = [df.copy(deep=True) for df in modality_list_in]

        train_px = outcomes.index[train]
        valid_px  = outcomes.index[test]

        for pos, filter in l1_dfs_filter.items():
            l1_filter_features_list(modality_list, filter['l1_selection_df'], outcomes, valid_px, pos, **filter['kwargs'])

        empty_flag = False
        for modality_df in modality_list:
            if len(modality_df.columns)==0: empty_flag = True

        if empty_flag: continue

        train_feature_inputs = [df.loc[train_px].values for df in modality_list]
        valid_feature_inputs = [df.loc[valid_px].values for df in modality_list]

        train_feature_mask   = modality_mask.loc[train_px].astype(int).values
        valid_feature_mask   = modality_mask.loc[valid_px].astype(int).values
        
        train_labels = outcomes.loc[train_px, 'label']
        valid_labels = outcomes.loc[valid_px, 'label'].values

        clf    = MultiModalDynamicModel(**model_params)
        scores = clf.fit(train_feature_inputs, train_feature_mask, train_labels)
        
        valid_scores = clf.predict_proba(valid_feature_inputs, valid_feature_mask)
            
        if 1.0 in valid_labels and 0.0 in valid_labels:
            auc, ci = auc_roc_ci(valid_labels, valid_scores, 0.95)
            l_aucs_res.append ( ( auc, ci, valid_scores, valid_labels) )
    
    print (np.array(l_aucs_res)[:, 0].mean())
    return l_aucs_res

def train_eval_all(modality_list_in, modality_mask, outcomes, l1_dfs_filter, model_params, train_px, valid_px):
    l_v_scores = []
    l_v_labels = []
    d_summarys_all = {}

    modality_list = [df.copy(deep=True) for df in modality_list_in]

    for pos, filter in l1_dfs_filter.items():
        drop_rad_px = valid_px
        if len(valid_px.difference(train_px))==0: drop_rad_px = []
        df_coef = l1_filter_features_list(modality_list, filter['l1_selection_df'], outcomes, drop_rad_px, pos, **filter['kwargs'])

    train_feature_inputs = [df.loc[train_px].values for df in modality_list]
    valid_feature_inputs = [df.loc[valid_px].values for df in modality_list]

    train_feature_mask   = modality_mask.loc[train_px].astype(int).values
    valid_feature_mask   = modality_mask.loc[valid_px].astype(int).values
    
    train_labels = outcomes.loc[train_px, 'label']
    valid_labels = outcomes.loc[valid_px, 'label'].values

    print (f"Training on {len(train_labels)} patients and testing on {len(valid_labels)} patients")

    clf    = MultiModalDynamicModel(**model_params)
    scores = clf.fit(train_feature_inputs, train_feature_mask, train_labels)
    
    valid_scores = clf.predict_proba(valid_feature_inputs, valid_feature_mask)
        
    l_v_scores.extend(valid_scores)
    l_v_labels.extend(valid_labels)
    
    score, risks, attentions, shares = clf.get_summary_scores(valid_feature_inputs, valid_feature_mask)

    for idx, px in enumerate(valid_px):
        d_summarys_all[px] = {}

        d_summarys_all[px]['label'] = valid_labels[idx]
        d_summarys_all[px]['score'] = valid_scores[idx]
    
        for i,risk in enumerate(risks[idx]):
            d_summarys_all[px][f'risk_{modality_mask.columns[i]}'] = risk
        for i,attn in enumerate(attentions[idx]):
            d_summarys_all[px][f'attn_{modality_mask.columns[i]}'] = attn
        for i,share in enumerate(shares[idx]):
            d_summarys_all[px][f'share_{modality_mask.columns[i]}'] = share 
        
        
    if 1.0 in l_v_labels and 0.0 in l_v_labels:
        auc, ci = auc_roc_ci(l_v_labels, l_v_scores, 0.95)
    print ("Fold [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL".format(0 + 1, auc, (ci[1] - ci[0]) / 2.0) )
    
    return get_summary_df(d_summarys_all)

def average_models(summary_dfs_in, models_to_average):
    l_df_score = [summary_dfs_in[model][['score']] for model in models_to_average]
    df_score = pd.concat(l_df_score, axis=1).mean(axis=1).rename('score')

    l_df_label = [summary_dfs_in[model][['label']] for model in models_to_average]
    df_label = pd.concat(l_df_label, axis=1).mean(axis=1).rename('label')

    dfs_mock_mods = []
    for model in models_to_average:
        df_mod = summary_dfs_in[model][['score']].rename(columns={'score':f'risk_{model.lower()}'})
        df_mod[f'attn_{model.lower()}'] = 1.0
        dfs_mock_mods.append(df_mod)
        
    df_mock_mod = pd.concat(dfs_mock_mods, axis=1).fillna(0)

    df = pd.concat([df_score, df_label, df_mock_mod], axis=1)

    df['score_norm'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
    df['error'] = abs(df['label'] - df['score_norm'])
    df = df.sort_values(['label', 'score'])
    return df

def train_LR_eval_all(df, outcomes, train_px, valid_px, filter=None):

    l_v_scores = []
    l_v_labels = []
    d_summarys_all = {}

    modality_df = df.copy(deep=True)

    if filter is not None:
        modality_df  = l1_filter_features_df(modality_df, filter['l1_selection_df'], outcomes, valid_px, **filter['kwargs'])

    train_px = modality_df.index.intersection(train_px)
    valid_px = modality_df.index.intersection(valid_px)
    
    train_feature_inputs = modality_df.loc[train_px].values
    valid_feature_inputs = modality_df.loc[valid_px].values
    
    train_labels = outcomes.loc[train_px, 'label']
    valid_labels = outcomes.loc[valid_px, 'label'].values

    print (f"Training on {len(train_labels)} patients and testing on {len(valid_labels)} patients")

    for fx in modality_df.columns:
        pval = ttest_ind(modality_df.loc[train_px, fx], modality_df.loc[valid_px, fx], equal_var=True).pvalue
        if  pval < 0.05:
            print (pval, ":", fx)

    scaler  = RobustScaler()
    scaler.fit(train_feature_inputs)
    X_train = scaler.transform(train_feature_inputs)
    X_valid = scaler.transform(valid_feature_inputs)

    clf     =  LogisticRegression(penalty='elasticnet', max_iter=2500, solver='saga', l1_ratio=0.5, C=0.1, class_weight='balanced')
    clf.fit(X_train, train_labels)
    valid_scores = clf.predict_proba(X_valid)[:,1] - 0.5
        
    l_v_scores.extend(valid_scores)
    l_v_labels.extend(valid_labels)
    
    for idx, px in enumerate(valid_px):
        d_summarys_all[px] = {}

        d_summarys_all[px]['label'] = valid_labels[idx]
        d_summarys_all[px]['score'] = valid_scores[idx]
        
    if 1.0 in l_v_labels and 0.0 in l_v_labels:
        auc, ci = auc_roc_ci(l_v_labels, l_v_scores, 0.95)

    print ("Fold [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL".format("Test", auc, (ci[1] - ci[0]) / 2.0) )
    
    return get_summary_df(d_summarys_all)


def get_lesion_dfs(df_by_site, df_outcomes, index_col='global_lesion_id'):

    df_lesions = df_by_site.copy(deep=True)

    df_lesions['global_lesion_id'] = df_lesions.index.get_level_values(0) + '-' + df_lesions['lesion_index'].astype(str)

    df_lesions = df_lesions.loc[df_lesions.index.isin(['filtered-radiomics'], level='job_tag')]

    df_lesions_data     = df_outcomes.join(df_lesions, how='inner')[df_lesions.columns].reset_index().set_index(index_col)
    df_lesions_outcomes = df_outcomes.join(df_lesions, how='inner').reset_index()[[index_col, 'label']].set_index(index_col).append(df_outcomes[['label']])

    reindex_data = df_outcomes.index.intersection (df_lesions_data.index)
    reindex_outcomes = df_outcomes.index.intersection (df_lesions_outcomes.index)

    df_lesions_data = df_lesions_data.loc[reindex_data]
    df_lesions_outcomes = df_lesions_outcomes.loc[reindex_outcomes]

    return df_lesions_data, df_lesions_outcomes

def train_MILR(df, outcomes, filter=None, n_splits=10, cv_index=None, model_params=None, return_metrics=False):
    if model_params is None: model_params = {}
    l_v_scores = []
    l_v_labels = []
    d_summarys_all = {}

    if cv_index is None:
        cv_index = outcomes.index

    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    for fold, (train, test) in enumerate(tqdm(list(kf.split( cv_index )), file=sys.stdout)):
        
        modality_df = df.copy(deep=True)

        train_px  = cv_index[train]
        valid_px  = cv_index[test]

        if filter is not None:
            modality_df  = l1_filter_features_df(modality_df, filter['l1_selection_df'], outcomes, valid_px, **filter['kwargs'])


        df_train_data   = modality_df.loc[modality_df.index.isin(train_px)]
        df_valid_data   = modality_df.loc[modality_df.index.isin(valid_px)]

        df_train_labels = outcomes.loc[train_px]

        clf = MultiLesionModel(input_size=len(modality_df.columns), **model_params)
        clf.fit(df_train_data, df_train_labels)

        df_scores = clf.predict_proba(df_train_data)

        # auc, ci = auc_roc_ci(outcomes.loc[df_scores.index, 'label'], df_scores['score'], 0.95)
        # print ("Train [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL".format(fold + 1, auc, (ci[1] - ci[0]) / 2.0) )

        df_scores = clf.predict_proba(df_valid_data)
        l_v_scores.extend(list(df_scores['score']))
        l_v_labels.extend(list(outcomes.loc[df_scores.index, 'label']))
    
        for idx, px in enumerate(valid_px):
            d_summarys_all[px] = {}

            d_summarys_all[px]['label'] = outcomes.loc[px, 'label']
            d_summarys_all[px]['score'] = df_scores.loc[px,  'score']
    
    auc, ci = auc_roc_ci(l_v_labels, l_v_scores, 0.95)
    low_ci = ci[0]

    print ("Agg [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL".format(fold + 1, auc, (ci[1] - ci[0]) / 2.0) )

    if return_metrics:
        return auc, low_ci
    else:
        return get_summary_df(d_summarys_all)

def train_MILR_subsample(df, outcomes, filter=None, n_splits=10, cv_index=None, model_params=None, return_metrics=False):
    if model_params is None: model_params = {}

    l_aucs_res = []

    if cv_index is None:
        cv_index = outcomes.index

    kf = ShuffleSplit(n_splits=20, test_size=0.1, random_state=42)
    for fold, (train, test) in enumerate(tqdm(list(kf.split( cv_index )), file=sys.stdout)):
        
        modality_df = df.copy(deep=True)

        train_px  = cv_index[train]
        valid_px  = cv_index[test]

        if filter is not None:
            modality_df  = l1_filter_features_df(modality_df, filter['l1_selection_df'], outcomes, valid_px, **filter['kwargs'])


        df_train_data   = modality_df.loc[modality_df.index.isin(train_px)]
        df_valid_data   = modality_df.loc[modality_df.index.isin(valid_px)]

        df_train_labels = outcomes.loc[train_px]

        clf = MultiLesionModel(input_size=len(modality_df.columns), **model_params)
        clf.fit(df_train_data, df_train_labels)
        
        df_scores = clf.predict_proba(df_valid_data)
        valid_scores = list(df_scores['score'])
        valid_labels = list(outcomes.loc[df_scores.index, 'label'])
    
        if 1.0 in valid_labels and 0.0 in valid_labels:
            auc, ci = auc_roc_ci(valid_labels, valid_scores, 0.95)
            l_aucs_res.append ( ( auc, ci, valid_scores, valid_labels) )
    
    print (np.array(l_aucs_res)[:, 0].mean())
    return l_aucs_res


def train_MILR_eval_all(df, outcomes, train_px, valid_px, filter=None, cv_index=None, model_params=None):
    print (f"Training on {len(train_px)} patients and testing in {len(valid_px)} patients!")

    if model_params is None: model_params = {}
    l_v_scores = []
    l_v_labels = []
    d_summarys_all = {}

    modality_df = df.copy(deep=True)

    if filter is not None:
        print ("Filtering:")
        modality_df  = l1_filter_features_df(modality_df, filter['l1_selection_df'], outcomes, valid_px, **filter['kwargs'])

    print ("Overlap:", train_px.intersection(valid_px))

    df_train_data   = modality_df.loc[modality_df.index.isin(train_px)]
    df_valid_data   = modality_df.loc[modality_df.index.isin(valid_px)]

    df_train_labels = outcomes.loc[train_px]

    clf = MultiLesionModel(input_size=len(modality_df.columns), **model_params)
    clf.fit(df_train_data, df_train_labels)

    df_scores = clf.predict_proba(df_train_data)

    auc, ci = auc_roc_ci(outcomes.loc[df_scores.index, 'label'], df_scores['score'], 0.95)
    print ("Train [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL".format('Train', auc, (ci[1] - ci[0]) / 2.0) )

    df_scores = clf.predict_proba(df_valid_data)
    l_v_scores.extend(list(df_scores['score']))
    l_v_labels.extend(list(outcomes.loc[df_scores.index, 'label']))

    auc, ci = auc_roc_ci(outcomes.loc[df_scores.index, 'label'], df_scores['score'], 0.95)
    print ("Valid [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL N={3}".format('Test', auc, (ci[1] - ci[0]) / 2.0, len(l_v_scores) ) )

    auc, ci = auc_roc_ci(l_v_labels, l_v_scores, 0.95)

    low_ci = ci[0]

    print ("Agg [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL".format("Test", auc, (ci[1] - ci[0]) / 2.0) )

    for idx, px in enumerate(valid_px):
        d_summarys_all[px] = {}

        d_summarys_all[px]['label'] = outcomes.loc[px, 'label']
        d_summarys_all[px]['score'] = df_scores.loc[px,  'score']


    return get_summary_df(d_summarys_all)
    # return auc, low_ci

def get_LR_auc_dist(df, outcomes, filter=None, n_splits=25, cv_index=None):
    l_v_scores = []
    l_v_labels = []
    d_summarys_all = {}

    if cv_index is None:
        cv_index = outcomes.index

    kf = ShuffleSplit(n_splits=n_splits, test_size=.247, random_state=0)

    aucs = []

    for fold, (train, test) in enumerate(tqdm(list(kf.split( cv_index )), file=sys.stdout)):
        
        modality_df = df.copy(deep=True)

        train_px = cv_index[train]
        valid_px  = cv_index[test]

        if filter is not None:
            modality_df  = l1_filter_features_df(modality_df, filter['l1_selection_df'], outcomes, valid_px, **filter['kwargs'])

        train_feature_inputs = modality_df.loc[train_px].values
        valid_feature_inputs = modality_df.loc[valid_px].values
        
        train_labels = outcomes.loc[train_px, 'label']
        valid_labels = outcomes.loc[valid_px, 'label'].values

        scaler  = RobustScaler()
        scaler.fit(train_feature_inputs)
        X_train = scaler.transform(train_feature_inputs)
        X_valid = scaler.transform(valid_feature_inputs)

        clf     =  LogisticRegression(penalty='elasticnet', max_iter=2500, solver='saga', l1_ratio=0.5, C=0.1, class_weight='balanced')
        clf.fit(X_train, train_labels)
        valid_scores = clf.predict_proba(X_valid)[:,1] - 0.5
       
        auc, ci = auc_roc_ci(valid_labels, valid_scores, 0.95)

        aucs.append(auc)
                
    return aucs

def gen_folds(df_cohort, outcomes, n_splits=10):

    cv_index = outcomes.index

    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)

    for fold, (train, test) in enumerate(tqdm(list(kf.split( cv_index )), file=sys.stdout)):
        
        train_px = cv_index[train]
        valid_px  = cv_index[test]

        df_cohort.loc[train_px, f'fold_{fold}'] = 'train'
        df_cohort.loc[valid_px, f'fold_{fold}'] = 'check'

    return df_cohort[df_cohort['cohort'].eq('discovery')]

def train_LR_subsample(df, outcomes, filter=None, n_samples=20, cv_index=None, get_coef_data=False):

    if cv_index is None:
        cv_index = outcomes.index

    kf = ShuffleSplit(n_splits=n_samples, test_size=0.2, random_state=42)

    n_features = []

    df_coef_agg = pd.DataFrame()

    l_aucs_res = []

    for fold, (train, test) in enumerate(tqdm(list(kf.split( cv_index )), file=sys.stdout)):
        
        modality_df = df.copy(deep=True)

        train_px = cv_index[train]
        valid_px  = cv_index[test]

        if filter is not None:
            modality_df  = l1_filter_features_df(modality_df, filter['l1_selection_df'], outcomes, valid_px, **filter['kwargs'])

        if len(modality_df.columns)==0: continue

        n_features.append(len(modality_df.columns)) 

        train_feature_inputs = modality_df.loc[train_px].values
        valid_feature_inputs = modality_df.loc[valid_px].values
        
        train_labels = outcomes.loc[train_px, 'label']
        valid_labels = outcomes.loc[valid_px, 'label'].values

        scaler  = RobustScaler()
        scaler.fit(train_feature_inputs)
        X_train = scaler.transform(train_feature_inputs)
        X_valid = scaler.transform(valid_feature_inputs)

        clf     =  LogisticRegression(penalty='elasticnet', max_iter=2500, solver='saga', l1_ratio=0.5, C=0.1, class_weight='balanced')
        clf.fit(X_train, train_labels)

        df_coef = pd.DataFrame([clf.coef_[0], np.sign(clf.coef_[0])], columns=modality_df.columns, index=['coef', 'sign'])

        df_coef_agg = pd.concat([df_coef_agg, df_coef])

        valid_scores = clf.predict_proba(X_valid)[:,1] - 0.5
            
        if 1.0 in valid_labels and 0.0 in valid_labels:
            auc, ci = auc_roc_ci(valid_labels, valid_scores, 0.95)
            l_aucs_res.append ( ( auc, ci, valid_scores, valid_labels) )
   
    print (np.array(l_aucs_res)[:, 0].mean())

    return l_aucs_res

def train_LR(df, outcomes, filter=None, n_splits=10, cv_index=None, label_col='label', get_coef_data=False):
    l_v_scores = []
    l_v_labels = []
    d_summarys_all = {}

    if cv_index is None:
        cv_index = outcomes.index

    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)

    n_features = []

    df_coef_agg = pd.DataFrame()

    for fold, (train, test) in enumerate(tqdm(list(kf.split( cv_index )), file=sys.stdout)):
        
        modality_df = df.copy(deep=True)

        train_px = cv_index[train]
        valid_px  = cv_index[test]

        if filter is not None:
            modality_df  = l1_filter_features_df(modality_df, filter['l1_selection_df'], outcomes, valid_px, **filter['kwargs'])

        n_features.append(len(modality_df.columns)) 

        train_feature_inputs = modality_df.loc[train_px].values
        valid_feature_inputs = modality_df.loc[valid_px].values
        
        train_labels = outcomes.loc[train_px, label_col]
        valid_labels = outcomes.loc[valid_px, label_col].values

        scaler  = RobustScaler()
        scaler.fit(train_feature_inputs)
        X_train = scaler.transform(train_feature_inputs)
        X_valid = scaler.transform(valid_feature_inputs)

        clf     =  LogisticRegression(penalty='elasticnet', max_iter=2500, solver='saga', l1_ratio=0.5, C=0.1, class_weight='balanced')
        clf.fit(X_train, train_labels)

        df_coef = pd.DataFrame([clf.coef_[0], np.sign(clf.coef_[0])], columns=modality_df.columns, index=['coef', 'sign'])

        df_coef_agg = pd.concat([df_coef_agg, df_coef])

        valid_scores = clf.predict_proba(X_valid)[:,1] - 0.5
            
        l_v_scores.extend(valid_scores)
        l_v_labels.extend(valid_labels)
        
        for idx, px in enumerate(valid_px):
            d_summarys_all[px] = {}

            d_summarys_all[px]['label'] = valid_labels[idx]
            d_summarys_all[px]['score'] = valid_scores[idx]
            d_summarys_all[px]['fold'] = fold

            
        if 1.0 in l_v_labels and 0.0 in l_v_labels:
            auc, ci = auc_roc_ci(l_v_labels, l_v_scores, 0.95)
    print ("Fold [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL".format(fold + 1, auc, (ci[1] - ci[0]) / 2.0) )
    print ("Avg N features:", np.mean(n_features))

    return get_summary_df(d_summarys_all), df_coef_agg

def rf_pca(df, hue_series, palette=None, plot=True, hold=False, init=True, marker=None, transform=None, hue_order=None, save_name=None, color=None, legend_name=''):
    df = df.reset_index().set_index('main_index').drop(columns=["job_tag", "lesion_index", "site"], errors='ignore')

    n_com = 2
    
    fx_to_use = df

    if transform is None:
        scaler = PowerTransformer()
        pca = PCA(n_components=n_com, whiten=True)
        print ("Running PCA()")
        # pca = MDS()
        df_pca = pd.DataFrame(pca.fit_transform(scaler.fit_transform(fx_to_use)), columns=[f'pc{i}' for i in range(n_com)], index=fx_to_use.index)
    else:
        scaler, pca = transform
        df_pca = pd.DataFrame(pca.transform(scaler.transform(fx_to_use)), columns=[f'pc{i}' for i in range(n_com)], index=fx_to_use.index)

    if hue_series is not None:
        hue_series = hue_series.values

    if plot:
        if init: plt.figure(figsize=(8,8))    
        sns.scatterplot(df_pca['pc0'], df_pca['pc1'], hue=hue_series, hue_order=hue_order, palette=palette, marker=marker, color=color)
        plt.legend(title=legend_name, loc='bottom right')#, bbox_to_anchor=(1, 0.5))
        # plt.xlim(-2, 4)
        # plt.ylim(-2, 4)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        if save_name is not None: plt.savefig(save_name)
        # if not hold: plt.show()
                
    return scaler, pca

def rf_pca_V2(df, hue_series, transform=None):
    df = df.reset_index().set_index('main_index').drop(columns=["job_tag", "lesion_index", "site"], errors='ignore')

    n_com = 2
    
    fx_to_use = df

    if transform is None:
        scaler = PowerTransformer()
        pca = PCA(n_components=n_com, whiten=True)
        print ("Running PCA()")
        # pca = MDS()
        df_pca = pd.DataFrame(pca.fit_transform(scaler.fit_transform(fx_to_use)), columns=[f'pc{i}' for i in range(n_com)], index=fx_to_use.index)
    else:
        scaler, pca = transform
        df_pca = pd.DataFrame(pca.transform(scaler.transform(fx_to_use)), columns=[f'pc{i}' for i in range(n_com)], index=fx_to_use.index)

    print (len(df_pca), len(hue_series))         
    return pd.concat([df_pca, hue_series], axis=1)

def rf_features(df, hue_series, fx, palette=None, plot=True, hold=False, init=True, marker=None, transform=None, save_name=None):
    df = df.reset_index().set_index('main_index').drop(columns=["job_tag", "lesion_index", "site"], errors='ignore')

    n_com = 2
    df = df[list(fx)]
    if transform is None:
        scaler = PowerTransformer()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    else:
        scaler = transform
        df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    
    if plot:
        if init: plt.figure(figsize=(8,8))    
        sns.scatterplot(df[fx[0]], df[fx[1]], hue=hue_series.values, palette=palette, marker=marker)
        plt.legend(title=hue_series.name)
        # plt.xlim(-2, 4)
        # plt.ylim(-2, 4)
        if save_name is not None: plt.savefig(save_name)
        if not hold: plt.show()
            
def generate_pca_figure_clusters(df, lesion_ids=None):
    df_rand = df[df.index.isin(['pertubation-radiomics'], level='job_tag')]
    df_filt = df[df.index.isin(['filtered-radiomics'], level='job_tag')]

    df_rand_ALL = df_rand[df_rand.index.isin(['PC', 'PL', 'LN'], level='site')]
    df_filt_ALL = df_filt[df_filt.index.isin(['PC', 'PL', 'LN'], level='site')]

    df_rand_ALL['global_lesion_id'] = df_rand_ALL.index.get_level_values(0) + '-' + df_rand_ALL['lesion_index'].astype(str)
    df_filt_ALL['global_lesion_id'] = df_filt_ALL.index.get_level_values(0) + '-' + df_filt_ALL['lesion_index'].astype(str)

    if lesion_ids is None:
        lesion_ids = np.random.default_rng(4).choice(df_rand_ALL.set_index('global_lesion_id').index.unique(), size=20, replace=False)

    df_rand_sub = df_rand_ALL[df_rand_ALL['global_lesion_id'].isin(lesion_ids)]
    df_filt_sub = df_filt_ALL[df_filt_ALL['global_lesion_id'].isin(lesion_ids)]

    transform = rf_pca(
        df_rand_ALL.drop(columns=['global_lesion_id']),
        hue_series=df_rand_ALL['global_lesion_id'],
        plot=False,
    )
    # return df_ALL
    transform = rf_pca(
        df_rand_sub.drop(columns=['global_lesion_id']),
        transform = transform,
        hue_series=df_rand_sub['global_lesion_id'],
        plot=True,
        hold=True
    )

    transform = rf_pca(
        df_filt_sub.drop(columns=['global_lesion_id']),
        transform = transform,
        hue_series=None,
        plot=True,
        init=False,
        color='k'
    )

    apply_style()

def generate_pca_figure_1d(df, df_outcome, lesion_ids=None, color=None, replace_id=None):
    df_rand = df[df.index.isin(['pertubation-radiomics'], level='job_tag')]
    df_filt = df[df.index.isin(['filtered-radiomics'], level='job_tag')]

    # outlier_counts,  sel_fx_outlier = select_outlier_stable_features(df, cutoff=6)
    # variance_scores, sel_fx_robust  = select_by_interlesion_variance(df, cutoff=0.15)
    # fx_to_use = sorted(set(sel_fx_outlier).intersection(set(sel_fx_robust)))

    fx_to_use, df_coef = select_radiomics_features_elastic(df.reset_index().set_index('main_index'), df_outcome, robustness_cutoff=0.15, outlier_cutoff=6, l1_strength=0.1) 
    print (len(fx_to_use))

    df_rand_ALL = df_rand[df_rand.index.isin(['PC', 'PL', 'LN'], level='site')]
    df_filt_ALL = df_filt[df_filt.index.isin(['PC', 'PL', 'LN'], level='site')]

    df_rand_ALL['global_lesion_id'] = df_rand_ALL.index.get_level_values(0) + '-' + df_rand_ALL['lesion_index'].astype(str)
    df_filt_ALL['global_lesion_id'] = df_filt_ALL.index.get_level_values(0) + '-' + df_filt_ALL['lesion_index'].astype(str)

    if lesion_ids is None:
        lesion_ids = np.random.default_rng(4).choice(df_rand_ALL.set_index('global_lesion_id').index.unique(), size=20, replace=False)

    
    
    df_rand_sub = df_rand_ALL[df_rand_ALL['global_lesion_id'].isin(lesion_ids)]
    df_filt_sub = df_filt_ALL[df_filt_ALL['global_lesion_id'].isin(lesion_ids)]

    transform = rf_pca(
        df_filt_ALL.drop(columns=['global_lesion_id'])[fx_to_use],
        hue_series=df_filt_ALL['global_lesion_id'],
        plot=False,
    )

    df_rand_sub = df_rand_sub.reset_index().set_index('global_lesion_id').drop(columns=["main_index", "job_tag", "lesion_index", "site"], errors='ignore')
    df_filt_sub = df_filt_sub.reset_index().set_index('global_lesion_id').drop(columns=["main_index", "job_tag", "lesion_index", "site"], errors='ignore')

    scaler, pca = transform
    df_rand_sub = pd.DataFrame(pca.transform(scaler.transform(df_rand_sub[fx_to_use])), columns=[f'pc{i}' for i in range(2)], index=df_rand_sub.index).reset_index()
    df_filt_sub = pd.DataFrame(pca.transform(scaler.transform(df_filt_sub[fx_to_use])), columns=[f'pc{i}' for i in range(2)], index=df_filt_sub.index)

    return df_rand_sub, df_filt_sub
    # # return df_rand_sub

    # plt.figure(figsize=(4,4))
    # sns.violinplot(data=df_rand_sub, y='global_lesion_id', x='pc0', color=color, label='Pertubations')

    # if replace_id: lesion_id = replace_id
    # else: lesion_id = lesion_ids[0]

    # plt.plot([df_filt_sub['pc0'], df_filt_sub['pc0']], [-0.5, 0.5], color='k', linestyle='--', label=f'Original [{lesion_id}]')
    # plt.legend()
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('')
    # plt.yticks([])

    # return df_rand_sub

def generate_genomic_lr_fig(df_genomic, df_outcomes, panel='4B'):
    df_genomic_relabled = df_genomic.copy(deep=True)
    df_genomic_relabled.columns = [col.replace("driver_binarized", "Mutation").replace(": AMP_binarized", " Amplification") for col in df_genomic_relabled.columns]

    df_nontmb = df_genomic.loc[:, ~df_genomic.columns.str.contains('TMB')]
    df_tmb    = df_genomic.loc[:,  df_genomic.columns.str.contains('TMB')]

    df_nontmb_relabled = df_nontmb
    df_nontmb_relabled.columns = [col.replace("driver_binarized", "Mutation").replace(": AMP_binarized", " Amplification") for col in df_nontmb_relabled.columns]

    _, gen_coefs_N = train_LR(df_nontmb_relabled, df_outcomes.loc[df_tmb.index], None, get_coef_data=True)
    _, gen_coefs_H = train_LR(df_genomic_relabled, df_outcomes.loc[df_genomic.index], get_coef_data=True)

    gen_coefs_N = gen_coefs_N[['EGFR Mutation', 'STK11 Mutation']]
    gen_coefs_H = gen_coefs_H[['EGFR Mutation', 'STK11 Mutation', 'TMB']]

    gen_coefs_N['Fit'] = 'Fit Without TMB'
    gen_coefs_H['Fit'] = 'Fit With TMB'

    df_gen_data = pd.concat( [
        gen_coefs_N.loc['coef'].melt(id_vars='Fit'),
        gen_coefs_H.loc['coef'].melt(id_vars='Fit')
    ])

    if panel is not None:
        df_gen_data.to_excel('./excel/4B.xlsx', sheet_name='4B')

    plt.figure(figsize=(10,6))
    ax = sns.violinplot(data=df_gen_data, x='variable', y='value', hue='Fit', cut=0, palette=sns.color_palette("GnBu", 3))
    pairs = [
        [('EGFR Mutation', 'Fit Without TMB'), ('EGFR Mutation', 'Fit With TMB')],
        [('STK11 Mutation', 'Fit Without TMB'), ('STK11 Mutation', 'Fit With TMB')],
    ]

    annotator = Annotator(ax, pairs, data=df_gen_data, x='variable', y='value', hue='Fit')
    annotator.configure(test='Mann-Whitney', text_format='star')
    annotator.apply_and_annotate()

    plt.legend(frameon=False)
    plt.ylim(None,0.6)
    apply_style()
    plt.xlabel(None)

    plt.ylabel("LR Coefficient")

def filter_dict(old_dict, keys):
    new_dict = {}
    for key in keys:
        if key in old_dict.keys():
            new_dict[key] = copy.deepcopy(old_dict[key])

    return new_dict

def check_modality_counts(df_mask_disc, df_outcome_disc, df_mask_test, df_outcome_test):
    df_disc = df_mask_disc.join(df_outcome_disc)
    df_test = df_mask_test.join(df_outcome_test)

    for mod in df_mask_test.columns:
        print (mod)
        print ( df_disc[df_disc[mod]]['label'].mean() )
        print ( df_test[df_test[mod]]['label'].mean() )

def generate_pca_comparision(df_std, df_alt, df_std_clinical, df_alt_clinical, sites, filter=None):

    df_std = df_std[df_std.index.isin(['pertubation-radiomics'], level='job_tag')]
    df_alt = df_alt[df_alt.index.isin(['pertubation-radiomics'], level='job_tag')]

    df_std_site  = df_std[df_std.index.isin(sites, level='site')]
    df_alt_site  = df_alt[df_alt.index.isin(sites, level='site')]


    transform = rf_pca(
        df_std,
        hue_series=None,
        plot=False,
    )

    rf_pca(
        df_std_site,
        hue_series=df_std_clinical.loc[df_std_site.index.get_level_values(0),'label'].replace({1:'POD-Disc', 0:'PR/CR-Disc'}), 
        hue_order=['POD-Disc', 'PR/CR-Disc'],
        transform = transform,
        palette='PuBu',
        hold=True
    )

    rf_pca(
        df_alt_site,
        hue_series=df_alt_clinical.loc[df_alt_site.index.get_level_values(0),'label'].replace({1:'POD-Valid', 0:'PR/CR-Valid'}), 
        hue_order=['POD-Valid', 'PR/CR-Valid'],
        transform = transform,
        palette='OrRd',
        init=False
    )

def generate_pca_figure(df, df_clinical):
    df = df[df.index.isin(['pertubation-radiomics'], level='job_tag')]

    df_ALL = df[df.index.isin(['PC', 'PL', 'LN'], level='site')]
    df_PC  = df[df.index.isin(['PC'], level='site')]
    df_PL  = df[df.index.isin(['PL'], level='site')]
    df_LN  = df[df.index.isin(['LN'], level='site')]

    print (df_clinical.loc[df_PC.index.get_level_values(0),'label'].mean())
    print (df_clinical.loc[df_PL.index.get_level_values(0),'label'].mean())
    print (df_clinical.loc[df_LN.index.get_level_values(0),'label'].mean())

    transform = rf_pca(
        df_ALL,
        hue_series=df_clinical.loc[df_ALL.index.get_level_values(0),'label'],
        plot=False,
    )

    rf_pca(
        df_PC,
        hue_series=df_clinical.loc[df_PC.index.get_level_values(0),'label'].replace({1:'SD/PD Parenchymal Lesions', 0:'PR/CR Parenchymal Lesions'}), 
        transform = transform,
        palette='PuBu',
        hue_order=['PR/CR Parenchymal Lesions', 'SD/PD Parenchymal Lesions'],
        hold=True
    )
    rf_pca(
        df_PL,
        hue_series=df_clinical.loc[df_PL.index.get_level_values(0),'label'].replace({1:'SD/PD Pleural Lesions', 0:'PR/CR Pleural Lesions'}),
        transform = transform,
        palette='YlGn',
        hold=True,
        hue_order=['PR/CR Pleural Lesions', 'SD/PD Pleural Lesions'],

        init=False,

    )
    rf_pca(
        df_LN,
        hue_series=df_clinical.loc[df_LN.index.get_level_values(0),'label'].replace({1:'SD/PD Nodal Lesions', 0:'PR/CR Nodal Lesions'}),
        transform = transform,
        palette='OrRd',
        init=False,
        hue_order=['PR/CR Nodal Lesions', 'SD/PD Nodal Lesions'],

        save_name="site_responders_breakdown.pdf"

    )

    apply_style()

def generate_pca_figure_V2(df, df_clinical):
    df = df.reset_index().set_index(['main_index', 'job_tag', 'site'])
    print (df.index)
    df = df[df.index.isin(['pertubation-radiomics'], level='job_tag')]

    df_ALL = df[df.index.isin(['PC', 'PL', 'LN'], level='site')]
    df_PC  = df[df.index.isin(['PC'], level='site')]
    df_PL  = df[df.index.isin(['PL'], level='site')]
    df_LN  = df[df.index.isin(['LN'], level='site')]

    print (df_clinical.loc[df_PC.index.get_level_values(0),'label'].mean())
    print (df_clinical.loc[df_PL.index.get_level_values(0),'label'].mean())
    print (df_clinical.loc[df_LN.index.get_level_values(0),'label'].mean())

    transform = rf_pca(
        df_ALL,
        hue_series=df_clinical.loc[df_ALL.index.get_level_values(0),'label'],
        plot=False,
    )

    df_pca_pc = rf_pca_V2(
        df_PC,
        hue_series=df_clinical.loc[df_PC.index.get_level_values(0),'label'].replace({1:'SD/PD', 0:'PR/CR'}), 
        transform = transform,
    )
    df_pca_pl = rf_pca_V2(
        df_PL,
        hue_series=df_clinical.loc[df_PL.index.get_level_values(0),'label'].replace({1:'SD/PD', 0:'PR/CR'}),
        transform = transform
    )
    df_pca_ln = rf_pca_V2(
        df_LN,
        hue_series=df_clinical.loc[df_LN.index.get_level_values(0),'label'].replace({1:'SD/PD', 0:'PR/CR'}),
        transform = transform
    )

    df_pca_pc_all = rf_pca_V2(
        df_PC,
        hue_series=df_clinical.loc[df_PC.index.get_level_values(0),'label'].replace({1:'All', 0:'All'}), 
        transform = transform,
    )
    df_pca_pl_all = rf_pca_V2(
        df_PL,
        hue_series=df_clinical.loc[df_PL.index.get_level_values(0),'label'].replace({1:'All', 0:'All'}),
        transform = transform
    )
    df_pca_ln_all = rf_pca_V2(
        df_LN,
        hue_series=df_clinical.loc[df_LN.index.get_level_values(0),'label'].replace({1:'All', 0:'All'}),
        transform = transform
    )
    df_pca_pc['site'] = 'Parencymal'
    df_pca_pl['site'] = 'Pleural'
    df_pca_ln['site'] = 'Nodal'

    df_pca_pc_all['site'] = 'Parencymal'
    df_pca_pl_all['site'] = 'Pleural'
    df_pca_ln_all['site'] = 'Nodal'
    # plt.figure(figsize=(10,10))
    df = pd.concat([df_pca_pc, df_pca_pl, df_pca_ln])

    print (df)
    # sns.scatterplot(df['pc0'], df['pc1'], hue=df['label'])
    # apply_style()

    plt.figure(figsize=(8,8))
    df = pd.concat([df, df_pca_pc_all, df_pca_pl_all, df_pca_ln_all], axis=0)
    print (df.to_excel('./excel/2C.xlsx', sheet_name='2C'))
    ax = sns.violinplot(data=df,  x='site', y='pc0', hue='label', cut=0, hue_order=['PR/CR', 'SD/PD', 'All'], split=False, palette=['#a6bddb', '#0570b0', '#341C8D'])

    print (df.groupby(['site', 'label']).count())

    pairs = [
        [('Parencymal', 'SD/PD'), ('Parencymal', 'PR/CR')],
        [('Pleural', 'SD/PD'), ('Pleural', 'PR/CR')],
        [('Nodal', 'SD/PD'), ('Nodal', 'PR/CR')],

        [('Parencymal', 'All'), ('Pleural', 'All')],
        [('Parencymal', 'All'), ('Nodal', 'All')],
        [('Nodal', 'All'), ('Pleural', 'All')],
    ]
    annotator = Annotator(ax, pairs, data=df, x='site', y='pc0', hue='label', cut=0, palette='YlOrRd')
    annotator.configure(test='Mann-Whitney', text_format='star')
    annotator.apply_and_annotate()

    plt.ylabel("Principal Component")
    plt.xlabel("Lesion Site")

    ax.get_legend().remove()

    plt.legend(loc='lower center',bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    apply_style()

def check_pca_figure(df_disc, df_clinical_disc, df_test, df_clinical_test, site='PC'):
    df_disc = df_disc[df_disc.index.isin(['pertubation-radiomics'], level='job_tag')]
    df_test = df_test[df_test.index.isin(['pertubation-radiomics'], level='job_tag')]

    df_ALL = df_disc[df_disc.index.isin(['PC', 'PL', 'LN'], level='site')]
    df_PC_test  = df_test[df_test.index.isin([site], level='site')]
    df_PC_disc  = df_disc[df_disc.index.isin([site], level='site')]


    transform = rf_pca(
        df_ALL,
        hue_series=None,
        plot=False,
    )

    rf_pca(
        df_PC_disc,
        hue_series=df_clinical_disc.loc[df_PC_disc.index.get_level_values(0),'label'].replace({1:'Discovery POD-PC', 0:'Discovery PR/CR-PC'}), 
        transform = transform,
        palette='PuBu',
        hold=True
    )
  
    rf_pca(
        df_PC_test,
        hue_series=df_clinical_test.loc[df_PC_test.index.get_level_values(0),'label'].replace({1:'Test POD-PC', 0:'Test PR/CR-PC'}), 
        transform = transform,
        palette='OrRd',
        init=False,
    )
  
def check_features_figure(df_disc, df_clinical_disc, df_test, df_clinical_test, fx, site='PC'):
    df_disc = df_disc[df_disc.index.isin([RAD_JOB_TAG], level='job_tag')]
    df_test = df_test[df_test.index.isin([RAD_JOB_TAG], level='job_tag')]

    df_ALL = df_disc[df_disc.index.isin(['PC', 'PL', 'LN'], level='site')]
    df_PC_test  = df_test[df_test.index.isin([site], level='site')]
    df_PC_disc  = df_disc[df_disc.index.isin([site], level='site')]

    transform = rf_features(
        df_PC_disc,
        fx=fx,
        hue_series=None,
        plot=False,
    )
    rf_features(
        df_PC_disc,
        hue_series=df_clinical_disc.loc[df_PC_disc.index.get_level_values(0),'label'].replace({1:'Discovery POD-PC', 0:'Discovery PR/CR-PC'}), 
        fx=fx,
        transform = transform,
        palette='PuBu',
        hold=True
    )
  
    rf_features(
        df_PC_test,
        hue_series=df_clinical_test.loc[df_PC_test.index.get_level_values(0),'label'].replace({1:'Test POD-PC', 0:'Test PR/CR-PC'}), 
        fx=fx,
        transform = transform,
        palette='OrRd',
        init=False,
    )

def get_null_mean_error(model):
    if 'MILR' in model:
        df_permutations = pd.concat([pd.read_csv(csv_file_path).set_index('model') for csv_file_path in glob.glob('tests/null_permutation_MILR/*csv')])
    else:
        df_permutations = pd.concat([pd.read_csv(csv_file_path).set_index('model') for csv_file_path in glob.glob('tests/null_permutation/*csv')])

    
    try:
        df_permutations = df_permutations.loc[[model]]
    except:
        return 0.5, 0

    model_mean  = df_permutations['auc'].mean()
    sem_error   =  df_permutations['auc'].sem()
    prop_error  = np.sqrt(df_permutations['error'].pow(2).mean())

    return model_mean, np.sqrt(sem_error**2 + prop_error**2)

from sklearn.metrics import roc_curve, auc

def generate_auc_fold_plot(labels, scores, title='Receiver operating characteristic', merge=True, average=True, figsize=(6,6), max_folds=10, names=None, extra_opts=None):
    """ Makes the nice fold-wise, average, and merged AUROC plots
    labels: a list of label arrays
    scores: a list of score arrays
    title: A title for the plot
    figsize: Figure size
    merge: Whether to merge the list of labels, scores in CV fashion (e.g. set to false for subsampling)
    max_folds: Maximum number of folds to show
    """
    if extra_opts is None: extra_opts = {}
        
    plt.figure(figsize=figsize)
    tprs = []
    aucs = []
    
    l_v_scores =[]
    l_v_labels = []

    mean_fpr = np.linspace(0, 1, 100)

    for i, (fold_labels, fold_scores) in enumerate(zip(labels, scores)):
        fpr, tpr, thresholds = roc_curve(fold_labels, fold_scores)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        
        if names is not None:
            label = '%s (AUC = %0.2f), N=%d patients' % (names[i], roc_auc, len(fold_labels))
            opts = {'alpha':1.0, 'lw':1}
            opts.update(extra_opts.get(names[i], {}))
        else:
            label = 'Fold %d (AUC = %0.2f), N=%d patients' % (i + 1, roc_auc, len(fold_labels))
            opts = {'alpha':0.5, 'lw':1}
            alpha = 0.5
            
        if max_folds is None or i < max_folds: 
            plt.plot(fpr, tpr, **opts,  label=label)
        else:
            plt.plot(fpr, tpr, **opts)

        # print ('Fold %d (AUC = %0.2f), N=%d' % (i + 1, roc_auc, len(fold_labels)))
        
        if merge:
            l_v_scores.extend(fold_scores)
            l_v_labels.extend(fold_labels)

    if merge:
        fpr, tpr, thresholds = roc_curve(l_v_labels, l_v_scores)
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, color='k', label='Merged ROC (AUC = %0.2f), N=%d patients' % (roc_auc, len(l_v_labels)))

    if average: 
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f), N=%d folds' % (mean_auc, std_auc, len (tprs)),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f'vector_figs/sup_{title}.svg', bbox_inches='tight')
    plt.close()

def generate_auc_plot_V2(d_summarys_dfs, models, annot_list=None, show_errors=False, ss_aucs = None, panel= None):
  
    if ss_aucs is None: ss_aucs = {}

    xnames = []
    aucs = []
    errs = []
    colors = []
    sigmas = []

    null_bottoms = []
    null_tops = []
    null_heights = []
    null_means = []

    ss_means = []
    ss_tops = []
    ss_bottoms = []
    ss_heights = []
    ss_meds = []

    for num, model in enumerate(models.keys()):

        if model in d_summarys_dfs.keys():
            df = d_summarys_dfs[model]
            auc, ci = auc_roc_ci((df['label']), df['score'], 0.95)

            print (f" >>> {model}, AUC={auc:.2f}, 95% CI {ci[0]:.2f}-{ci[1]:.2f}")

            if 'fold' in df:
                folds = df['fold'].unique()
                folds.sort()
                score_list = [df.loc[df.fold==fold, 'score'] for fold in folds]
                label_list = [df.loc[df.fold==fold, 'label'] for fold in folds]
                generate_auc_fold_plot(label_list, score_list, title=f'ROC 10-CV {model}', merge=True, max_folds= None, figsize=(12, 12))

        else:
            auc, ci = 0, (0, 0)
        # print (auc, ci)
        xnames.append(f"{model}")
        aucs.append (auc)
        errs.append ((ci[1]-ci[0]) / 2.0)
        colors.append(models[model]['color'])

        null_mean, null_err = get_null_mean_error(model)
        null_bottoms.append (null_mean-null_err)
        null_heights.append (2 * null_err)
        null_means.append (null_mean)
        null_tops.append( null_mean + null_err)

        if model in d_summarys_dfs.keys():

            auc, ci = auc_roc_ci((df['label']), df['score'], 0.68)
            sigma_deviation = np.abs(auc - null_mean) / ((((ci[1]-ci[0]) / 2.0) + null_err) / 2.0) 
        else:
            sigma_deviation = 0

        sigmas.append( sigma_deviation )

        if ss_aucs.get(model, None) is not None:
            ss_mean, ss_err = np.array(ss_aucs.get(model))[:, 0].mean(), ( np.array(ss_aucs.get(model))[:, 0] ).std()

            ss_med = np.median (np.array(ss_aucs.get(model))[:, 0])

            ss_labels = np.array(ss_aucs.get(model))[:, 3]
            ss_scores = np.array(ss_aucs.get(model))[:, 2]

            generate_auc_fold_plot(ss_labels, ss_scores, title=f'ROC Subsample {model}', merge=False, max_folds= 10, figsize=(12, 12))
        else:
            ss_mean, ss_err, ss_med = 0, 0, 0

        ss_means.append(ss_mean)
        ss_bottoms.append (ss_mean - ss_err )
        ss_tops.append (ss_mean + ss_err )
        ss_heights.append (2 * ss_err )
        ss_meds.append(ss_med)


    plt.figure(figsize=((len(models) + 1), 8))

    out = {
        'model': xnames,
        'auc': aucs,
        'auc_upper_95CI': np.array(aucs) + np.array(errs),
        'auc_lower_95CI': np.array(aucs) - np.array(errs),
    }
    
    if show_errors:
        out.update( {
            'null_upper_1sigma': null_tops,
            'null_lower_1sigma': null_bottoms,
            'subsample_upper_1sigma': ss_tops,
            'subsample_lower_1sigma': ss_bottoms,     
        }
        )

    print (panel)
    if panel is not None:
        print (pd.DataFrame(out))
        pd.DataFrame(out).to_excel(f'./excel/{panel}.xlsx', sheet_name=panel)

    bars = plt.bar(xnames, aucs, bottom=0., yerr=errs, color=colors, capsize=4, edgecolor = (0,0,0, 0.8))
    if show_errors:
        for i, bar in enumerate(bars): 
            bbox = bar.get_bbox()
            plt.plot([bbox.x0, bbox.x1],[bbox.y1, bbox.y1], color='k', lw=3)

            stars = '*' * min(int(sigmas[i]), 4)

            plt.text((bbox.x0+bbox.x1)/2.0, bbox.y1 + errs[i] + 0.005, stars, weight='bold', va='center', ha='center',  fontsize=FONT_BASE['size'] * 1.5)

    plt.xticks(rotation=35, ha='right')

    plt.ylim(0.5, 1)

    plt.ylabel("AUC")
    # plt.xlabel("Model")

    if annot_list:
        for section in annot_list:
            left = -1
            right  = -1
            for num, model in enumerate(models.keys()):
                if model == section['left']: left=num-0.5
                if model == section['right']: right=num-0.5
            if not left > -1 and right > -1: continue

            plt.plot([left, left], [0.5, 1], 'k--')
            plt.text((left+right)/2.0 + 0.5, 0.93, section['name'], weight='bold', va='bottom', ha='center')

    plt.xlim(-0.5, len(xnames) - 0.5 )


    if show_errors:
        bars = plt.bar(xnames, null_heights, bottom=null_bottoms, color='None', fc='None', hatch='//', edgecolor = (0,0,0, 0.8))
        for i, bar in enumerate(bars): 
            bbox = bar.get_bbox()
            bar.set_edgecolor('r')
            plt.plot([bbox.x0, bbox.x1],[null_means[i],null_means[i]], 'r--', lw=1)

    if ss_aucs is not None:
        bars = plt.bar(xnames, ss_heights, bottom=ss_bottoms, color='None', alpha=0.5, hatch='\\\\\\\\', edgecolor = (0,0,0, 0.8))
        for i, bar in enumerate(bars): 
            bbox = bar.get_bbox()
            # plt.plot([bbox.x0, bbox.x1],[ss_means[i],ss_means[i]], 'r', lw=3)
            # plt.plot([bbox.x0, bbox.x1],[ss_meds[i],ss_meds[i]], 'g',  lw=3)
            
    apply_style()
    # plt.errorbar(aucs, range(len(xnames)), xerr=errs, color='k', ecolor=colors, elinewidth=3, fmt='o')
    # plt.yticks(range(len(xnames)), xnames)
    # plt.ylim(-1,(len(models)))
    # if xlims:
    #     plt.xlim(xlims[0], xlims[1])
    # plt.ylim(0.45, None)
    return bars

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score 
def generate_metric_plot_V2(d_summarys_dfs, models, annot_list=None, panel=None):
    out = {}

    for metric in ['F1-score', 'Precision Score', 'Recall Score', 'AUC', 'Accuracy']:

        xnames = []
        aucs = []
        colors = []

        for num, model in enumerate(models.keys()):
        #     if not "fusion" in model: continue
            if model in d_summarys_dfs.keys():
                df = d_summarys_dfs[model]

                if metric == 'F1-score':
                    model_metric = f1_score(df['label'], np.where(df['score'] > 0, 1, 0))
                if metric == 'Precision Score':
                    model_metric = precision_score(df['label'], np.where(df['score'] > 0, 1, 0))
                if metric == 'Recall Score':
                    model_metric = recall_score(df['label'], np.where(df['score'] > 0, 1, 0))
                if metric == 'AUC':
                    model_metric = roc_auc_score(df['label'], df['score'])
                if metric == 'Accuracy':
                    model_metric = accuracy_score(df['label'], np.where(df['score'] > 0, 1, 0))
                print (f" >>> {model}, {metric}={model_metric:.2f}")

            else:
                auc, ci = 0, (0, 0)
            # print (auc, ci)
            xnames.append(f"{model}")
            aucs.append (model_metric)
            colors.append(models[model]['color'])

        plt.figure(figsize=((len(models) + 1), 8))

        out['Model'] = xnames
        out[metric] = aucs

        if panel is not None:
            print (pd.DataFrame(out))
            pd.DataFrame(out).to_excel(f'./excel/{panel}.xlsx', sheet_name=panel)

        bars = plt.bar(xnames, aucs, bottom=0., color=colors, capsize=4, edgecolor = (0,0,0, 0.8))
        for i, bar in enumerate(bars): 
            bbox = bar.get_bbox()
            plt.plot([bbox.x0, bbox.x1],[bbox.y1, bbox.y1], color='k', lw=3)
        plt.xticks(rotation=35, ha='right')

        plt.ylim(0., 1)

        plt.ylabel(metric)
        # plt.xlabel("Model")

        if annot_list:
            for section in annot_list:
                left = -1
                right  = -1
                for num, model in enumerate(models.keys()):
                    if model == section['left']: left=num-0.5
                    if model == section['right']: right=num-0.5
                if not left > -1 and right > -1: continue

                plt.plot([left, left], [0., 1], 'k--')
                plt.text((left+right)/2.0 + 0.5, 0.93, section['name'], weight='bold', va='bottom', ha='center')

        plt.xlim(-0.5, len(xnames) - 0.5 )
        apply_style()
        # plt.show()

        plt.savefig(f"vector_figs/{panel}-{metric}.svg", bbox_inches='tight')
        plt.show()
    return pd.DataFrame(out)

def make_rad_violin_plots(df):
    df = df.reset_index().set_index(['main_index', 'job_tag', 'lesion_index'])
    df = df[df.index.isin([RAD_JOB_TAG], level='job_tag')]

    hue_order= list(df.index.get_level_values('lesion_index').sort_values().unique().astype(int))

    print ("PC", df.query(f"lesion_index == 1 or lesion_index == 2")[['original_shape_MeshVolume', 'original_shape_MajorAxisLength', 'original_shape_MinorAxisLength', 'original_shape_Sphericity']].describe())
    print ("PL", df.query(f"lesion_index == 3 or lesion_index == 4")[['original_shape_MeshVolume', 'original_shape_MajorAxisLength', 'original_shape_MinorAxisLength', 'original_shape_Sphericity']].describe())
    print ("LN", df.query(f"lesion_index == 5 or lesion_index == 6")[['original_shape_MeshVolume', 'original_shape_MajorAxisLength', 'original_shape_MinorAxisLength', 'original_shape_Sphericity']].describe())

    plt.figure(figsize=(30,10))
    plt.subplot(1,3,1)
    melted = df.reset_index().drop(columns="main_index").melt(id_vars=['lesion_index'], var_name='RF', value_name='vals')
    
    melted = melted[ melted["RF"].isin(["original_shape_MinorAxisLength", "original_shape_MajorAxisLength"])]
    melted['lesion_index'] = melted['lesion_index'].astype(int)
    melted['vals'] = melted['vals'].astype(float)

    ax = sns.violinplot(data=melted, x="RF", y='vals', hue='lesion_index', hue_order=hue_order, scale='width',cut=0, palette='rainbow_r')
    plt.tight_layout()
    plt.legend(ncol=3)
    plt.subplot(1,3,2)
    melted = df.reset_index().drop(columns="main_index").melt(id_vars=['lesion_index'], var_name='RF', value_name='vals')

    melted = melted[ melted["RF"].isin(["original_shape_Sphericity"])]
    melted['lesion_index'] = melted['lesion_index'].astype(int)
    melted['vals'] = melted['vals'].astype(float)
    ax = sns.violinplot(data=melted, x="RF", y='vals', hue='lesion_index', hue_order=hue_order, scale='width',cut=0, palette='rainbow_r')
    plt.legend(ncol=3)

    plt.subplot(1,3,3)
    melted = df.reset_index().drop(columns="main_index").melt(id_vars=['lesion_index'], var_name='RF', value_name='vals')

    melted = melted[ melted["RF"].isin(["original_shape_MeshVolume"])]
    melted['lesion_index'] = melted['lesion_index'].astype(int)
    melted['vals'] = melted['vals'].astype(float)
    ax = sns.violinplot(data=melted, x="RF", y='vals', hue='lesion_index', hue_order=hue_order, scale='width',cut=0, palette='rainbow_r')
    plt.legend(ncol=3)
    plt.ylim(0, 100000)
    plt.show()

def make_path_violin_plots(df, df_clinical, ylims=None):

#    df.columns = [col.replace('pixel_original_', '').replace('channel_1_', '').replace('scale_None_', '').replace('glcm_Autocorrelation', 'GLCM autocorrelation').replace('glcm_None', 'Pixel intensity').replace('_', ' ') for col in df.columns]
    
    df.columns = [col.replace('original_pixels', 'original_pixels_PD-L1 Pixel Intensity') for col in df.columns]
    df.columns = [col.replace('_', '__') for col in df.columns]
    df.columns = [col.replace('lognorm__fit__p0', r'lognorm $P_0$') for col in df.columns]
    df.columns = [col.split('__')[2] + ' ' + col.split('__')[5] for col in df.columns]

    plt.figure(figsize=(20,5))
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    df_clinical['Outcome'] = df_clinical['label'].replace({1:'SD/PD', 0:'PR/CR'})

    melted = df.join(df_clinical['Outcome']).melt(id_vars=['Outcome'], var_name='PD-L1 Quantification Feature', value_name='Normalized Value [A.U.]')
    print (melted)
    melted.to_excel('./excel/3B.xlsx', sheet_name='3B')
    ax = sns.violinplot(data=melted, x="PD-L1 Quantification Feature", y='Normalized Value [A.U.]',  cut=0, hue='Outcome',  hue_order=['PR/CR', 'SD/PD'], split=False, palette=['#a6bddb', '#0570b0'], orient='v')
    if ylims:
        plt.ylim(ylims[0], ylims[1])

    # ax.set_xticks(list(ax.get_xticks()))

    plt.xticks(list(ax.get_xticks()), rotation=30, ha='right')

    pairs = [
        [(col, 'SD/PD'), (col, 'PR/CR')] for col in df.columns
    ]

    annotator = Annotator(ax, pairs, data=melted, x="PD-L1 Quantification Feature", y='Normalized Value [A.U.]', hue='Outcome')
    annotator.configure(test='t-test_ind', text_format='star')

    annotator.apply_test()

    print ([annotation.data.pvalue for annotation in annotator.annotations] [0])

    annotator.annotations = np.array(annotator.annotations)[ [annotation.data.pvalue < 0.05 for annotation in annotator.annotations]  ] # only show signficant ones!

#     annotator.annotate()
    # plt.xlim(-1.5, None)
    plt.ylim(-5, 5)

    # annotator.apply_and_annotate()

    plt.legend(frameon=False, loc='lower left')
    apply_style()

def make_mask_violin_plots(df, df_clinical, ylims=None):

    plt.figure(figsize=(20,5))

    df_clinical['Outcome'] = df_clinical['label'].replace({1:'SD/PD', 0:'PR/CR'})

    melted = df.join(df_clinical['Outcome']).melt(id_vars=['Outcome'], var_name='Data Availability', value_name='Normalized Value [A.U.]')
    ax = sns.violinplot(data=melted, x="Data Availability", y='Normalized Value [A.U.]',  cut=0, hue='Outcome',  hue_order=['PR/CR', 'SD/PD'], split=False, palette=['#a6bddb', '#0570b0'], orient='v')
    if ylims:
        plt.ylim(ylims[0], ylims[1])

    # ax.set_xticks(list(ax.get_xticks()))

    plt.xticks(list(ax.get_xticks()), rotation=30, ha='right')

    pairs = [
        [(col, 'SD/PD'), (col, 'PR/CR')] for col in df.columns
    ]

    annotator = Annotator(ax, pairs, data=melted, x="Data Availability", y='Normalized Value [A.U.]', hue='Outcome')
    annotator.configure(test='Mann-Whitney', text_format='star', comparisons_correction='Benjamini-Yekutieli')

    annotator.apply_test()

    annotator.annotations = np.array(annotator.annotations)[ [annotation.data.corrected_significance for annotation in annotator.annotations]  ] # only show signficant ones!

    # annotator.annotate()

    # annotator.apply_and_annotate()

    plt.legend(frameon=False, loc='lower right')
    apply_style()

def apply_style(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

def make_general_violin_plot(df, df_clinical, ylims=None, units=None):
    df = df.drop(columns=df.columns[df.columns.str.contains("nobs|min")])

    df.columns = [col.replace('pixel_original_', '').replace('scale_None_', '').replace('glcm_Autocorrelation_', 'GLCM Autocorrelation').replace('glcm_None_', 'Pixel Intensity').replace('_', ' ') for col in df.columns]

    plt.figure(figsize=(12,8))

    # scaler = PowerTransformer()
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # df = df.fillna(df.median()).astype(float)
    df = df.astype(float)

    print (df)

    df_clinical['Outcome'] = df_clinical['label'].replace({1:'SD/PD', 0:'PR/CR'})

    for i_plot, col in enumerate(df.columns):
        plt.subplot(1,3,i_plot + 1)

        df_by_col = df[[col]].dropna()



        melted = df_by_col.join(df_clinical['Outcome']).melt(id_vars=['Outcome'], var_name='Feature', value_name=col)

        print (len(melted))

        print (melted)
        ax = sns.violinplot(data=melted, x="Feature", y=col, cut=0, hue='Outcome',  hue_order=['PR/CR', 'SD/PD'], split=False, palette=['#a6bddb', '#0570b0'], orient='v')
        ax.get_legend().remove()
        if ylims:
            plt.ylim(ylims[0], ylims[1])

        pairs = [
            [(col, 'SD/PD'), (col, 'PR/CR')] for col in df_by_col.columns
        ]
        plt.xlabel(None)
        if units: plt.ylabel(units[i_plot])
        if col == 'Num. Lesions': plt.ylim (ylims[0], 5)
        annotator = Annotator(ax, pairs, data=melted, x="Feature", y=col, hue='Outcome')
        annotator.configure(test='Mann-Whitney', text_format='star', show_test_name=False)
        annotator.apply_and_annotate()
        apply_style()


def get_training_data(modal_list, modal_dict, df_mask, df_outcomes):
    select_expr = ' | '.join([f"df_mask['{modality}']" for modality in modal_list])
    px_subset = eval(f"df_mask[({select_expr})].index")

    modality_INPUT_subset = [modal_dict[modality] for modality in modal_list]
    modality_MASK_subset  = df_mask.loc[px_subset, modal_list]
    outcomes_subset       = df_outcomes.loc[px_subset]

    return modality_INPUT_subset, modality_MASK_subset, outcomes_subset

def generate_alpine_plot(df, df_mask, x_axis="attn_gen_driver_mutations", aucs_to_show=None, label_mapping=None):
    if label_mapping is None: label_mapping = {}

    risk_cols = list(df.columns[df.columns.str.contains('risk')])
    print (risk_cols)

    aucs_to_show = copy.copy(aucs_to_show)

    if not aucs_to_show: aucs_to_show = risk_cols

    aucs_to_show.append('overall_score')

    x    = []
    N    = []
    ys   = {fx:[] for fx in risk_cols}
    yerr = {fx:[] for fx in risk_cols}

    ys['overall_score'] = []
    yerr['overall_score'] = []

    # for attn_cutoff in range(10):
    bbins = 9
    for qtile, attn_cutoff in enumerate(pd.qcut(df[df[x_axis]> 0][x_axis], bbins, retbins=True)[1]):
        if qtile==0: continue
    # for attn_cutoff in np.linspace(df[df[ATTN_SCORE]> 0][ATTN_SCORE].min(), df[ATTN_SCORE].max(), 10):
        
        df_below_cut = df[  (df[x_axis] <= attn_cutoff) ]

    #     print (classification_report(df_cut['label'], df_cut[col] > 0))
        if not (len(np.unique(df_below_cut['label']).astype(int)) == 2 and len(df_below_cut)>=10): continue

        # print (attn_cutoff, "N=", len(df_below_cut))
        print (qtile)
        x.append((100 / (bbins + 1)) * (qtile+1))
        N.append(len(df_below_cut))
        
        auc, ci = auc_roc_ci(df_below_cut['label'], df_below_cut['score'], 0.95)
        print (qtile, 'overall_score', auc, ci)

        if not ci[0] < 1.0 or not ci[1] < 1.0: 
            ci[0] = 0.0
            ci[1] = 0.0

        ys["overall_score"].append(auc)
        if auc == 0.5:
            yerr['overall_score'].append(0.)
        else:
            yerr['overall_score'].append((ci[1] - ci[0]) / 2.0)

        for fx in df.columns[df.columns.str.contains('risk')]:
            
            idx_modality = df_mask[df_mask[fx.replace('risk_', '')]].index.intersection(df_below_cut.index)
            
    #         print (idx_modality)
    #         auc, ci = auc_roc_ci(df_above_cut['label'], df_above_cut[fx], 0.95)
    #         print ("Post-fit Above [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL".format(fx, auc, (ci[1] - ci[0]) / 2.0) )
            try:
                auc, ci = auc_roc_ci(df_below_cut.loc[idx_modality, 'label'], df_below_cut.loc[idx_modality, fx], 0.95)
                print (qtile, fx, auc, ci)
    #             print ("Post-fit Below [{0}] AUC = {1:.3f} +/- {2:.3f} 95% CL N={3}".format(fx, auc, (ci[1] - ci[0]) / 2.0, len(df_below_cut.loc[idx_modality])))
            except:
                auc, ci = 0.5, (0.0, 0.0)
                
            if not ci[0] < 1.0 or not ci[1] < 1.0: 
                ci[0] = 0.0
                ci[1] = 0.0
        
            ys[fx].append(auc)
            if auc == 0.5:
                yerr[fx].append(0.0)
            else:
                yerr[fx].append((ci[1] - ci[0]) / 2.0)
            
    plt.figure(figsize=(8,8))
    # for key in ys.keys(): plt.errorbar(x, ys[key], yerr=yerr[fx], label=str(key), capsize=10)
    print (ys.keys())

    for i, key in enumerate(aucs_to_show): 
        # 300 represents number of points to make between T.min and T.max
        T = np.array(x)
        
        xnew = np.linspace(T.min(), T.max(), 100) 

        spl = make_interp_spline(x, ys[key], k=3)  # type: BSpline
        ynew = spl(xnew)
        
        up    = make_interp_spline(x, np.array(ys[key]) + np.array(yerr[key]), k=3)(xnew)
        down =  make_interp_spline(x, np.array(ys[key]) - np.array(yerr[key]), k=3)(xnew)

        if "risk" in key:
            c = sns.color_palette("Set2", len(aucs_to_show))[i]
        else:
            c = 'k'

        print (key, x, ys[key], yerr[key])
        # plt.plot(xnew, ynew,label=str(key), c=c)
        # plt.fill_between(xnew, up, down, color=c, alpha=0.05)
        # plt.plot(xnew, up , c=c, linestyle='--',  linewidth=1, alpha=0.2)
        # plt.plot(xnew, down , c=c, linestyle='--',  linewidth=1, alpha=0.2)

        plt.errorbar(x, ys[key], yerr=yerr[key], c=c, linestyle='--', marker='o', ms=10, capsize=3, label=label_mapping.get(str(key), str(key)))

    plt.xlabel(label_mapping.get(x_axis, x_axis) + " [Percentile]")
    plt.ylabel("Modality specific AUC of Post-fit risk scores")
    plt.xlim(-1, 101)
    # plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right', frameon=False)
    apply_style()
    # plt.figure(figsize=(8,2))
    # plt.ylim(0,275)
    # plt.plot(x, N, label="Number of Patients", c='k')
    # plt.ylabel("Number of Patients in cut")
    # plt.xlabel("Cutoff")

    cph = CoxPHFitter()
    cph.fit(df, duration_col='pfs', event_col='pfs_censor')
    cph.print_summary()

def generate_alpine_plot_V2(df, df_outcomes, x_axes="attn_gen_driver_mutations", stat='auc', label_mapping=None, panel=None):
    
    if label_mapping is None: label_mapping = {}

    df_timelines = df_outcomes[['pfs', 'pfs_censor']]

    risk_cols = list(df.columns[df.columns.str.contains('risk')])
    attn_cols = list(df.columns[df.columns.str.contains('attn')])

    plt.figure(figsize=(8,8))

    df_origional = df.copy(deep=True)

    print (attn_cols)

    if stat=="hr": ylabel = "Hazard Ratio after Reweighting"
    if stat=="auc": ylabel = "AUC after Reweighting"
    if stat=="kmf": ylabel = "PFS ratio (Q1 vs. Q4) after Reweighting"

    out = {}
    
    for i, x_axis in enumerate(x_axes):
        recombs = []

        c = sns.color_palette("Set2", len(x_axes))[i]

        x = []
        ys = []
        yerr = []
        og_ys = []
        og_yerr = []

        for exp_factor in np.linspace(-2.5, 2.5, 15):

            factor = pow(10, exp_factor)

            df = df_origional.copy(deep=True)
            # df[x_axis] += factor
            df[x_axis] *= factor

            for attn_col in attn_cols:
                df[attn_col] = df[attn_col].div(df[attn_cols].sum(axis=1))

            df_recomb = (
                (df[attn_cols].rename(columns={attn_col: attn_col.replace('attn_', '') for attn_col in attn_cols})) 
            *  (df[risk_cols].rename(columns={risk_col: risk_col.replace('risk_', '') for risk_col in risk_cols})))\
                .sum(axis=1).rename('recomb_score')
            recombs.append(df_recomb)

            df_tmp = df_origional.join(df_recomb)

            if stat=="auc":
                auc, ci = auc_roc_ci(df_tmp['label'], df_tmp['score'], 0.68)
                og_ys.append(auc)
                og_yerr.append((ci[1] - ci[0]) / 2.0)

                auc, ci = auc_roc_ci(df_tmp['label'], df_tmp['recomb_score'], 0.68)

                ys.append(auc)
                yerr.append((ci[1] - ci[0]) / 2.0)
            if stat=="hr":
                cph = CoxPHFitter()

                df_recomb_std = (df_recomb- df_recomb.min()) / (df_recomb.max() - df_recomb.min())
                cph.fit(df_timelines.join(df_recomb_std), duration_col='pfs', event_col='pfs_censor')

                ys.append(cph.hazard_ratios_['recomb_score'])
                yerr.append(cph.standard_errors_['recomb_score'])

            if stat=="kmf":
                time = 4
                df_event_data = df_timelines.join(df_recomb)

                q1 = pd.qcut(df_event_data['recomb_score'], 4, labels=False) == 0
                q2 = pd.qcut(df_event_data['recomb_score'], 4, labels=False) == 1
                q3 = pd.qcut(df_event_data['recomb_score'], 4, labels=False) == 2
                q4 = pd.qcut(df_event_data['recomb_score'], 4, labels=False) == 3

                df_pred_low  = df_event_data['recomb_score'].le(0)
                df_pred_high = df_event_data['recomb_score'].ge(0)

                df_q1 = q1
                df_q2 = q4

                kmf_q1 = KaplanMeierFitter()
                kmf_q1.fit(df_event_data.loc[df_q1, 'pfs'],  event_observed=df_event_data.loc[df_q1]['pfs_censor'], label="Q1")

                kmf_q2 = KaplanMeierFitter()
                kmf_q2.fit(df_event_data.loc[df_q2, 'pfs'],  event_observed=df_event_data.loc[df_q2]['pfs_censor'], label="Q2")

                surv_func_low  = kmf_q1.survival_function_at_times(time)[time]
                surv_func_high = kmf_q2.survival_function_at_times(time)[time]
                survival_ratio = surv_func_low / surv_func_high # Low risk divided by high risk, 2 means 2x more events in risk group than in non-risk group
                survival_ratio_err = np.sqrt ( (1-surv_func_low)/len(df_q1) + (1-surv_func_high)/len(df_q2)  ) * 2 # (~ 2x possion counting stat)

                # print (kmf_q1.confidence_interval_survival_function_)

                # print (kmf_q1.survival_function_at_times(time)[time], kmf_q2.survival_function_at_times(time)[time])

                ys.append(survival_ratio)
                yerr.append(survival_ratio_err)



            x.append(exp_factor)

        # print (label_mapping.get(str(x_axis), {'label':str(x_axis)})['label'], list(zip(x, ys, yerr)))

        label = label_mapping.get(str(x_axis), {'label':str(x_axis)})['label']

        out[label + ' Attention Multiplier'] = x
        out[label + ' ' + ylabel] = ys
        out[label + ' ' + ylabel + '_err'] = yerr
        

        plt.errorbar(x, ys, yerr=yerr, linestyle='--', c=label_mapping.get(str(x_axis), {'color':None})['color'], marker='o', ms=10, capsize=3, label=label_mapping.get(str(x_axis), {'label':str(x_axis)})['label'])
    plt.xlabel("Attention Multiplier (log10)")
    plt.ylabel(ylabel)
    
    if stat=='auc':
        plt.ylim(0.5, None)
    else:
        plt.ylim(0.0, None)

    if panel is not None:
        pd.DataFrame(out).to_excel(f'./excel/{panel}.xlsx', sheet_name=panel)

    plt.legend(loc='center left',bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)
    apply_style()
    df = pd.concat(recombs, axis=1)

def generate_cox_plot(dfs, df_outcomes, label_map=None, xlim=None):

    df = pd.concat(dfs, axis=1)
    scaler = MinMaxScaler()
#    scaler = StandardScaler()

    plt.figure(figsize=(9, 3.5 * len(df.columns)/6.0))

    df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    df = pd.concat([df, df_outcomes[['pfs', 'pfs_censor']]], axis=1)

    df = df.fillna(df.median(axis=0, skipna=True))
    df = df.rename(columns=label_map)
    cph = CoxPHFitter()
    cph.fit(df, duration_col='pfs', event_col='pfs_censor')
    cph.print_summary()

    for col in df.columns:
        auc, ci = auc_roc_ci((df_outcomes.loc[df.index, 'label']), df[col], 0.95)
        print ("AUC {1:.3f} +/- {2:.3f} 95% CL: [{0:20}] ".format(col, auc, (ci[1] - ci[0]) / 2.0) )
    cph.plot(hazard_ratios=False, marker='s', ms=8,  alpha=1.0, fillstyle='full', markerfacecolor='k')
    plt.gca().yaxis.grid(True)

    if xlim:
        plt.xlim(*xlim)
    apply_style()

from collections import defaultdict
def get_best_glcm(dfs, df_outcomes):

    cols = dfs.columns
    aucs = []
    errs = []

    best_sigma = defaultdict(float)
    best_fx = {}
    for col in cols:
        base_name = ''.join(col.split('_')[:3])
        auc, ci = auc_roc_ci((df_outcomes.loc[dfs.index, 'label']), dfs[col], 0.68)
        auc_sigma = np.abs(auc-0.5) / ((ci[1] - ci[0]) / 2.0)
        # print ("AUC sigmas", auc_sigma)

        if auc_sigma > best_sigma[base_name]:
            best_sigma[base_name] = auc_sigma
            best_fx[base_name] = col

        aucs.append( np.abs(auc -0.5) + 0.5)
        errs.append ((ci[1]-ci[0])/ 2.0)

    return [fx for fx in best_fx.values()]

def generate_cox_plot_V2(dfs, df_outcomes, label_map=None, xlim=None, panel=None):
    df = pd.concat(dfs, axis=1)
    scaler = MinMaxScaler()
#    scaler = StandardScaler()

    df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    df = pd.concat([df, df_outcomes[['pfs', 'pfs_censor']]], axis=1)

    df = df.fillna(df.median(axis=0, skipna=True))
    df = df.rename(columns=label_map)
    cph = CoxPHFitter()
    cph.fit(df, duration_col='pfs', event_col='pfs_censor')

    df_summary = cph.summary

    df_hrs = cph.hazard_ratios_.sort_values(ascending=True)

    size = len(df_hrs.index)
    cols = df_hrs.index

    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True, gridspec_kw={'width_ratios': [3, 0.75, 1]})
    fig.subplots_adjust(wspace=0)

    aucs = []
    errs = []
    for col in cols:
        auc, ci = auc_roc_ci((df_outcomes.loc[df.index, 'label']), df[col], 0.95)
        print ("AUC {1:.3f} +/- {2:.3f} 95% CL: [{0:20}] ".format(col, auc, (ci[1] - ci[0]) / 2.0) )
        aucs.append( np.abs(auc -0.5) + 0.5)
        errs.append ((ci[1]-ci[0])/ 2.0)
        df_summary.loc[col, 'auc'] = np.abs(auc -0.5) + 0.5
        df_summary.loc[col, 'auc_lower_95CI'] = np.abs(auc -0.5) + 0.5 - (ci[1]-ci[0])/ 2.0
        df_summary.loc[col, 'auc_upper_95CI'] = np.abs(auc -0.5) + 0.5 + (ci[1]-ci[0])/ 2.0
    cph.plot(ax=axs[0], hazard_ratios=False, marker='s', ms=8,  alpha=1.0, fillstyle='full', markerfacecolor='k')
    axs[0].yaxis.grid(True)
    apply_style(axs[0])

    if panel is not None:
        print (panel)
        df_summary.to_excel(f'./excel/{panel}.xlsx', sheet_name=panel)

    axs[1].barh(cols.values, [0 for col in cols])
    axs[1].set_xlim(0, 1)

    for i_col, i in enumerate(axs[1].patches):

        if df_summary.loc[cols[i_col], 'p'] < 0.01:
            pval_text = rf"$p = ${df_summary.loc[cols[i_col], 'p']:.2e}"
        else:
            pval_text = rf"$p = ${df_summary.loc[cols[i_col], 'p']:.2f}"

        axs[1].text(i.get_x() + 0.1, i.get_y() + 0.5 * i.get_height(),
                pval_text,
                fontsize = 18, fontweight ='bold', va='center',
                color ='black')    

    axs[2].barh(cols.values, aucs,  xerr=errs, capsize=4, color='grey')
    axs[2].set_xlabel("AUC")
    axs[2].yaxis.grid(True)
    axs[2].set_xlim(0.5, 0.9)

    apply_style(axs[2])
    # apply_style(axs[2])
    axs[1].axis('off')

def make_ihc_vs_pdl1_plot(df_pdl1, df, fx_name, colors_list, ylabel=None):
    plt.figure(figsize=(8,8))
    # colors_list = ['#2D719F', '#FFD166',  '#EE2F5C']
    # colors_list = ['#F7EC59', '#92DCE5',  '#F9564F']

    df_pdl1 = df_pdl1.dropna().astype(float)
    df_pdl1['TPS Category'] = 'Zero'
    df_pdl1.loc[df_pdl1['Sauter PD-L1 Score'].ge(1),  'TPS Category']   = 'Low\n\u2265 1%, <25%'
    df_pdl1.loc[df_pdl1['Sauter PD-L1 Score'].ge(25), 'TPS Category']   = 'Moderate\n≥ 25%, < 75%'
    df_pdl1.loc[df_pdl1['Sauter PD-L1 Score'].ge(75), 'TPS Category']   = 'High\n≥ 75%'

    df = df_pdl1.drop('Sauter PD-L1 Score', axis=1).join(df)

    print (df_pdl1)

    order = ['Zero', 'Low\n\u2265 1%, <25%', 'Moderate\n≥ 25%, < 75%', 'High\n≥ 75%']
    order = order[1:4] # Exclude zero

    print (order)

    ax = sns.violinplot(data=df, x='TPS Category', y=fx_name, order=order, cut=0, palette=colors_list)
    sns.swarmplot(data=df, x='TPS Category', y=fx_name, order=order, color='k')

    pairs = []
    for i, col_i in enumerate(order):
        for j, col_j in enumerate(order):
            if col_i!=col_j and i < j:
                pairs.append((col_i, col_j))

    annotator = Annotator(ax, pairs, data=df, x='TPS Category', y=fx_name, order=order)
    annotator.configure(test='Mann-Whitney', text_format='star')
    annotator.apply_and_annotate()

    if ylabel:
        plt.ylabel(ylabel)

    apply_style()

    return df

def make_ihc_vs_pdl1_plot_V2(df_pdl1, df, fx_names, fx_labels, colors_list, ylabel=None):
    plt.figure(figsize=(8,8))
    # colors_list = ['#2D719F', '#FFD166',  '#EE2F5C']
    # colors_list = ['#F7EC59', '#92DCE5',  '#F9564F']

    df_pdl1 = df_pdl1.dropna().astype(float)
    df_pdl1['TPS Category'] = 'Zero'
    df_pdl1.loc[df_pdl1['Sauter PD-L1 Score'].ge(1),  'TPS Category']   = 'Low\n[1%, 25%]'
    df_pdl1.loc[df_pdl1['Sauter PD-L1 Score'].ge(25), 'TPS Category']   = 'Moderate\n[25%, 75%]'
    df_pdl1.loc[df_pdl1['Sauter PD-L1 Score'].ge(75), 'TPS Category']   = 'High\n[75%, 100%]'

    df = df_pdl1.drop('Sauter PD-L1 Score', axis=1).join(df)

    order = ['Zero', 'Low\n[1%, 25%]', 'Moderate\n[25%, 75%]', 'High\n[75%, 100%]']
    order = order[1:4] # Exclude zero


    df[fx_names] = (df[fx_names] - df[fx_names].mean()) / (df[fx_names].std() )

    df= df.rename(columns={col:lab for col, lab in zip(fx_names, fx_labels)})

    df = df.melt(id_vars=['TPS Category'], value_vars=fx_labels, value_name='value')


    print (df)
    df.to_excel('./excel/3D.xlsx', sheet_name='3D')
    ax = sns.violinplot(data=df, x='TPS Category', y='value', hue='variable', order=order, cut=0, palette=colors_list, scale='count')
    # sns.swarmplot(data=df, x='TPS Category', y='value', hue='variable', order=order, color='k')

    pairs = []
    for fx in fx_labels:
        for i, col_i in enumerate(order):
            for j, col_j in enumerate(order):
                if col_i!=col_j and i < j:
                    pairs.append( ( ( col_i, fx), (col_j, fx) ) )

    # annotator = Annotator(ax, pairs, data=df, x='TPS Category', y='value',hue='variable',  order=order)
    # annotator.configure(test='t-test_ind', text_format='star')
    # annotator.apply_and_annotate()
    plt.legend(frameon=False, loc='upper right')
    if ylabel:
        plt.ylabel(ylabel)

    # plt.ylim(0, 1.5)
    apply_style()

    return df

def make_bar_plot(df, idx_col, val_col, color):

    # sns.set(font_scale = 3)

    plt.figure(figsize=(8,8))

    df = df.dropna(subset=[val_col])

    df[val_col] = df[val_col].astype(float)
    df = df.sort_values(val_col, ascending = False).reset_index()


    bp = sns.barplot(x = idx_col, y = val_col, data = df, label = 'Total', color = color, edgecolor = None, linewidth=0)
    bp.set(xticklabels=[])

    # sns.set(font_scale = 1)


    plt.xlabel("Ordered Patients")

def make_dist_plot(df, val_col, color):

    # sns.set(font_scale = 1)

    plt.figure()

    df = df.dropna(subset=[val_col])

    bp = sns.displot(x = val_col, data = df, color = color, edgecolor = None, linewidth=0)

    # sns.set(font_scale = 1)

def get_inter_intra_var(df, df_outcome):
    df_rand = df[df.index.isin(['pertubation-radiomics'], level='job_tag')]
    df_filt = df[df.index.isin(['filtered-radiomics'], level='job_tag')]

    # outlier_counts,  sel_fx_outlier = select_outlier_stable_features(df, cutoff=6)
    # variance_scores, sel_fx_robust  = select_by_interlesion_variance(df, cutoff=0.15)
    # fx_to_use = sorted(set(sel_fx_outlier).intersection(set(sel_fx_robust)))

    fx_to_use, df_coef = select_radiomics_features_elastic(df.reset_index().set_index('main_index'), df_outcome, robustness_cutoff=0.15, outlier_cutoff=6, l1_strength=0.1) 
    print (len(fx_to_use))

    df_rand_ALL = df_rand[df_rand.index.isin(['PC', 'PL', 'LN'], level='site')]
    df_filt_ALL = df_filt[df_filt.index.isin(['PC', 'PL', 'LN'], level='site')]

    df_rand_ALL['global_lesion_id'] = df_rand_ALL.index.get_level_values(0) + '-' + df_rand_ALL['lesion_index'].astype(str)
    df_filt_ALL['global_lesion_id'] = df_filt_ALL.index.get_level_values(0) + '-' + df_filt_ALL['lesion_index'].astype(str)

#     if lesion_ids is None:
#         lesion_ids = np.random.default_rng(4).choice(df_rand_ALL.set_index('global_lesion_id').index.unique(), size=20, replace=False)
    df_filt_ALL = df_filt_ALL.set_index('global_lesion_id')[fx_to_use]

    scaler = PowerTransformer()
    pca = PCA(n_components=2, whiten=True)
    df_rand_sub = pd.DataFrame(pca.fit_transform(scaler.fit_transform(df_filt_ALL)), columns=[f'pc{i}' for i in range(2)], index=df_filt_ALL.index)

#     transform = rf_pca(
#         df_filt_ALL.reset_index().set_index(['main_index', 'global_lesion_id'])[fx_to_use],
#         hue_series=df_filt_ALL['global_lesion_id'],
#         plot=False,
#     )

#     scaler, pca = transform
#     df_rand_sub = pd.DataFrame(pca.transform(scaler.transform(df_rand_ALL[fx_to_use])), columns=[f'pc{i}' for i in range(2)], index=df_rand_ALL.index).reset_index()
#     df_filt_sub = pd.DataFrame(pca.transform(scaler.transform(df_filt_ALL[fx_to_use])), columns=[f'pc{i}' for i in range(2)], index=df_filt_ALL.index)

#     modality_site = df_rand_sub \
#         .set_index(['main_index', "lesion_index"])\
#         .groupby(level=[0]).agg(np.mean) \
#         .join(modality_mask[name], how='right').drop(columns=name)
#     df_rand_sub = df_rand_sub.set_index('main_index')
#     interlesion_variance = df_rand_sub.groupby(level=[0,1]).agg(np.std).mean() / df.groupby(level=[0,1]).agg(np.mean).std()
#     return df_rand_sub
#     return df_rand_sub, df_filt_sub


def plot_coefs(df, try_nice_columns=False, panel=None):
    """
    "Being selected is defined as meeting the following criteria: 
        (1) if applicable, the feature was selected as part of the radiomics filtering process
        (2) the magnitutde of the coefficent is at least 1 standard deviation from zero
        (3) the magntitude of the coefficent is greater than 0.01
    Only coefficients selected more than once are shown
    """
    df = df.loc['coef']

    if try_nice_columns: 
        df.columns = [col.replace('original_pixels', 'original_pixels_PD-L1 Pixel Intensity') for col in df.columns]
        df.columns = [col.replace('_', '__') for col in df.columns]
        df.columns = [col.replace('lognorm__fit__p0', r'lognorm $P_0$') for col in df.columns]
        df.columns = [col.split('__')[2] + ' ' + col.split('__')[5] for col in df.columns]

    
    df[df.isna()] = np.nan # (1)
    df[(df.abs() / df.std() < 1) ] = np.nan # (2)
    df[df.abs() < 0.01] = np.nan # (3)

    s_counts = ((~df.isna()).sum()).rename('counts')
    s_counts = s_counts[s_counts > 1]
    df.loc[:, s_counts.index]

    s_mag = (df.mean().abs()).rename('mag')

    sort_index = pd.concat([s_counts, s_mag], axis=1).sort_values(by=['counts', 'mag'], ascending=False).index

    sort_index = sort_index.intersection(s_counts.index)

    if panel is not None:
        pd.concat ([df.loc[:, sort_index], pd.DataFrame(s_counts).T]).to_excel(f'./excel/{panel}.xlsx', sheet_name=panel)

    if len(sort_index) > 100:
        sort_index = sort_index[:100]

    df = df.loc[:, sort_index]
    s_counts = s_counts[sort_index]

    size = len(df.columns)
    print (size)

    fig, axs = plt.subplots(1, 2, figsize=(20, size / 2.5), sharey=True, gridspec_kw={'width_ratios': [3, 1]})
    fig.subplots_adjust(wspace=0)
    
    sns.boxplot(data=df, orient="h", palette="Set2", ax=axs[0], linewidth=2.0, boxprops={'facecolor':'none'})
    axs[0].plot([0, 0], [-1, size], color='r', linestyle='--')
    axs[0].set_xlabel("LR Coefficient Values")
    axs[0].grid()

    sns.barplot(x=s_counts.values, y=s_counts.index, ax=axs[1], palette='husl')
    axs[1].set_xlabel("Num. Times Selected")
    axs[1].grid()


    
def save_table_1A(df_clinical):
    df_genomic = pd.read_csv("./omnibus/genomic_inventory_v1_FINAL.csv")
    df_genomic['main_index'] = df_genomic['Patient ID']
    df_genomic = df_genomic.set_index('main_index')
    df_genomic = df_genomic.drop(columns=['Study ID','Sample ID', 'Patient ID'])

    df_comb = df_clinical[['label', 'age', 'pack_years', 'n_lesions', 'js_pdl1_score']].join(df_genomic)

    df_comb = df_comb.rename(columns={'label':'Response', 'age':'Age', 'pack_years':'Pack Years', 'n_lesions':'Num. Lesions', 'js_pdl1_score':'PD-L1 Score'})

    df_comb.to_excel('./excel/1A.xlsx', sheet_name='1A')

    return df_comb

def save_table_1A(df_clinical):
    df_genomic = pd.read_csv("./omnibus/genomic_inventory_v1_FINAL.csv")
    df_genomic['main_index'] = df_genomic['Patient ID']
    df_genomic = df_genomic.set_index('main_index')
    df_genomic = df_genomic.drop(columns=['Study ID','Sample ID', 'Patient ID'])

    df_comb = df_clinical[['label', 'age', 'pack_years', 'n_lesions', 'js_pdl1_score']].join(df_genomic)

    df_comb = df_comb.rename(columns={'label':'Response', 'age':'Age', 'pack_years':'Pack Years', 'n_lesions':'Num. Lesions', 'js_pdl1_score':'PD-L1 Score'})
    df_comb['Response'] = df_comb['Response'].replace({1:'SD/PD', 0:'PR/CR'})

    df_comb.to_excel('./excel/1A.xlsx', sheet_name='1A')

    return df_comb

def save_table_1B(df_clinical):
    df_comb = df_clinical[['histo']].rename(columns={'histo':'Histology'})
    df_comb.to_excel('./excel/1B.xlsx', sheet_name='1B')


def save_table_1D(df_clinical, df_genomic):
    df_comb = df_clinical[['label','n_lesions', 'js_pdl1_score']].join(df_genomic[['TMB']]).rename(columns={'label':'Response', 'n_lesions':'Num. Lesions', 'js_pdl1_score':'TPS'})
    df_comb['Response'] = df_comb['Response'].replace({1:'SD/PD', 0:'PR/CR'})

    df_comb.to_excel('./excel/1D.xlsx', sheet_name='1D')
