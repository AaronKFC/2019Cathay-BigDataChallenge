# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 16:30:09 2021

@author: AllenPC
"""


##############################################################################
###########################preproceesing.py ##################################

import pandas as pd
import numpy as np
# import seaborn as sns

data=np.loadtxt('data/train.csv',dtype=np.str,delimiter=',',encoding='big5')[1:]
# data=np.loadtxt('data/test.csv',dtype=np.str,delimiter=',',encoding='big5')[1:]
df=pd.read_csv('data/train.csv',encoding='big5')
# df=pd.read_csv('data/test.csv',encoding='big5')
df['Y1'] = df['Y1'].map({'Y':1, 'N':0})
df_cp = df.copy()

def FT_27to48_ISSUE_and_ADD_IND(df):
    
    df_ISSUE = df.iloc[:,26:48]
    for i in df_ISSUE.columns:
        df_ISSUE[i] = df_ISSUE[i].map({'Y':1, 'N':0})
    
    '''preprocess each column'''
    df_AEFK_addG_concat = pd.concat([df_ISSUE['IF_ISSUE_A_IND'],df_ISSUE['IF_ISSUE_E_IND'],
                                     df_ISSUE['IF_ISSUE_F_IND'],df_ISSUE['IF_ISSUE_K_IND'],
                                     df_ISSUE['IF_ADD_G_IND']],axis=1)
    df_IF_ISSUE_AEFK_addG_IND = df_AEFK_addG_concat.sum(axis=1)
    df_IF_ISSUE_AEFK_addG_IND = df_IF_ISSUE_AEFK_addG_IND**(1/max(df_IF_ISSUE_AEFK_addG_IND))
    df_NP_addF_concat = pd.concat([df_ISSUE['IF_ISSUE_N_IND'],df_ISSUE['IF_ISSUE_P_IND'],
                                   df_ISSUE['IF_ADD_F_IND']],axis=1)
    df_IF_ISSUE_NP_addF_IND = df_NP_addF_concat.sum(axis=1)
    df_IF_ISSUE_NP_addF_IND = df_IF_ISSUE_NP_addF_IND**(1/max(df_IF_ISSUE_NP_addF_IND))
    df_CD_concat = pd.concat([df_ISSUE['IF_ISSUE_C_IND'],df_ISSUE['IF_ISSUE_D_IND']],axis=1)
    df_IF_ISSUE_CD_IND = df_CD_concat.sum(axis=1)
    df_IF_ISSUE_CD_IND = df_IF_ISSUE_CD_IND**(1/max(df_IF_ISSUE_CD_IND))
    df_IF_ADD_R_IND = df_ISSUE.iloc[:,-1]
    df_Q_addLQ_concat = pd.concat([df_ISSUE['IF_ISSUE_Q_IND'],
                                    df_ISSUE['IF_ADD_L_IND'],df_ISSUE['IF_ADD_Q_IND']],axis=1)
    df_IF_ISSUE_Q_addLQ_IND = df_Q_addLQ_concat.sum(axis=1)
    df_IF_ISSUE_Q_addLQ_IND = df_IF_ISSUE_Q_addLQ_IND**(1/max(df_IF_ISSUE_Q_addLQ_IND))
    df_BGJ_concat = pd.concat([df_ISSUE['IF_ISSUE_B_IND'],df_ISSUE['IF_ISSUE_G_IND'],
                                    df_ISSUE['IF_ISSUE_J_IND'],],axis=1)
    df_IF_ISSUE_BGJ_IND = df_BGJ_concat.sum(axis=1)
    df_IF_ISSUE_BGJ_IND = df_IF_ISSUE_BGJ_IND**(1/max(df_IF_ISSUE_BGJ_IND))

    '''conbine all preprocessed series'''
    df_ISSUE_ADD_concat = pd.concat([df_IF_ISSUE_AEFK_addG_IND,   #0.028~0.031
                                     df_IF_ISSUE_NP_addF_IND,     #0.033~0.036
                                     df_IF_ISSUE_CD_IND,          #0.04
                                     df_IF_ADD_R_IND,             #0.042
                                     df_IF_ISSUE_Q_addLQ_IND ,    #0.047~0.048
                                     df_IF_ISSUE_BGJ_IND],axis=1) #0.048~0.049
    df_ISSUE_ADD_concat = pd.DataFrame(df_ISSUE_ADD_concat.values, 
                                       columns=['IF_ISSUE_AEFK_addG_IND',
                                                'IF_ISSUE_NP_addF_IND',
                                                'IF_ISSUE_CD_IND',
                                                'IF_ADD_R_IND',
                                                'IF_ISSUE_Q_addLQ_IND',
                                                'IF_ISSUE_BGJ_IND'])
    return df_ISSUE_ADD_concat

'''將IF_ISSUE_INSD_A_IND到IF_ADD_INSD_R_IND，以Y1購買率等等為基準，22欄合成10欄'''

def FT_100to121_ISSUE_and_ADD_INSD_IND(df):
    '''preprocess IF_ISSUE_INSD_AtoQ_IND'''
    df_ISSUE_INSD = df.iloc[:,99:116]
    for i in df_ISSUE_INSD.columns:
        df_ISSUE_INSD[i] = df_ISSUE_INSD[i].map({'Y':1, 'N':0})
    df_IF_ISSUE_INSD_P_IND = df_ISSUE_INSD.iloc[:,-2]
    df_IF_ISSUE_INSD_P_IND.fillna(-1, inplace=True)
    df_IF_ISSUE_INSD_P_IND.replace({0:-0.12}, inplace=True)
    df_IF_ISSUE_INSD_P_IND.value_counts()
    df_IF_ISSUE_INSD_Q_IND = df_ISSUE_INSD.iloc[:,-1]
    df_IF_ISSUE_INSD_Q_IND.fillna(-1, inplace=True)
    df_IF_ISSUE_INSD_Q_IND.replace({0:-0.15}, inplace=True)
    df_IF_ISSUE_INSD_Q_IND.value_counts()
    df_ISSUE_INSD_FMNO_concat = pd.concat([df_ISSUE_INSD['IF_ISSUE_INSD_F_IND'],
                                           df_ISSUE_INSD['IF_ISSUE_INSD_M_IND'],
                                           df_ISSUE_INSD['IF_ISSUE_INSD_N_IND'],
                                           df_ISSUE_INSD['IF_ISSUE_INSD_O_IND']],axis=1)
    df_IF_ISSUE_INSD_FMNO_IND = df_ISSUE_INSD_FMNO_concat.sum(axis=1,skipna = False)
    df_IF_ISSUE_INSD_FMNO_IND = df_IF_ISSUE_INSD_FMNO_IND**(1/max(df_IF_ISSUE_INSD_FMNO_IND))
    df_IF_ISSUE_INSD_FMNO_IND.fillna(-1, inplace=True)
    df_IF_ISSUE_INSD_FMNO_IND.replace({0:0.1}, inplace=True)
    df_IF_ISSUE_INSD_FMNO_IND.value_counts()
    df_ISSUE_INSD_JKL_concat = pd.concat([df_ISSUE_INSD['IF_ISSUE_INSD_J_IND'],
                                           df_ISSUE_INSD['IF_ISSUE_INSD_K_IND'],
                                           df_ISSUE_INSD['IF_ISSUE_INSD_L_IND']],axis=1)
    df_IF_ISSUE_INSD_JKL_IND = df_ISSUE_INSD_JKL_concat.sum(axis=1,skipna = False)
    df_IF_ISSUE_INSD_JKL_IND = df_IF_ISSUE_INSD_JKL_IND**(1/max(df_IF_ISSUE_INSD_JKL_IND))
    df_IF_ISSUE_INSD_JKL_IND.fillna(-1, inplace=True)
    df_IF_ISSUE_INSD_JKL_IND.replace({0:-0.05}, inplace=True)
    df_IF_ISSUE_INSD_JKL_IND.value_counts()
    df_ISSUE_INSD_AD_concat = pd.concat([df_ISSUE_INSD['IF_ISSUE_INSD_A_IND'],
                                         df_ISSUE_INSD['IF_ISSUE_INSD_D_IND']],axis=1)
    df_IF_ISSUE_INSD_AD_IND = df_ISSUE_INSD_AD_concat.sum(axis=1,skipna = False)
    df_IF_ISSUE_INSD_AD_IND = df_IF_ISSUE_INSD_AD_IND**(1/max(df_IF_ISSUE_INSD_AD_IND))
    df_IF_ISSUE_INSD_AD_IND.fillna(-1, inplace=True)
    df_IF_ISSUE_INSD_AD_IND.replace({0:0.1}, inplace=True)
    df_IF_ISSUE_INSD_AD_IND.value_counts()
    df_ISSUE_INSD_BCG_concat = pd.concat([df_ISSUE_INSD['IF_ISSUE_INSD_B_IND'],
                                          df_ISSUE_INSD['IF_ISSUE_INSD_C_IND'],
                                          df_ISSUE_INSD['IF_ISSUE_INSD_G_IND']],axis=1)
    df_IF_ISSUE_INSD_BCG_IND = df_ISSUE_INSD_BCG_concat.sum(axis=1,skipna = False)
    df_IF_ISSUE_INSD_BCG_IND = df_IF_ISSUE_INSD_BCG_IND**(1/max(df_IF_ISSUE_INSD_BCG_IND))
    df_IF_ISSUE_INSD_BCG_IND.fillna(-1, inplace=True)
    df_IF_ISSUE_INSD_BCG_IND.value_counts()
    df_ISSUE_INSD_HI_concat = pd.concat([df_ISSUE_INSD['IF_ISSUE_INSD_H_IND'],
                                         df_ISSUE_INSD['IF_ISSUE_INSD_I_IND']],axis=1)
    df_IF_ISSUE_INSD_HI_IND = df_ISSUE_INSD_HI_concat.sum(axis=1,skipna = False)
    df_IF_ISSUE_INSD_HI_IND = df_IF_ISSUE_INSD_HI_IND**(1/max(df_IF_ISSUE_INSD_HI_IND))
    df_IF_ISSUE_INSD_HI_IND.fillna(-1, inplace=True)
    df_IF_ISSUE_INSD_HI_IND.replace({0:-0.1}, inplace=True)
    df_IF_ISSUE_INSD_HI_IND.value_counts()
    
    '''preprocess IF_ADD_INSD_FLQGR_IND'''
    df_ADD_INSD = df.iloc[:,116:121]
    for j in df_ADD_INSD.columns:
        df_ADD_INSD[j] = df_ADD_INSD[j].map({'Y':1, 'N':0})
    df_IF_ADD_INSD_F_IND = df_ADD_INSD.loc[:,'IF_ADD_INSD_F_IND']
    df_IF_ADD_INSD_F_IND.fillna(-1, inplace=True)
    df_IF_ADD_INSD_F_IND.replace({0:0.4}, inplace=True)
    df_IF_ADD_INSD_F_IND.value_counts()
    df_ADD_INSD_LQG_concat = pd.concat([df_ADD_INSD['IF_ADD_INSD_L_IND'],
                                        df_ADD_INSD['IF_ADD_INSD_Q_IND'],
                                        df_ADD_INSD['IF_ADD_INSD_G_IND']],axis=1)
    df_IF_ADD_INSD_LQG_IND = df_ADD_INSD_LQG_concat.sum(axis=1,skipna = False)
    df_IF_ADD_INSD_LQG_IND.fillna(-1, inplace=True)
    df_IF_ADD_INSD_LQG_IND = df_IF_ADD_INSD_LQG_IND**(1/max(df_IF_ADD_INSD_LQG_IND))
    df_IF_ADD_INSD_LQG_IND.fillna(-1, inplace=True)
    df_IF_ADD_INSD_LQG_IND.replace({0:0.2,1:1.1}, inplace=True)
    df_IF_ADD_INSD_LQG_IND.value_counts()
    df_IF_ADD_INSD_R_IND = df_ADD_INSD.loc[:,'IF_ADD_INSD_R_IND']
    df_IF_ADD_INSD_R_IND.fillna(-1, inplace=True)
    df_IF_ADD_INSD_R_IND.replace({0:-0.15,1:1.3}, inplace=True)
    df_IF_ADD_INSD_R_IND.value_counts()
    
    '''conbine all preprocessed series'''
    df_ISSUE_ADD_INSD_concat = pd.concat([df_IF_ISSUE_INSD_P_IND,   #0.028~0.031
                                          df_IF_ISSUE_INSD_Q_IND,     #0.033~0.036
                                          df_IF_ISSUE_INSD_FMNO_IND,          #0.04
                                          df_IF_ISSUE_INSD_JKL_IND,             #0.042
                                          df_IF_ISSUE_INSD_AD_IND,    #0.047~0.048
                                          df_IF_ISSUE_INSD_BCG_IND,
                                          df_IF_ISSUE_INSD_HI_IND,
                                          df_IF_ADD_INSD_F_IND,
                                          df_IF_ADD_INSD_LQG_IND,
                                          df_IF_ADD_INSD_R_IND],axis=1) #0.048~0.049
    df_ISSUE_ADD_INSD_concat = pd.DataFrame(df_ISSUE_ADD_INSD_concat.values, 
                                       columns=['IF_ISSUE_INSD_P_IND',
                                                'IF_ISSUE_INSD_Q_IND',
                                                'IF_ISSUE_INSD_FMNO_IND',
                                                'IF_ISSUE_INSD_JKL_IND',
                                                'IF_ISSUE_INSD_AD_IND',
                                                'IF_ISSUE_INSD_BCG_IND',
                                                'IF_ISSUE_INSD_HI_IND',
                                                'IF_ADD_INSD_F_IND',
                                                'IF_ADD_INSD_LQG_IND',
                                                'IF_ADD_INSD_R_IND'])
    return df_ISSUE_ADD_INSD_concat
   
def FT_122_IF_ADD_INSD_IND(df):
    df_ADD_INSD = df['IF_ADD_INSD_IND']
    df_ADD_INSD = df_ADD_INSD.map({'Y':1, 'N':0})
    df_ADD_INSD.fillna(0.6, inplace=True)
    return df_ADD_INSD

def FT_123_L1YR_GROSS_PRE_AMT(df):
    fn='L1YR_GROSS_PRE_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.5
    df_log_minmax_shift.replace({-np.inf:0}, inplace=True)
    return df_log_minmax_shift

def FT_124_CUST_9_SEGMENTS_CD(df):
    fn='CUST_9_SEGMENTS_CD'
    df_CUST9 = df[fn]
    df_CUST9.replace({'A':'CUST9_AFG','F':'CUST9_AFG','G':'CUST9_AFG',
                      'B':'CUST9_BDE','D':'CUST9_BDE','E':'CUST9_BDE',
                      'C':'CUST9_C',
                      'H':'CUST9_H'}, inplace=True)
    df_CUST9_onehot = pd.get_dummies(df_CUST9)
    return df_CUST9_onehot

'''將FINANCETOOLS_A 到 G，以Y1購買率等等為基準，7欄合成4欄'''

def FT_125to131_FIN_TOOL(df):
    '''preprocess FINANCETOOLS_AtoG''' # All NA_rate:0.0147
    df_FIN = df.iloc[:,124:131]
    for i in df_FIN.columns:
        df_FIN[i] = df_FIN[i].map({'Y':1, 'N':0})
    df_FIN_A = df_FIN.iloc[:,0]
    df_FIN_A.fillna(-1, inplace=True)
    df_FIN_A.replace({0:0.4}, inplace=True)
    df_FIN_A.value_counts()
    df_FIN_C = df_FIN.iloc[:,2]
    df_FIN_C.fillna(-1, inplace=True)
    df_FIN_C.replace({0:0.6}, inplace=True)
    df_FIN_C.value_counts()
    df_FIN_BDF_concat = pd.concat([df_FIN['FINANCETOOLS_B'],
                                   df_FIN['FINANCETOOLS_D'],
                                   df_FIN['FINANCETOOLS_F']],axis=1)
    df_FIN_BDF = df_FIN_BDF_concat.sum(axis=1, skipna=False)
    df_FIN_BDF.fillna(-1, inplace=True)
    df_FIN_BDF = df_FIN_BDF**(1/max(df_FIN_BDF))
    df_FIN_BDF.fillna(-1, inplace=True)
    df_FIN_BDF.replace({0:0.75}, inplace=True)
    df_FIN_BDF.value_counts()
    df_FIN_EG_concat = pd.concat([df_FIN['FINANCETOOLS_E'],df_FIN['FINANCETOOLS_G']],axis=1)
    df_FIN_EG = df_FIN_EG_concat.sum(axis=1, skipna=False)
    df_FIN_EG.fillna(-1, inplace=True)
    df_FIN_EG = df_FIN_EG**(1/max(df_FIN_EG))
    df_FIN_EG.fillna(-1, inplace=True)
    df_FIN_EG.replace({0:0.7,1:0.9}, inplace=True)
    df_FIN_EG.value_counts()

    '''conbine all preprocessed series'''
    df_FIN_TOOL_concat = pd.concat([df_FIN_A, df_FIN_C, df_FIN_BDF, df_FIN_EG],axis=1) #0.048~0.049
    df_FIN_TOOL_concat = pd.DataFrame(df_FIN_TOOL_concat.values, 
                                       columns=['FINANCETOOLS_A', 'FINANCETOOLS_C',
                                                'FINANCETOOLS_BDF', 'FINANCETOOLS_EG'])
    return df_FIN_TOOL_concat

def FT_83_DIEBENEFIT_AMT(df):
    fn = 'DIEBENEFIT_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.5
    df_log_minmax_shift.replace({-np.inf:-1}, inplace=True)
    df_log_minmax_shift.fillna(-0.3, inplace=True)
    return df_log_minmax_shift

def FT_84_DIEACCIDENT_AMT(df):
    fn = 'DIEACCIDENT_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.5
    df_log_minmax_shift.replace({-np.inf:-1}, inplace=True)
    df_log_minmax_shift.fillna(-0.3, inplace=True)
    return df_log_minmax_shift

def FT_85_POLICY_VALUE_AMT(df):
    fn = 'POLICY_VALUE_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.5
    df_log_minmax_shift.replace({-np.inf:-0.1}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_86_ANNUITY_AMT(df):
    fn = 'ANNUITY_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.3
    df_log_minmax_shift.replace({-np.inf:0}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_87_EXPIRATION_AMT(df):
    fn = 'EXPIRATION_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 0.75  #'''改加0.75'''
    df_log_minmax_shift.replace({-np.inf:-0.9}, inplace=True)   #'''改成-0.9'''
    df_log_minmax_shift.fillna(1, inplace=True)   #'''改成 1'''
    return df_log_minmax_shift

def FT_88_ACCIDENT_HOSPITAL_REC_AMT(df):
    fn = 'ACCIDENT_HOSPITAL_REC_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.4
    df_log_minmax_shift.replace({-np.inf:0.1}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_89_DISEASES_HOSPITAL_REC_AMT(df):
    fn = 'DISEASES_HOSPITAL_REC_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.4
    df_log_minmax_shift.replace({-np.inf:0.1}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_90_OUTPATIENT_SURGERY_AMT(df):
    fn = 'OUTPATIENT_SURGERY_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.4
    df_log_minmax_shift.replace({-np.inf:0.23}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_91_INPATIENT_SURGERY_AMT(df):
    fn = 'INPATIENT_SURGERY_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.4
    df_log_minmax_shift.replace({-np.inf:0.23}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_92_PAY_LIMIT_MED_MISC_AMT(df):
    fn = 'PAY_LIMIT_MED_MISC_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.4
    df_log_minmax_shift.replace({-np.inf:0.23}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_93_FIRST_CANCER_AMT(df):
    fn = 'FIRST_CANCER_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 0.9 #'''改加0.75'''
    df_log_minmax_shift.replace({-np.inf:1}, inplace=True)  #'''改加 1'''
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_94_ILL_ACCELERATION_AMT(df):
    fn = 'ILL_ACCELERATION_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.4
    df_log_minmax_shift.replace({-np.inf:0.2}, inplace=True)  
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_95_ILL_ADDITIONAL_AMT(df):
    fn = 'ILL_ADDITIONAL_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.4
    df_log_minmax_shift.replace({-np.inf:0.2}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_96_LONG_TERM_CARE_AMT(df):
    fn = 'LONG_TERM_CARE_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.6
    df_log_minmax_shift.replace({-np.inf:-0.2}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_97_MONTHLY_CARE_AMT(df):
    fn = 'MONTHLY_CARE_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.3
    df_log_minmax_shift.replace({-np.inf:0.15}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_18_APC_1ST_YEARDIF(df):
    fn = 'APC_1ST_YEARDIF'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.2
    df_log_minmax_shift.replace({-np.inf:0}, inplace=True)
    df_log_minmax_shift.fillna(-1, inplace=True)
    return df_log_minmax_shift

def FT_50_ANNUAL_PREMIUM_AMT(df):
    fn = 'ANNUAL_PREMIUM_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.5
    df_log_minmax_shift.replace({-np.inf:-0.3}, inplace=True)
    df_log_minmax_shift.fillna(-0.7, inplace=True)
    return df_log_minmax_shift

def FT_54_ANNUAL_INCOME_AMT(df):
    fn = 'ANNUAL_INCOME_AMT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.2
    df_log_minmax_shift.replace({-np.inf:-0.1}, inplace=True)
    df_log_minmax_shift.fillna(-0.8, inplace=True)
    return df_log_minmax_shift

def FT_62_L1YR_C_CNT(df):
    fn = 'L1YR_C_CNT'
    df_conti = df[fn]
    df_conti.loc[df_conti[(14 <= df_conti)].index] = 14
    df_not0 = df_conti.loc[df_conti > 0]
    df_conti_norm = df_conti/(max(df_not0)-min(df_not0))*1.2
    df_conti_norm.fillna(-1, inplace=True)
    return df_conti_norm

def FT_63_BANK_NUMBER_CNT(df):
    fn = 'BANK_NUMBER_CNT'
    df_conti = df[fn]
    df_conti.loc[df_conti[(0.625 < df_conti)].index] = 0.625
    df_conti_shift = df_conti + 0.8
    df_conti_shift.replace({0.8:0}, inplace=True)
    return df_conti_shift

def FT_65_BMI(df):
    fn = 'BMI'
    df_target = df[fn]
    df_not025 = df_target.loc[df_target > 0.275]#.index
    df_target.fillna(df_not025.mean() - 0.025, inplace=True)
    return df_target

def FT_73_TERMINATION_RATE(df):
    fn = 'TERMINATION_RATE'
    df_target = df[fn]
    df_target_scale = df_target/100
    df_target_scale.replace({1:0.99}, inplace=True)
    df_not0 = df_target_scale.loc[df_target_scale > 0]
    df_not0.replace({1:0.99}, inplace=True)
    df_log = np.log(df_target_scale)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.01
    df_log_minmax_shift.replace({-np.inf:0}, inplace=True)
    df_log_minmax_shift.fillna(1.2, inplace=True)
    return df_log_minmax_shift

def FT_82_TOOL_VISIT_1YEAR_CNT(df):
    fn = 'TOOL_VISIT_1YEAR_CNT'
    df_target = df[fn]
    df_target_scale = df_target/200
    df_not0 = df_target_scale.loc[df_target_scale > 0]
    df_log = np.log(df_target_scale)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.2
    df_log_minmax_shift.replace({-np.inf:-0.8}, inplace=True)
    return df_log_minmax_shift

def FT_98_IF_HOUSEHOLD_CLAIM_IND(df):
    fn = 'IF_HOUSEHOLD_CLAIM_IND'
    df_target = df[fn].map({'Y':1, 'N':0})
    return df_target

def FT_99_LIFE_INSD_CNT(df):
    fn = 'LIFE_INSD_CNT'
    df_conti = df[fn]
    df_not0 = df_conti.loc[df_conti > 0]
    df_log = np.log(df_conti)
    df_log_minmax_shift = df_log/(np.log(max(df_not0))-np.log(min(df_not0))) + 1.2
    df_log_minmax_shift.replace({-np.inf:0}, inplace=True)
    return df_log_minmax_shift


def concatSeries(sr):
  df_result = pd.concat(sr, axis=1)
  return df_result

#多個類別轉0,1
def mulClass_to_01(df_ts):
  df_ts = df_ts.transform(lambda x: 0 if x == 0 else 1)
  return df_ts

#Y,N轉0,1
def YN_to_01(df_ts):
  df_ts = df_ts.map({'Y':1, 'N':0})
  return df_ts

#正常01 series -> df
sr_YN_to_01 = []
fn_YN = ['IF_ADD_IND','L1YR_PAYMENT_REMINDER_IND','LAST_B_CONTACT_DT','LAST_C_DT','IF_S_REAL_IND','IM_IS_B_IND','IM_IS_C_IND','IM_IS_D_IND','X_A_IND','X_B_IND','X_C_IND','X_D_IND','X_E_IND','X_F_IND','X_G_IND','X_H_IND']
for fn in fn_YN:
  df_cp[fn] = YN_to_01(df_cp[fn])
  sr_YN_to_01.append(df_cp[fn])

#多類別變0,1 -> df
sr_mulClass_to_01 = []
fn_mulClass = ['AG_CNT','AG_NOW_CNT','CLC_CUR_NUM']
for fn in fn_mulClass:
  df_cp[fn] = mulClass_to_01(df_cp[fn])
  sr_mulClass_to_01.append(df_cp[fn])

#特殊處理(電子報)
def IND(df_ts):
  df_ts = df_ts.map({'Y':1, 'N':0.8})
  df_ts.fillna(0, inplace = True)
  return df_ts
sr_IND = []
fn_IND = ['A_IND','B_IND','C_IND']
for fn in fn_IND:
  df_cp[fn] = IND(df_cp[fn])
  sr_IND.append(df_cp[fn])
print(df_cp['A_IND'].value_counts())
print(df_cp['A_IND'].isna().sum())
# sns.catplot(x="A_IND", y="Y1", kind="bar", data=df, ci=None)

#處理NA
def NA_to_mean(df_ts):
  df_ts.fillna(df_ts.mean(skipna = True), inplace = True)
  return df_ts
fn_na_to_mean = ['X_A_IND','X_B_IND','X_C_IND','X_D_IND','X_E_IND','X_F_IND','X_G_IND','X_H_IND']
for fn in fn_na_to_mean:
  df_cp[fn] = NA_to_mean(df_cp[fn])
print(df_cp['X_B_IND'].value_counts())
print(df_cp['X_B_IND'].isna().sum())
# sns.catplot(x="X_B_IND", y="Y1", kind="bar", data=df, ci=None)

def IM_CNT(df_ts):
  df_ts['IM_CNT'] = df_ts['IM_CNT'].transform(lambda x: 0 if x == 0 else(0.5 if x == 1 else(0.7 if x == 2 else(0.9 if x == 3 else 1.1))))
  return df_ts
df_cp = IM_CNT(df_cp)

#Series to df
sr = sr_YN_to_01 + sr_mulClass_to_01 + sr_IND
sr.append(df_cp['IM_CNT'])

#Gender 
c2=np.zeros(data.shape[0])
c2[data[:,1]=='F']=1
c2[data[:,1]=='NA']=0.5

#Age
c3=np.zeros(data.shape[0])
c3[data[:,2]=='低']=0
c3[data[:,2]=='中高']=0.8
c3[data[:,2]=='中']=1
c3[data[:,2]=='高']=0.4

c4=np.zeros(data.shape[0])
c4[data[:,3]=='B1']=0
c4[data[:,3]=='A1']=0.4
c4[data[:,3]=='A2']=0.55
c4[data[:,3]=='B2']=0.4
c4[data[:,3]=='D']=0.5
c4[data[:,3]=='C1']=0.45
c4[data[:,3]=='C2']=0.4
c4[data[:,3]=='E']=1

#Education CD
c6=np.zeros(data.shape[0])
c6[data[:,5]=='NA']=0
c6[data[:,5]=='1']=0.2
c6[data[:,5]=='2']=0.5
c6[data[:,5]=='3']=0.6
c6[data[:,5]=='4']=1

#Marriage CD
c7=np.zeros(data.shape[0])
c7[data[:,6]=='NA']=-1
c7[data[:,6]=='0']=0.3
c7[data[:,6]=='1']=0.7
c7[data[:,6]=='2']=1

#Last A Contact Dt
c8=np.zeros(data.shape[0])
c8[data[:,7]=='Y']=1

# L1YR_A_ISSUE_CNT
c9=np.ones(data.shape[0])
c9[data[:,8]=='0']=0

# LAST_A_ISSUE_DT
c10=np.zeros(data.shape[0])
c10[data[:,9]=='Y']=1

# CHANNEL_A_POL_CNT
c13=np.ones(data.shape[0])
c13[data[:,12]=='0']=0

# OCCUPATION_CLASS_CD
c15=np.zeros(data.shape[0])
c15[data[:,14]=='1']=0.4
c15[data[:,14]=='2']=0.7
c15[data[:,14]=='3']=0.7
c15[data[:,14]=='4']=0.5
c15[data[:,14]=='5']=0.7
c15[data[:,14]=='6']=0.9
c15[data[:,14]=='NA']=1.2

# APC_CNT
c16=np.zeros(data.shape[0])
c16[data[:,15]=='0']=1
c16[data[:,15]=='1']=0.7
c16[data[:,15]=='2']=0.7
c16[data[:,15]=='3']=0.1

# INSD_CNT
c17=np.zeros(data.shape[0])
c17[data[:,16]=='0']=1
c17[data[:,16]=='1']=0.85
c17[data[:,16]=='2']=0.7
c17[data[:,16]=='3']=0.1
c17[data[:,16]=='4']=1
c17[data[:,16]=='5']=0.7
c17[data[:,16]=='6']=1

# APC_1ST_AGE
c18=np.zeros(data.shape[0])
c18[data[:,21]=='低']=1
c18[data[:,21]=='中']=1
c18[data[:,21]=='中高']=0.8
c18[data[:,21]=='高']=0.5
c18[data[:,21]=='NA']=-1

# INSD_1ST_AGE
c19=np.zeros(data.shape[0])
c19[data[:,21]=='低']=0.1
c19[data[:,21]=='中']=1
c19[data[:,21]=='中高']=1
c19[data[:,21]=='高']=0.6
c19[data[:,21]=='NA']=c19[data[:,21]!='NA'].mean()

# IF_2ND_GEN_IND
c20=np.zeros(data.shape[0])
c20[data[:,19]=='0']=0.7
c20[data[:,19]=='1']=0.5

# RFM_R
c22=np.zeros(data.shape[0])
c22[data[:,21]=='低']=1
c22[data[:,21]=='中']=0.2
c22[data[:,21]=='中高']=0.5
c22[data[:,21]=='高']=0.3
c22[data[:,21]=='NA']=-1

# REBUY_TIMES_CNT
c23=np.zeros(data.shape[0])
c23[data[:,22]=='低']=0.4
c23[data[:,22]=='中']=0.5
c23[data[:,22]=='中高']=0.6
c23[data[:,22]=='高']=0.8
c23[data[:,22]=='NA']=-1

# LEVEL
c24=np.zeros(data.shape[0])
c24[data[:,23]=='1']=0.1
c24[data[:,23]=='2']=0.3
c24[data[:,23]=='3']=0.4
c24[data[:,23]=='4']=0.6
c24[data[:,23]=='5']=1
c24[data[:,23]=='NA']=-1

# RFM_M_LEVEL
c25=np.zeros(data.shape[0])
c25[data[:,24]=='3']=0.3
c25[data[:,24]=='5']=0.4
c25[data[:,24]=='7']=0.5
c25[data[:,24]=='8']=0.6
c25[data[:,24]=='9']=0.6
c25[data[:,24]=='10']=1
c25[data[:,24]=='NA']=-1

# LIFE_CNT
c26=np.zeros(data.shape[0])
c26[data[:,25]=='低']=0.4
c26[data[:,25]=='中']=0.1
c26[data[:,25]=='高']=1

header=np.array([['GENDER','AGE','CHARGE_CITY_CD','EDUCATION_CD','MARRIAGE_CD','LAST_A_CCONTACT_DT','L1YR_A_ISSUE_CNT','LAST_A_ISSUE_DT','CHANNEL_A_POL_CNT','OCCUPATION_CLASS_CD','APC_CNT','INSD_CNT','APC_1ST_AGE','INSD_1ST_AGE','IF_2ND_GEN_IND','RFM_R','REBUY_TIMES_CNT','LEVEL','RFM_M_LEVEL','LIFE_CNT']])
res=np.concatenate(([c2],[c3],[c4],[c6],[c7],[c8],[c9],[c10],[c13],[c15],[c16],[c17],[c18],[c19],[c20],[c22],[c23],[c24],[c25],[c26])).T
# deron=np.concatenate((header,res))
# deron=pd.DataFrame(data=res,index=[i for i in range(150000)],columns=['GENDER','AGE','CHARGE_CITY_CD','EDUCATION_CD','MARRIAGE_CD','LAST_A_CCONTACT_DT','L1YR_A_ISSUE_CNT','LAST_A_ISSUE_DT','CHANNEL_A_POL_CNT','OCCUPATION_CLASS_CD','APC_CNT','INSD_CNT','APC_1ST_AGE','INSD_1ST_AGE','IF_2ND_GEN_IND','RFM_R','REBUY_TIMES_CNT','LEVEL','RFM_M_LEVEL','LIFE_CNT'])
deron=pd.DataFrame(data=res,index=[i for i in range(100000)],columns=['GENDER','AGE','CHARGE_CITY_CD','EDUCATION_CD','MARRIAGE_CD','LAST_A_CCONTACT_DT','L1YR_A_ISSUE_CNT','LAST_A_ISSUE_DT','CHANNEL_A_POL_CNT','OCCUPATION_CLASS_CD','APC_CNT','INSD_CNT','APC_1ST_AGE','INSD_1ST_AGE','IF_2ND_GEN_IND','RFM_R','REBUY_TIMES_CNT','LEVEL','RFM_M_LEVEL','LIFE_CNT'])
allen=pd.concat([FT_18_APC_1ST_YEARDIF(df),FT_27to48_ISSUE_and_ADD_IND(df)
                ,FT_50_ANNUAL_PREMIUM_AMT(df),FT_54_ANNUAL_INCOME_AMT(df)
                ,FT_62_L1YR_C_CNT(df),FT_63_BANK_NUMBER_CNT(df)
                ,FT_65_BMI(df),FT_73_TERMINATION_RATE(df),FT_82_TOOL_VISIT_1YEAR_CNT(df)
                ,FT_83_DIEBENEFIT_AMT(df),FT_84_DIEACCIDENT_AMT(df),FT_85_POLICY_VALUE_AMT(df)
                ,FT_86_ANNUITY_AMT(df),FT_87_EXPIRATION_AMT(df),FT_88_ACCIDENT_HOSPITAL_REC_AMT(df)
                ,FT_89_DISEASES_HOSPITAL_REC_AMT(df),FT_90_OUTPATIENT_SURGERY_AMT(df)
                ,FT_91_INPATIENT_SURGERY_AMT(df),FT_92_PAY_LIMIT_MED_MISC_AMT(df)
                ,FT_93_FIRST_CANCER_AMT(df),FT_94_ILL_ACCELERATION_AMT(df)
                ,FT_95_ILL_ADDITIONAL_AMT(df),FT_96_LONG_TERM_CARE_AMT(df)
                ,FT_97_MONTHLY_CARE_AMT(df),FT_98_IF_HOUSEHOLD_CLAIM_IND(df)
                ,FT_99_LIFE_INSD_CNT(df),FT_100to121_ISSUE_and_ADD_INSD_IND(df)
                ,FT_122_IF_ADD_INSD_IND(df),FT_123_L1YR_GROSS_PRE_AMT(df)
                ,FT_124_CUST_9_SEGMENTS_CD(df),FT_125to131_FIN_TOOL(df)],axis=1)

mau = concatSeries(sr)
final = pd.concat([df["CUS_ID"].to_frame(),deron, mau,allen,df["Y1"].to_frame()], axis=1)
# final = pd.concat([df["CUS_ID"].to_frame(),deron,mau,allen], axis=1)
final.to_csv('data/trainpre.csv', index=False)
# final.to_csv('data/testpre.csv', index=False)

##############################################################################
########################### plot.py ########################################## 

import pandas as pd
import numpy as np
from string import ascii_uppercase
import matplotlib.pyplot as plt

def Counting(i,target):
    Tnum=data[ data[:,i] ==target].shape[0]
    Tindex=index[data[:,i] ==target]
    Tbuy=np.intersect1d(Y,Tindex).shape[0]
    Tbuyrate=round(Tbuy*100/Tnum,3)
    print('{} = {}，有買 = {}，{} %'.format(target,Tnum,Tbuy,Tbuyrate))
    return Tbuyrate

def Plot(name,i,row):
    print('{} [{}]'.format(name,i+1))
    column=[]
    for r in row:
        column.append(Counting(i,r))
    plt.figure()
    plt.bar(row, column)
    plt.savefig('image/{}.jpg'.format(i+1))

def Plot2(name,i):
    print('{} [{}]'.format(name,i+1))
    num0=data[ data[:,i] =='0'].shape[0]
    index0=index[data[:,i] =='0']
    buy0=np.intersect1d(Y,index0).shape[0]
    buyrate0=round(buy0*100/num0,3)
    print('0 = {}，有買 = {}，{} %'.format(num0,buy0,buyrate0))
    numother=data[ data[:,i] !='0'].shape[0]
    indexother=index[data[:,i] !='0']
    buyother=np.intersect1d(Y,indexother).shape[0]
    buyrateother=round(buyother*100/numother,3)
    print('其他 = {}，有買 = {}，{} %'.format(numother,buyother,buyrateother))
    row=['0','其他']
    column=[buyrate0,buyrateother]
    plt.figure()
    plt.bar(row, column)
    plt.savefig('image/{}.jpg'.format(i+1))

data=np.loadtxt('data/train.csv',dtype=np.str,delimiter=',',encoding='big5')[1:]
index=np.array([i for i in range(100000)])
Y=index[data[:,131]=='Y']
N=index[data[:,131]=='N']
ch=np.array([c for c in ascii_uppercase])

Plot('Gender',1,['F','M','NA'])
Plot('AGE',2,['低','中','中高','高'])
Plot('CHARGE_CITY_CD',3,['A1','A2','B1','B2','C1','C2','D','E'])
Plot('CONTACT_CITY_CD',4,['A1','A2','B1','B2','C1','C2','D','E'])
Plot('EDUCATION_CD',5,['1','2','3','4','NA'])
Plot('MARRIAGE_CD',6,['0','1','2','NA'])
Plot('LAST_A_CCONTACT_DT',7,['Y','N'])
Plot2('L1YR_A_ISSUE_CNT',8)
# Plot('L1YR_A_ISSUE_CNT',8,['0','1','2','3','4','5','6','7','8','9','10','12','13','14','15','16','22'])
Plot('LAST_A_ISSUE_DT',9,['Y','N'])
Plot('L1YR_B_ISSUE_CNT',10,['0','1','2','3','4'])
Plot('LAST_B_ISSUE_DT',11,['Y','N'])
Plot2('CHANNEL_A_POL_CNT',12)
# Plot('CHANNEL_A_POL_CNT',12,['0','1','2','3','4','NA'])
Plot2('CHANNEL_B_POL_CNT',13)
Plot('OCCUPATION_CLASS_CD',14,['0','1','2','3','4','5','6','NA'])
Plot('APC_CNT',15,['0','1','2','3','4'])
Plot('INSD_CNT',16,['0','1','2','3','4','5','6','9','12','19'])
Plot('APC_1ST_AGE',17,['低','中','中高','高','NA'])
Plot('INSD_1ST_AGE',18,['低','中','中高','高','NA'])
Plot('IF_2ND_GEN_IND',19,['Y','N'])
# Plot('APC_1ST_YEARDIF',20,['0','1','2','NA'])
Plot('RFM_R',21,['低','中','中高','高','NA'])
Plot('REBUY_TIMES_CNT',22,['低','中','中高','高','NA'])
Plot('LEVEL',23,['1','2','3','4','5','NA'])
Plot('RFM_M_LEVEL',24,['3','5','7','7','9','10','NA'])
Plot('LIFE_CNT',25,['低','中','高'])

Plot('IF_ADD_IND',48,['Y','N'])
Plot2('AG_CNT',50)
Plot2('AG_NOW_CNT',51)
Plot2('CLC_CUR_NUM',52)
Plot('L1YR_PAYMENT_REMINDER_IND',54,['Y','N'])
Plot('L1YR_LAPSE_IND',55,['Y','N'])
Plot('LAST_B_CONTACT_DT',56,['Y','N'])
Plot('A_IND',57,['Y','N'])
Plot('B_IND',58,['Y','N'])
Plot('C_IND',59,['Y','N'])
Plot('LAST_C_DT',60,['Y','N'])
Plot('IF_S_REAL_IND',65,['Y','N'])
Plot('IF_Y_REAL_IND',66,['Y','N'])
Plot('IM_CNT',67,['1','2','3','4'])
Plot('IM_IS_A_IND',68,['Y','N'])
Plot('IM_IS_B_IND',69,['Y','N'])
Plot('IM_IS_C_IND',70,['Y','N'])
Plot('IM_IS_D_IND',71,['Y','N'])
Plot('X_A_IND',73,['Y','N','NA'])
Plot('X_B_IND',74,['Y','N','NA'])
Plot('X_C_IND',75,['Y','N','NA'])
Plot('X_D_IND',76,['Y','N','NA'])
Plot('X_E_IND',77,['Y','N','NA'])
Plot('X_F_IND',78,['Y','N','NA'])
Plot('X_G_IND',79,['Y','N','NA'])
Plot('X_H_IND',80,['Y','N','NA'])

##############################################################################
################################# CNN.py #####################################

from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from keras.backend import set_session
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import plot_model
from IPython.display import Image
import os

def validation_DL(_model,x_test, y_test):
    y_predict = _model.predict(x_test)
    print(y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC = ", auc)
    plt.plot(fpr, tpr, 'b', label="AUC="+str(round(auc,3)))
    plt.plot([0, 1],[0, 1], 'r--', label="Baseline")
    plt.legend(loc=4)
    plt.savefig('CNN.jpg')

def result_DL(file_path,model):
    test_df = pd.read_csv(file_path, encoding="big5")
    fr_dummies = pd.get_dummies(test_df.iloc[:, 1:])
    fr_dummies.fillna(0, inplace=True)
    x_real = fr_dummies.values
    print(x_real.shape)
    y_predict = model.predict(x_real)[:,0]
    output = np.array(list(zip(test_df.iloc[:,0].values.astype(str), y_predict)))
    # np.savetxt()output
    odf = pd.DataFrame(output, columns=["CUS_ID","Ypred"])
    odf.to_csv("CNN.csv", index=False)

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.Session(config=config) 
set_session(sess)

model = Sequential()  
# model.add(Conv2D(32, (3, 3),padding='same', input_shape = (10, 10, 1), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
# model.add(Conv2D(32, (3, 3),padding='same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# # # Third convolutional layer
model.add(Conv2D(64, (3, 3),padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# model.add(Dense(output_dim = 128, activation = 'relu'))
# model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 256, activation = 'relu'))
model.add(Dense(output_dim = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpointer = ModelCheckpoint('filt_weights_best.h5', save_best_only=True, save_weights_only=False, 
                                   verbose=1, monitor='val_loss', mode='min')
earlystop = EarlyStopping(patience=30, monitor='val_loss')
train = np.loadtxt('data/trainpre.csv',dtype=np.str,delimiter=',')[1:,1:].astype(np.float)
train_set = np.concatenate(( [[0,0,0] for i in range(train.shape[0])] , train[:,:train.shape[1]-1] , [[0,0,0] for i in range(train.shape[0])] ),axis=1).reshape((100000,10,10))
label = train[:,train.shape[1]-1]
x_train, x_test, y_train, y_test = train_test_split(train_set, label, test_size = 0.2, random_state = 0)
model.fit(np.expand_dims(x_train,axis=3) , y_train , batch_size=50, epochs = 30,validation_data=(np.expand_dims(x_test,axis=3),y_test) ,  callbacks=[checkpointer, earlystop])
# model.fit(x_train, y_train , batch_size=50, epochs = 30,validation_data=(x_test,y_test) ,  callbacks=[checkpointer, earlystop])

# test = np.loadtxt('data/testpre.csv',dtype=np.str,delimiter=',')[1:,1:].astype(np.float)
# test_set = np.concatenate(( [[0,0,0] for i in range(test.shape[0])] , test , [[0,0,0] for i in range(test.shape[0])] ),axis=1).reshape((150000,10,10))
# model.predict(np.expand_dims(test_set,axis=3))
# plot_model(model, to_file="model.png", show_shapes=True)
# Image('model.png')
m=load_model('filt_weights_best.h5')
validation_DL(m,np.expand_dims(x_test,axis=3), np.expand_dims(y_test,axis=3))


##################################################################################
####################### XGBoost & MLP ############################################
### Run on Colab ###


from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from xgboost import plot_importance
from keras.utils import plot_model
from IPython.display import Image
import os
from xgboost import plot_tree

# Read file from google drive
df = pd.read_csv(r"drive/My Drive/Colab Notebooks/trainpre.csv", encoding="big5")
print(df.shape)

"""pd.set_option('display.max_columns', None) 
df1 = df[li][:]
print(df1.shape)
print("Y:" + str(len(df1[df1['Y1'] == "Y"])) + " N:" + str(len(df1[df1['Y1'] == "N"])), end="\n\n")
print(df1.dtypes)
print(df1.head(5))
"""

def data_preprocess(df, flag):
    pd.set_option('display.max_columns', None)
    df_cp = df.copy()
#     df_N = df_cp[df_cp["Y1"]=="N"]
#     df_Y = df_cp[df_cp["Y1"]=="Y"]
    df_N = df_cp[df_cp["Y1"]==0]
    df_Y = df_cp[df_cp["Y1"]==1]
    if flag:
      frames = [df_N,df_Y]
    else:
      frames = [df_N,df_Y]
    df_cp = pd.concat(frames)
    df_data = df_cp.iloc[:, 1:len(df_cp.columns.values)-1]
    df_data.fillna(0, inplace=True)
    x = df_data.values
    y = df_cp.iloc[:, -1].values
    print(x.shape, y.shape)
    return x, y

from sklearn import metrics
def validation(xg_cf, x_test, y_test):
    y_predict = xg_cf.predict_proba(x_test)[:, 1]
    print(y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC = ", auc)
    plt.plot(fpr, tpr, 'b', label="AUC="+str(round(auc,3)))
    plt.plot([0, 1],[0, 1], 'r--', label="Baseline")
    plt.legend(loc=4)
    plt.savefig(r"drive/My Drive/Colab Notebooks/XGB.png")
    plt.show()

def result_XGB(file_path, xg_cf):
    #test_df = pd.read_csv(file_path, encoding="big5")
    test_df = df_test
    fr_dummies = pd.get_dummies(test_df.iloc[:, 1:])
    fr_dummies.fillna(0, inplace=True)
    x_real = fr_dummies.values
    print(x_real.shape)
    y_predict = xg_cf.predict_proba(x_real)[:, 1]
    fn = "predict"
    output = np.array(list(zip(test_df.iloc[:,0].values.astype(str), y_predict)))
    output_df = pd.DataFrame(output, columns=["CUS_ID","Ypred"])
    output_df.to_csv(r"drive/My Drive/Colab Notebooks/" + fn + ".csv", index=False)

def get_xgb_imp(xgb, feat_names):
    from numpy import array
    imp_vals = xgb.booster().get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = array(imp_dict.values()).sum()
    return {k:v/total for k,v in imp_dict.items()}

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
if __name__ == '__main__':
#     ada = ADASYN(random_state=42)
#     ros = RandomOverSampler(random_state=0)
    x, y = data_preprocess(df,0)
#     #x, y = ros.fit_resample(x, y)
#     #print(sorted(Counter(y_resampled).items()))
#     x, y = ada.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)
    print(x_train.shape)
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=df.columns.values[1:-1])
    xg_cf = XGBClassifier(probability=True)
    xg_cf = XGBClassifier(
         probability=True,
         learning_rate =0.1,
         n_estimators=200,
         max_depth=3,
         min_child_weight=1,
         gamma=0,
         subsample=0.8,
         colsample_bytree=0.6,
         reg_alpha=0.5,
         eval_metric = 'auc',
         scale_pos_weight=1)
    xg_cf.fit(x_train, y_train)
    validation(xg_cf, x_test, y_test)
    result_XGB(r"drive/My Drive/Colab Notebooks/testpre.csv", xg_cf)

fig, ax = plt.subplots(figsize=(20, 20))
xgb.plot_importance(model_xg, ax=ax)
plt.savefig(r"drive/My Drive/Colab Notebooks/feature.png")
plt.show()

# def ceate_feature_map(features):
#     outfile = open('xgb.fmap', 'w')
#     i = 0
#     for feat in features:
#         outfile.write('{0}\t{1}\tq\n'.format(i, feat))
#         i = i + 1
#     outfile.close()
# ceate_feature_map(df_cp.columns.values[1:-1])
# fig,ax = plt.subplots()
# fig.set_size_inches(60,30)
# xgb.plot_tree(xg_cf,ax = ax,fmap='xgb.fmap', num_trees=0,rankdir='LR')
# plt.savefig(r"drive/My Drive/Colab Notebooks/XGB_tree.png")

######### MLP #############

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

model = Sequential()

model.add(Dense(units=256, 
                input_dim=94, 
                kernel_initializer='normal', 
                activation='relu'))

# model.add(Dense(units=128, 
#                 kernel_initializer='normal', 
#                 activation='relu'))

model.add(Dense(units=1, 
                kernel_initializer='normal', 
                activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint('drive/My Drive/Colab Notebooks/filt_weights_best.h5'.format(0), save_best_only=True, save_weights_only=False, 
                                   verbose=1, monitor='val_loss', mode='min')
earlystop = EarlyStopping(patience=30, monitor='val_loss')

train_history =model.fit(x=x_train,
                         y=y_train,validation_data=(x_test, y_test), 
                         epochs=300, batch_size=2000,verbose=2, callbacks=[checkpointer, earlystop])

model_mlp = load_model(r"drive/My Drive/Colab Notebooks/filt_weights_best.h5")
Y_pred = model_mlp.predict(x_test)
# plt.scatter(x_train, y_train)
# plt.plot(x_train, Y_pred)
# plt.show()

def result_DL(file_path):
    test_df = pd.read_csv(file_path, encoding="big5")
    fr_dummies = pd.get_dummies(test_df.iloc[:, 1:])
    fr_dummies.fillna(0, inplace=True)
    x_real = fr_dummies.values
    print(x_real.shape)
    
    y_predict = model_mlp.predict(x_real)[:,0]
    fn = "MLP_test"

    output = np.array(list(zip(test_df.iloc[:,0].values.astype(str), y_predict)))
    # np.savetxt()output

    odf = pd.DataFrame(output, columns=["CUS_ID","Ypred"])
    odf.to_csv(r"drive/My Drive/Colab Notebooks/" + fn + ".csv", index=False)

result_DL(r"drive/My Drive/Colab Notebooks/testpre.csv")

from sklearn import metrics

def validation_DL(x_test, y_test):
    y_predict = model_mlp.predict(x_test)
    print(y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC = ", auc)
    plt.plot(fpr, tpr, 'b', label="AUC="+str(round(auc,3)))
    plt.plot([0, 1],[0, 1], 'r--', label="Baseline")
    plt.legend(loc=4)
    plt.savefig(r"drive/My Drive/Colab Notebooks/MLP.png")
    plt.show()
validation_DL(x_test, y_test)

def show_train_history(train_history, train, validation):
  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train History')
  plt.ylabel(train)
  plt.xlabel('Epoch')
  plt.legend(['train','validation'], loc='upper left')
  plt.show()
show_train_history(train_history, 'acc', 'val_acc')

show_train_history(train_history, 'loss', 'val_loss')

plot_model(model_mlp, to_file=r"drive/My Drive/Colab Notebooks/model.png", show_shapes=True)
Image("drive/My Drive/Colab Notebooks/model.png")

############################################################################################
########################## Other ML model ##################################################
### Run on Colab ###

from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from xgboost import plot_importance

test_oversmp = True

''' select the first 2000 of pred-prob from test data '''
if test_oversmp:
#    df_test = pd.read_csv('testpre.csv', encoding="big5")
    
    df = pd.read_csv(r"drive/My Drive/Colab Notebooks/trainpre.csv", encoding="big5")
    df_test = pd.read_csv(r"drive/My Drive/Colab Notebooks/testpre.csv", encoding="big5")
    df_xgb = pd.read_csv(r"drive/My Drive/Colab Notebooks/XGB_dcs_0926.csv")
    
    df_xgb_sort = df_xgb.sort_values(by='Ypred',ascending=False)
    df_xgb_first2000 = df_xgb_sort.iloc[0:600,:]
    
    df_aug = pd.merge(df_test,df_xgb_first2000, on=['CUS_ID'], how='right')
    df_aug = df_aug.drop(columns=['Ypred'])
    df_ones = pd.DataFrame(np.ones([len(df_xgb_first2000),1]), columns = ['Y1'])
 #   df_ones.replace({1:'Y'}, inplace=True)
    df_aug = pd.concat([df_aug,df_ones],axis=1)
    df = pd.concat([df,df_aug])

# Read file from google drive
df = pd.read_csv(r"drive/My Drive/Colab Notebooks/trainpre.csv", encoding="big5")
print(df.shape)

"""pd.set_option('display.max_columns', None) 
df1 = df[li][:]
print(df1.shape)
print("Y:" + str(len(df1[df1['Y1'] == "Y"])) + " N:" + str(len(df1[df1['Y1'] == "N"])), end="\n\n")
print(df1.dtypes)
print(df1.head(5))
"""

def data_preprocess(df, flag):
    pd.set_option('display.max_columns', None)
    df_cp = df.copy()
#     df_N = df_cp[df_cp["Y1"]=="N"]
#     df_Y = df_cp[df_cp["Y1"]=="Y"]
    df_N = df_cp[df_cp["Y1"]==0]
    df_Y = df_cp[df_cp["Y1"]==1]
    if flag:
      frames = [df_N,df_Y]
    else:
      frames = [df_N,df_Y]
    df_cp = pd.concat(frames)
    df_data = df_cp.iloc[:, 1:len(df_cp.columns.values)-1]
    df_data.fillna(0, inplace=True)
    x = df_data.values
    y = df_cp.iloc[:, -1].values
    print(x.shape, y.shape)
    return x, y

from sklearn import metrics
def validation(xg_cf, x_test, y_test):
    y_predict = xg_cf.predict_proba(x_test)[:, 1]
    print(y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC = ", auc)
    plt.plot(fpr, tpr, 'b', label="AUC="+str(round(auc,3)))
    plt.plot([0, 1],[0, 1], 'r--', label="Baseline")
    plt.legend(loc=4)
    plt.show()

def result_XGB(file_path, xg_cf):
    #test_df = pd.read_csv(file_path, encoding="big5")
    test_df = df_test
    fr_dummies = pd.get_dummies(test_df.iloc[:, 1:])
    fr_dummies.fillna(0, inplace=True)
    x_real = fr_dummies.values
    print(x_real.shape)
    y_predict = xg_cf.predict_proba(x_real)[:, 1]
    fn = "predict"
    output = np.array(list(zip(test_df.iloc[:,0].values.astype(str), y_predict)))
    output_df = pd.DataFrame(output, columns=["CUS_ID","Ypred"])
    output_df.to_csv(r"drive/My Drive/Colab Notebooks/" + fn + ".csv", index=False)

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

if __name__ == '__main__':
    x, y = data_preprocess(df,0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)
    pickle.dump(x, open('drive/My Drive/Colab Notebooks/x'+ '_0926.p', 'wb'))
    pickle.dump(y, open('drive/My Drive/Colab Notebooks/y'+ '_0926.p', 'wb'))
    print(x_train.shape)
#     xg_cf = XGBClassifier(probability=True)
#     xg_cf = XGBClassifier(
#          probability=True,
#          learning_rate =0.1,
#          n_estimators=200,
#          max_depth=4,
#          min_child_weight=1,
#          gamma=0,
#          subsample=0.8,
#          colsample_bytree=0.8,
#          eval_metric = 'error',
#          scale_pos_weight=1)
#     xg_cf.fit(x_train, y_train)
#     validation(xg_cf, x_test, y_test)
#     result_XGB(r"drive/My Drive/Colab Notebooks/testpre.csv", xg_cf)

# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'gini', random_state = 0 )
classifier.fit(x_train, y_train)

# Predicting the Test set results

#y_pred = classifier.predict_proba(y_test)
def validation_RF(x_test, y_test):
    y_predict = classifier.predict_proba(x_test)[:, 1]
    print(y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC = ", auc)
    plt.plot(fpr, tpr, 'b', label="AUC="+str(round(auc,3)))
    plt.plot([0, 1],[0, 1], 'r--', label="Baseline")
    plt.legend(loc=4)
    plt.savefig(r"drive/My Drive/Colab Notebooks/RandomForest.png")
    plt.show()
validation_RF(x_test, y_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
def validation_NB(x_test, y_test):
    y_predict = classifier.predict_proba(x_test)[:, 1]
    print(y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC = ", auc)
    plt.plot(fpr, tpr, 'b', label="AUC="+str(round(auc,3)))
    plt.plot([0, 1],[0, 1], 'r--', label="Baseline")
    plt.legend(loc=4)
    plt.savefig(r"drive/My Drive/Colab Notebooks/GaussianNB.png")
    plt.show()
validation_NB(x_test, y_test)

# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(x_train, y_train)


# Predicting the Test set results
def validation_DCT(x_test, y_test):
    y_predict = classifier.predict_proba(x_test)[:, 1]
    print(y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC = ", auc)
    plt.plot(fpr, tpr, 'b', label="AUC="+str(round(auc,3)))
    plt.plot([0, 1],[0, 1], 'r--', label="Baseline")
    plt.legend(loc=4)
    plt.savefig(r"drive/My Drive/Colab Notebooks/DecisionTree.png")
    plt.show()
validation_DCT(x_test, y_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
def validation_LR(x_test, y_test):
    y_predict = classifier.predict_proba(x_test)[:, 1]
    print(y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC = ", auc)
    plt.plot(fpr, tpr, 'b', label="AUC="+str(round(auc,3)))
    plt.plot([0, 1],[0, 1], 'r--', label="Baseline")
    plt.legend(loc=4)
    plt.savefig(r"drive/My Drive/Colab Notebooks/LogisticRegression.png")
    plt.show()
validation_LR(x_test, y_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, probability = True)
classifier.fit(x_train, y_train)

# Predicting the Test set results
def validation_rbfSVC(x_test, y_test):
    y_predict = classifier.predict_proba(x_test)[:, 1]
    print(y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC = ", auc)
    plt.plot(fpr, tpr, 'b', label="AUC="+str(round(auc,3)))
    plt.plot([0, 1],[0, 1], 'r--', label="Baseline")
    plt.legend(loc=4)
    plt.savefig(r"drive/My Drive/Colab Notebooks/rbfSVC.png")
    plt.show()
validation_rbfSVC(x_test, y_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, probability = True)
classifier.fit(x_train, y_train)

# Predicting the Test set results
def validation_SVC(x_test, y_test):
    y_predict = classifier.predict_proba(x_test)[:, 1]
    print(y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC = ", auc)
    plt.plot(fpr, tpr, 'b', label="AUC="+str(round(auc,3)))
    plt.plot([0, 1],[0, 1], 'r--', label="Baseline")
    plt.legend(loc=4)
    plt.savefig(r"drive/My Drive/Colab Notebooks/SVC.png")
    plt.show()
validation_SVC(x_test, y_test)
