"""
Name     : stcs.py
Author   : William Smyth/Yinsen Miao
Contact : drwss.academy@gmail.com/yinsenm@gmail.com
Time     : 25/12/2022
Desc     : covariance/co-movement stastistic functions
"""

import math
import random
import numpy as np
import pandas as pd
from nearest_correlation import nearcorr
from numpy import diag, inf, copy, dot
from numpy.linalg import norm, eigvals
from math import sqrt

def is_psd_def(cov_mat):
    """
    :param cov_mat: covariance matrix of p x p
    :return: true if positive semi definite (PSD)
    """
    return np.all(eigvals(cov_mat) > -1e-6)


# SQUEEZING STATISTIC IQ
def IQ(df_rets: pd.DataFrame, eta, gamma, dCplus, dCminus, dDplus, dDminus, center_method="zero",
       scale_method="none"):
    '''A 16-channel version. Can be setup to take up to 10 etas.
    Currently set up for basic-IQ [i.e., eta, eta^2, 1]. Whilst
    the function takes four delta parameters as input, dC/D +/-,
    they are constrained by dCplus=dDplus and dCminus = dDminus.
    Also, aa & cc replace dCplus & dCminus a la corrected 'ifs'.
    '''
    threshold = 0.005  # nominal 2x100*threshold %-sigma exclusion zone centred on mean
    epsilon = 0  # pre-decay delay within truncated window [0 means not implemented]

    rets = df_rets.values
    symbols = df_rets.columns
    n, p = rets.shape
    sd_vec = rets.std(axis=0)

    # center asset return using mean or median
    if center_method == "mean":
        center = np.mean(rets, axis=0)
    elif center_method == "median":
        center = np.median(rets, axis=0)
    elif center_method == "zero":
        center = np.zeros(p)

    SD = np.diag(sd_vec)
    std_rets = rets - center

    cor_mat = np.eye(p)
    for i in range(p):
        for j in range(i + 1):
            # initialise squeezing channel accumulators
            ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15, ch16 = [0] * 16

            # align eta values for basic-IQ
            eta1, eta2, eta3 = [eta] * 3;  # body
            eta7, eta8, eta9, eta10 = [eta ** 2] * 4;  # wing
            eta4, eta5, eta6 = [1] * 3  # tail

            # check eta conditions are met [formally correct correlation matrix]
            if not (eta3 <= sqrt(eta1 * eta2) and
                    eta6 <= sqrt(eta4 * eta5) and
                    eta7 <= sqrt(eta1 * eta4) and
                    eta8 <= sqrt(eta2 * eta5) and
                    eta9 <= sqrt(eta2 * eta4) and
                    eta10 <= sqrt(eta1 * eta5)):
                raise ValueError("at least one of the eta conditions violated")

            # check delta conditions are met [correct alignment of channels across quadrants]
            if not (dCplus == dDplus and
                    dCminus == dDminus):
                raise ValueError("at least one of the delta conditions violated")

            if scale_method == "vols_min":     # scale by the min of pair vols
                width_factor_aa = width_factor_cc = min(sd_vec[i], sd_vec[j])
            elif scale_method == "vols_max":   # scale by the max of pair vols
                width_factor_aa = width_factor_cc = max(sd_vec[i], sd_vec[j])
            elif scale_method == "vols_avg":  # scale by the average of pair vols
                width_factor_aa = width_factor_cc = 0.5 * (sd_vec[i] + sd_vec[j])
            elif scale_method == "vols":       # scale by the individual vol
                width_factor_aa = sd_vec[j]
                width_factor_cc = sd_vec[i]
            else:  # no vol scaling
                width_factor_aa = width_factor_cc = 1  # no vol scaling

            aa = dCplus * width_factor_aa
            cc = dCminus * width_factor_cc

            for k in range(n):

                # setting up temporal discounting [tau not incorporated]
                age_decay = np.exp(-gamma * max(0, n - (epsilon + 1) - k))

                # determine each squeezing channel individually.

                ## row 1 [channels 1-4]
                # channel 1
                if (std_rets[k, i] < -cc) and (std_rets[k, j] >= aa):
                    ch1 += age_decay
                # channel 2
                if (std_rets[k, i] >= -cc) and (std_rets[k, i] < -threshold) and (std_rets[k, j] >= aa):
                    ch2 += age_decay
                # channel 3
                if (std_rets[k, i] >= threshold) and (std_rets[k, i] < aa) and (std_rets[k, j] >= aa):
                    ch3 += age_decay
                # channel 4
                if (std_rets[k, i] >= aa) and (std_rets[k, j] >= aa):
                    ch4 += age_decay

                ## row 2 [channels 5-8]
                # channel 5
                if (std_rets[k, i] < -cc) and (std_rets[k, j] >= threshold) and (std_rets[k, j] < aa):
                    ch5 += age_decay
                # channel 6
                if (std_rets[k, i] >= -cc) and (std_rets[k, i] < -threshold) and (std_rets[k, j] >= threshold) and (
                        std_rets[k, j] < aa):
                    ch6 += age_decay
                # channel 7
                if (std_rets[k, i] >= threshold) and (std_rets[k, i] < aa) and (std_rets[k, j] >= threshold) and (
                        std_rets[k, j] < aa):
                    ch7 += age_decay
                # channel 8
                if (std_rets[k, i] >= aa) and (std_rets[k, j] >= threshold) and (std_rets[k, j] < aa):
                    ch8 += age_decay

                ## row 3 [channels 9-12]
                # channel 9
                if (std_rets[k, i] < -cc) and (std_rets[k, j] >= -cc) and (std_rets[k, j] < -threshold):
                    ch9 += age_decay
                # channel 10
                if (std_rets[k, i] >= -cc) and (std_rets[k, i] < -threshold) and (std_rets[k, j] >= -cc) and (
                        std_rets[k, j] < -threshold):
                    ch10 += age_decay
                # channel 11
                if (std_rets[k, i] >= threshold) and (std_rets[k, i] < aa) and (std_rets[k, j] >= -cc) and (
                        std_rets[k, j] < -threshold):
                    ch11 += age_decay
                # channel 12
                if (std_rets[k, i] >= aa) and (std_rets[k, j] >= -cc) and (std_rets[k, j] < -threshold):
                    ch12 += age_decay

                ## row 4 [channels 13-16]
                # channel 13
                if (std_rets[k, i] < -cc) and (std_rets[k, j] < -cc):
                    ch13 += age_decay
                # channel 14
                if (std_rets[k, i] >= -cc) and (std_rets[k, i] < -threshold) and (std_rets[k, j] < -cc):
                    ch14 += age_decay
                # channel 15
                if (std_rets[k, i] >= threshold) and (std_rets[k, i] < aa) and (std_rets[k, j] < -cc):
                    ch15 += age_decay
                # channel 16
                if (std_rets[k, i] >= aa) and (std_rets[k, j] < -cc):
                    ch16 += age_decay

                    # compute squeezing correlation matrix
            num_ij = eta1 * ch7 + eta2 * ch10 - eta3 * (ch6 + ch11) + eta4 * ch4 + eta5 * ch13 \
                     - eta6 * (ch1 + ch16) + eta7 * (ch3 + ch8) + eta8 * (ch9 + ch14) - eta9 * (ch2 + ch12) - eta10 * (
                                 ch5 + ch15)

            den_ii = eta5 * (ch1 + ch5 + ch9 + ch13) + eta2 * (ch2 + ch6 + ch10 + ch14) \
                     + eta1 * (ch3 + ch7 + ch11 + ch15) + eta4 * (ch4 + ch8 + ch12 + ch16)

            den_jj = eta5 * (ch13 + ch14 + ch15 + ch16) + eta2 * (ch9 + ch10 + ch11 + ch12) \
                     + eta1 * (ch5 + ch6 + ch7 + ch8) + eta4 * (ch1 + ch2 + ch3 + ch4)

            if (den_ii != 0) & (den_jj != 0):
                cor_mat[i, j] = num_ij / np.sqrt(den_ii * den_jj)
            cor_mat[j, i] = cor_mat[i, j]

    # # pass matrix to Higham if not PSD
    # if not is_psd_def(cor_mat):
    #     cor_mat = nearcorr(cor_mat)  # use Higham's alternating projections algorithm
    cov_mat = SD @ cor_mat @ SD
    return pd.DataFrame(cov_mat, index=symbols, columns=symbols)

## the fully general version below is not used in the Gerber dataset analysis
def IQfull(rets:np.array,delta_list,eta_list,gamma,full = False,reduced = False,basic = True): 
    '''
    IQfull has up to 29 parameters: 4 from the delta_list, 21 from
    eta_list, 1 threshold, 3 temporal parameters(tau,epsilon,gamma)
    Full-IQ is a 36-channel 21-eta framework. delta_list = [a,b,c,d] 
    hardwires 36 squeezing channels with a,b,c,d, free to vary subject 
    to a < b, c < d. eta_list= [eta01,eta02,eta04,eta05,eta11,eta12] 
    contains the six diagonal etas [free to vary independently]. All 
    other etas are derived from these via the sufficiency conditions.
    This functions contains Boolean flags ['full','reduced','basic'] 
    to provide variant versatility in a single function. Basic-IQ and 
    reduced-IQ are derived from full-IQ by letting bb and dd tend to 
    infinity[1000] to ensure no data populate outer layer of channels. 
    Also, the eta_list is reduced in dimensionality from 21 components 
    to ten: eta 1,2,3 [body]; eta 4,5,6 [tail]; eta 7,8,9 & 10 [wings].  
    '''
    
    aa,bb,cc,dd = delta_list
    if not (bb > aa and dd > cc):
        raise ValueError("a,b and c,d conditions are not met")
    
    threshold = 0.005 # nominal 2x100*threshold %-sigma exclusion zone centred on mean
    epsilon   = 0     # pre-decay delay within truncated window [0 means not implemented]
    
    n, p = rets.shape
    mean_vec = rets.mean(axis=0)
    sd_vec = rets.std(axis=0)
    SD = np.diag(sd_vec)
    std_rets= (rets-mean_vec)/sd_vec 
      
    cor_mat = np.eye(p)  
    for i in range(p):
        for j in range(i + 1): 
            
            # initialise squeezing channel accumulators
            ch01,ch02,ch03,ch04,ch05,ch06,ch07,ch08,ch09,ch10,ch11,ch12,\
            ch13,ch14,ch15,ch16,ch17,ch18,ch19,ch20,ch21,ch22,ch23,ch24,\
            ch25,ch26,ch27,ch28,ch29,ch30,ch31,ch32,ch33,ch34,ch35,ch36 = [0]*36
            
            if full == True:
                # align off-diagonal etas with sufficiency conditions
                eta01,eta02,eta04,eta05,eta11,eta12 = eta_list # diagonal etas
                
                '''All six elements in eta_list and all four in delta_list are 
                free to vary subject to b > a and d > c. If setting eta03 to 
                eta21 to draw equivalance with basic-IQ note that {eta, eta^2 
                and 1} is consistent with the sufficiency conditions but is
                not equivalent to setting each eta value to its upper limit.
                '''
                # for equivalence with basic=True
                ###eta03 = eta; eta06 = 1; eta07 = eta*eta; eta08 = eta*eta            
                ###eta09 = eta*eta; eta10 = eta*eta; eta13 = 1; eta14 = 1                  
                ###eta15 = eta*eta; eta16 = eta*eta; eta17 = 1; eta18 = eta*eta             
                ###eta19 = 1; eta20 = 1; eta21 = eta*eta  
                
                # in general
                eta03 = random.uniform(0,1)*np.sqrt(eta01*eta02)
                eta06 = random.uniform(0,1)*np.sqrt(eta04*eta05)          
                eta07 = random.uniform(0,1)*np.sqrt(eta01*eta04)             
                eta08 = random.uniform(0,1)*np.sqrt(eta02*eta05)            
                eta09 = random.uniform(0,1)*np.sqrt(eta02*eta04)           
                eta10 = random.uniform(0,1)*np.sqrt(eta01*eta05)            
                eta13 = random.uniform(0,1)*np.sqrt(eta11*eta12)            
                eta14 = random.uniform(0,1)*np.sqrt(eta04*eta11)            
                eta15 = random.uniform(0,1)*np.sqrt(eta01*eta11)            
                eta16 = random.uniform(0,1)*np.sqrt(eta02*eta12)            
                eta17 = random.uniform(0,1)*np.sqrt(eta05*eta12)            
                eta18 = random.uniform(0,1)*np.sqrt(eta02*eta11)            
                eta19 = random.uniform(0,1)*np.sqrt(eta05*eta11)             
                eta20 = random.uniform(0,1)*np.sqrt(eta04*eta12)            
                eta21 = random.uniform(0,1)*np.sqrt(eta01*eta12)
            
            if reduced == True:
                
                '''as for IQ_basic below, infinite bb and dd does away with 
                outer layer of channels [reduction from 36 channels to 16]. What
                separates reduced from basic is eta01,02,04,05 are free to vary.
                Last two elements in eta_list are redundant.
                '''
                bb = 1000; dd = 1000
                eta01,eta02,eta04,eta05,eta11,eta12 = eta_list # diagonal etas
                
                # for equivalence with basic=True
                ###eta03 = eta; eta06 = 1; eta07 = eta*eta               
                ###eta08 = eta*eta; eta09 = eta*eta; eta10 = eta*eta  
                
                # in general
                eta03 = random.uniform(0,1)*np.sqrt(eta01*eta02)
                eta06 = random.uniform(0,1)*np.sqrt(eta04*eta05)          
                eta07 = random.uniform(0,1)*np.sqrt(eta01*eta04)             
                eta08 = random.uniform(0,1)*np.sqrt(eta02*eta05)            
                eta09 = random.uniform(0,1)*np.sqrt(eta02*eta04)           
                eta10 = random.uniform(0,1)*np.sqrt(eta01*eta05)
                
                # always eta11-eta21 are redundant since bb and dd are 'infinite'
                eta11 = random.uniform(0,1); eta12 = random.uniform(0,1)           
                eta13 = random.uniform(0,1); eta14 = random.uniform(0,1)           
                eta15 = random.uniform(0,1); eta16 = random.uniform(0,1)          
                eta17 = random.uniform(0,1); eta18 = random.uniform(0,1)            
                eta19 = random.uniform(0,1); eta20 = random.uniform(0,1)            
                eta21 = random.uniform(0,1)
                
            if basic == True:
                
                '''infinite bb and dd does away with outer layer 
                of channels [reduction from 36 channels to 16]. Only 
                eta_list[0] = eta is used. Other elements redundant. 
                '''
                bb = 1000; dd = 1000
                eta01,eta02,eta04,eta05,eta11,eta12 = eta_list # diagonal etas
                
                # body
                eta01 = eta_list[0]
                eta02 = eta01; eta03 = eta01
                # tails
                eta04 = 1; eta05 = 1; eta06 = 1    
                # wings
                eta07 = eta01*eta01; eta08 = eta01*eta01
                eta09 = eta01*eta01; eta10 = eta01*eta01     
                # always eta11-eta21 are redundant since bb and dd are 'infinite'
                eta11 = random.uniform(0,1); eta12 = random.uniform(0,1)           
                eta13 = random.uniform(0,1); eta14 = random.uniform(0,1)           
                eta15 = random.uniform(0,1); eta16 = random.uniform(0,1)          
                eta17 = random.uniform(0,1); eta18 = random.uniform(0,1)            
                eta19 = random.uniform(0,1); eta20 = random.uniform(0,1)            
                eta21 = random.uniform(0,1)
                
                
            for k in range(n):  
                
                # setting up temporal discounting [tau not incorporated]
                age_decay = np.exp(-gamma*max(0,n-(epsilon+1)-k))                            
                
                # determine [quadrant I] squeezing channel contributions
                # [channels 4,5,6,10,11,12,16,17,18]
                if (std_rets[k, i] >= threshold) and (std_rets[k, j] >= threshold):
                    
                    if (std_rets[k, i] >= bb):
                    
                        if (std_rets[k, j] >= bb):
                            ch06 += age_decay  
                        
                        if (std_rets[k, j] >= aa) and (std_rets[k, j] < bb):
                            ch12 += age_decay    
                        
                        if (std_rets[k, j] < aa) and (std_rets[k, j] >= threshold): 
                            ch18 += age_decay
                        
                    if (std_rets[k, i] >= aa) and (std_rets[k, i] < bb):
                    
                        if (std_rets[k, j] >= bb):
                            ch05 += age_decay  
                        
                        if (std_rets[k, j] >= aa) and (std_rets[k, j] < bb):
                            ch11 += age_decay    
                        
                        if (std_rets[k, j] < aa) and (std_rets[k, j] >= threshold):  
                            ch17 += age_decay
                        
                    if (std_rets[k, i] < aa) and (std_rets[k, i] >= threshold):
                    
                        if (std_rets[k, j] >= bb):
                            ch04 += age_decay  
                        
                        if (std_rets[k, j] >= aa) and (std_rets[k, j] < bb):
                            ch10 += age_decay    
                        
                        if (std_rets[k, j] < aa) and (std_rets[k, j] >= threshold):  
                            ch16 += age_decay
                        

                # determine [quadrant II] squeezing channel contributions
                # [channels 22,23,24,28,29,30,34,35,36]
                if (std_rets[k, i] >= threshold) and (std_rets[k, j] < -threshold):
                    
                    if (std_rets[k, i] >= bb):
                    
                        if (std_rets[k, j] >= -cc) and (std_rets[k, j] < -threshold):
                            ch24 += age_decay  
                        
                        if (std_rets[k, j] >= -dd) and (std_rets[k, j] < -cc):
                            ch30 += age_decay    
                        
                        if (std_rets[k, j] < -dd): 
                            ch36 += age_decay
                        
                    if (std_rets[k, i] >= aa) and (std_rets[k, i] < bb):
                    
                        if (std_rets[k, j] >= -cc) and (std_rets[k, j] < -threshold):
                            ch23 += age_decay  
                        
                        if (std_rets[k, j] >= -dd) and (std_rets[k, j] < -cc):
                            ch29 += age_decay    
                        
                        if (std_rets[k, j] < -dd): 
                            ch35 += age_decay
                        
                    if (std_rets[k, i] < aa) and (std_rets[k, i] >= threshold):
                    
                        if (std_rets[k, j] >= -cc) and (std_rets[k, j] < -threshold):
                            ch22 += age_decay  
                        
                        if (std_rets[k, j] >= -dd) and (std_rets[k, j] < -cc):
                            ch28 += age_decay    
                        
                        if (std_rets[k, j] < -dd): 
                            ch34 += age_decay
                        
                        
                # determine [quadrant III] squeezing channel contributions
                # [channels 19,20,21,25,26,27,31,32,32]
                if (std_rets[k, i] < -threshold) and (std_rets[k, j] < -threshold):
                    
                    if (std_rets[k, i] >= -cc) and (std_rets[k, i] < -threshold):
                    
                        if (std_rets[k, j] >= -cc) and (std_rets[k, j] < -threshold):
                            ch21 += age_decay  
                        
                        if (std_rets[k, j] >= -dd) and (std_rets[k, j] < -cc):
                            ch27 += age_decay    
                        
                        if (std_rets[k, j] < -dd): 
                            ch33 += age_decay
                        
                    if (std_rets[k, i] >= -dd) and (std_rets[k, i] < -cc):
                    
                        if (std_rets[k, j] >= -cc) and (std_rets[k, j] < -threshold):
                            ch20 += age_decay  
                        
                        if (std_rets[k, j] >= -dd) and (std_rets[k, j] < -cc):
                            ch26 += age_decay    
                        
                        if (std_rets[k, j] < -dd): 
                            ch32 += age_decay
                        
                    if (std_rets[k, i] < -dd):
                    
                        if (std_rets[k, j] >= -cc) and (std_rets[k, j] < -threshold):
                            ch19 += age_decay  
                        
                        if (std_rets[k, j] >= -dd) and (std_rets[k, j] < -cc):
                            ch25 += age_decay    
                        
                        if (std_rets[k, j] < -dd): 
                            ch31 += age_decay
                        
                        
                # determine [quadrant IV] squeezing channel contributions
                # [channels 1,2,3,7,8,9,13,14,15]
                if (std_rets[k, i] < -threshold) and (std_rets[k, j] >= threshold):
                    
                    if (std_rets[k, i] >= -cc) and (std_rets[k, i] < -threshold):
                    
                        if (std_rets[k, j] >= bb):
                            ch03 += age_decay  
                        
                        if (std_rets[k, j] >= aa) and (std_rets[k, j] < bb):
                            ch09 += age_decay    
                        
                        if (std_rets[k, j] < aa) and (std_rets[k, j] >= threshold): 
                            ch15 += age_decay
                        
                    if (std_rets[k, i] >= -dd) and (std_rets[k, i] < -cc):
                    
                        if (std_rets[k, j] >= bb):
                            ch02 += age_decay  
                        
                        if (std_rets[k, j] >= aa) and (std_rets[k, j] < bb):
                            ch08 += age_decay    
                        
                        if (std_rets[k, j] < aa) and (std_rets[k, j] >= threshold): 
                            ch14 += age_decay
                        
                    if (std_rets[k, i] < -dd):
                    
                        if (std_rets[k, j] >= bb):
                            ch01 += age_decay  
                        
                        if (std_rets[k, j] >= aa) and (std_rets[k, j] < bb):
                            ch07 += age_decay    
                        
                        if (std_rets[k, j] < aa) and (std_rets[k, j] >= threshold):  
                            ch13 += age_decay            

            # compute squeezing correlation matrix
            num_ij = eta01*ch16 + eta02*ch21 - eta03*(ch15+ch22) + eta04*ch11 + eta05*ch26\
                    - eta06*(ch08+ch29) + eta07*(ch10+ch17) + eta08*(ch20+ch27) - eta09*(ch09+ch23)\
                    - eta10*(ch14+ch28) + eta11*ch06 + eta12*ch31 - eta13*(ch01+ch36) + eta14*(ch05+ch12)\
                    + eta15*(ch04+ch18) + eta16*(ch19+ch33) + eta17*(ch25+ch32) - eta18*(ch03+ch24)\
                    - eta19*(ch02+ch30) - eta20*(ch07+ch35) - eta21*(ch13+ch34)
    
            den_ii = eta12*(ch01+ch07+ch13+ch19+ch25+ch31) + eta05*(ch02+ch08+ch14+ch20+ch26+ch32)\
                   + eta02*(ch03+ch09+ch15+ch21+ch27+ch33) + eta01*(ch04+ch10+ch16+ch22+ch28+ch34)\
                   + eta04*(ch05+ch11+ch17+ch23+ch29+ch35) + eta11*(ch06+ch12+ch18+ch24+ch30+ch36)

            den_jj = eta12*(ch31+ch32+ch33+ch34+ch35+ch36) + eta05*(ch25+ch26+ch27+ch28+ch29+ch30)\
                   + eta02*(ch19+ch20+ch21+ch22+ch23+ch24) + eta01*(ch13+ch14+ch15+ch16+ch17+ch18)\
                   + eta04*(ch07+ch08+ch09+ch10+ch11+ch12) + eta11*(ch01+ch02+ch03+ch04+ch05+ch06)
            
            if (den_ii != 0) & (den_jj != 0):
                cor_mat[i, j] = num_ij/np.sqrt(den_ii*den_jj)
            cor_mat[j,i] = cor_mat[i,j]
    
    # pass matrix to Higham if not PSD
    if not is_psd_def(cor_mat):
        cor_mat = nearcorr(cor_mat) # use Higham's alternating projections algorithm
    cov_mat = SD@cor_mat@SD
    return cov_mat, cor_mat
    
####### GERBER
def gerber_cov_stat1(rets: np.array):
    """
    compute Gerber covariance Statistics 1
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    threshold = 0.5 # Gerber noise exclusion zone, threshold parameter
 
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    SD = np.diag(sd_vec)
    cor_mat = np.zeros((p, p))  # store correlation matrix

    for i in range(p):
        for j in range(i + 1):
            neg = 0
            pos = 0
            nn = 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                elif abs(rets[k, i]) < threshold * sd_vec[i] and abs(rets[k, j]) < threshold * sd_vec[j]:
                    nn += 1

            # compute Gerber correlation matrix
            cor_mat[i, j] = (pos - neg) / (n - nn)
            cor_mat[j, i] = cor_mat[i, j]
            
    cor_mat = nearcorr(cor_mat) # use Higham's alternating projections algorithm
    cov_mat = SD@cor_mat@SD
    return cov_mat, cor_mat
  
    
# LINEAR SHRINKAGE 1
def cov1Para(Y,k = None):
    
    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned
    
    import numpy as np
    import pandas as pd
    import math

    # de-mean returns if required
    N,p = Y.shape                      # sample size and matrix dimension
   
   
    #default setting
    if k is None or math.isnan(k):
        
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                    # adjust effective sample size
    
    
    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    
    
    # compute shrinkage target
    diag = np.diag(sample.to_numpy())
    meanvar= sum(diag)/len(diag)
    target=meanvar*np.eye(p)
    
    
    
    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    
    
    pihat = sum(piMat.sum())
    

    
    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    
    # diagonal part of the parameter that we call rho 
    rho_diag=0;
    
    # off-diagonal part of the parameter that we call rho 
    rho_off=0;
    
    # compute shrinkage intensity
    rhohat=rho_diag+rho_off
    kappahat=(pihat-rhohat)/gammahat
    shrinkage=max(0,min(1,kappahat/n))
    
    # compute shrinkage estimator
    sigmahat=shrinkage*target+(1-shrinkage)*sample
    
    return sigmahat


# LINEAR SHRINKAGE 2
def cov2Para(Y,k = None):
    
    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned
    
    import numpy as np
    import pandas as pd
    import math

    # de-mean returns if required
    N,p = Y.shape                      # sample size and matrix dimension
   
   
    #default setting
    if k is None or math.isnan(k):
        
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                    # adjust effective sample size
    
    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    
    
    #compute shrinkage target
    diag = np.diag(sample.to_numpy())
    meanvar= sum(diag)/len(diag)
    meancov = (np.sum(sample.to_numpy()) - np.sum(np.eye(p)*sample.to_numpy()))/(p*(p-1));
    target = pd.DataFrame(meanvar*np.eye(p)+meancov*(1-np.eye(p)))
    
    #estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat = sum(piMat.sum())
    
    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    # diagonal part of the parameter that we call rho 
    rho_diag = (sample2.sum().sum()-np.trace(sample.to_numpy())**2)/p;
    
    # off-diagonal part of the parameter that we call rho 
    sum1=Y.sum(axis=1)
    sum2=Y2.sum(axis=1)
    temp = (np.multiply(sum1.to_numpy(),sum1.to_numpy())-sum2)
    rho_off1 = np.sum(np.multiply(temp,temp))/(p*n)
    rho_off2 = (sample.sum().sum()-np.trace(sample.to_numpy()))**2/p
    rho_off = (rho_off1-rho_off2)/(p-1)
    
    # compute shrinkage intensity
    rhohat = rho_diag + rho_off
    kappahat = (pihat-rhohat) / gammahat
    shrinkage = max(0 , min(1 , kappahat/n))
    
    # compute shrinkage estimator
    sigmahat=shrinkage*target+(1-shrinkage)*sample
    
    return sigmahat


# LINEAR SHRINKAGE 3
def covCor(Y,k = None):
    
    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned
    
    import numpy as np
    import numpy.matlib as mt
    import pandas as pd
    import math

    # de-mean returns if required
    N,p = Y.shape                      # sample size and matrix dimension
   
   
    #default setting
    if k is None or math.isnan(k):
        
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                    # adjust effective sample size

    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
        
    # compute shrinkage target
    samplevar = np.diag(sample.to_numpy())
    sqrtvar = pd.DataFrame(np.sqrt(samplevar))
    rBar = (np.sum(np.sum(sample.to_numpy()/np.matmul(sqrtvar.to_numpy(),sqrtvar.T.to_numpy())))-p)/(p*(p-1)) # mean correlation
    target = pd.DataFrame(rBar*np.matmul(sqrtvar.to_numpy(),sqrtvar.T.to_numpy()))
    target[np.logical_and(np.eye(p),np.eye(p))] = sample[np.logical_and(np.eye(p),np.eye(p))];
    
    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat = sum(piMat.sum())
    
    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    # diagonal part of the parameter that we call rho 
    rho_diag =  np.sum(np.diag(piMat))
    
    # off-diagonal part of the parameter that we call rho 
    term1 = pd.DataFrame(np.matmul((Y**3).T.to_numpy(),Y.to_numpy())/n)
    term2 = pd.DataFrame(np.transpose(mt.repmat(samplevar,p,1))*sample)
    thetaMat = term1-term2
    thetaMat[np.logical_and(np.eye(p),np.eye(p))] = pd.DataFrame(np.zeros((p,p)))[np.logical_and(np.eye(p),np.eye(p))]
    rho_off = rBar*(np.matmul((1/sqrtvar).to_numpy(),sqrtvar.T.to_numpy())*thetaMat).sum().sum()
    
    # compute shrinkage intensity
    rhohat = rho_diag + rho_off
    kappahat = (pihat - rhohat) / gammahat
    shrinkage = max(0 , min(1 , kappahat/n))
    
    # compute shrinkage estimator
    sigmahat = shrinkage*target + (1-shrinkage) * sample;
    
    return sigmahat


# LINEAR SHRINKAGE 4
def covDiag(Y,k = None):
    
    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned
    
    import numpy as np
    import pandas as pd
    import math

    # de-mean returns if required
    N,p = Y.shape                      # sample size and matrix dimension
   
   
    #default setting
    if k is None or math.isnan(k):
        
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                    # adjust effective sample size

    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
        
    # compute shrinkage target
    target = pd.DataFrame(np.diag(np.diag(sample.to_numpy())))
    
    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat = sum(piMat.sum())
    
    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    # diagonal part of the parameter that we call rho 
    rho_diag =  np.sum(np.diag(piMat))
    
    # off-diagonal part of the parameter that we call rho 
    rho_off = 0
    
    # compute shrinkage intensity
    rhohat = rho_diag + rho_off
    kappahat = (pihat - rhohat) / gammahat
    shrinkage = max(0 , min(1 , kappahat/n))
    
    # compute shrinkage estimator
    sigmahat = shrinkage*target + (1-shrinkage) * sample;
    
    return sigmahat


# LINEAR SHRINKAGE 5
def covMarket(Y,k = None):
    
    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned
    
    import numpy as np
    import numpy.matlib as mt
    import pandas as pd
    import math

    # de-mean returns if required
    N,P = Y.shape                      # sample size and matrix dimension
   
    #default setting
    if k is None or math.isnan(k):
        
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)                               #demean
        k = 1
    
    #vars
    n = N-k                                    # adjust effective sample size
    
    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    #compute shrinkage target
    Ymkt = Y.mean(axis = 1) #equal-weighted market factor
    covmkt = pd.DataFrame(np.matmul(Y.T.to_numpy(),Ymkt.to_numpy()))/n #covariance of original variables with common factor
    varmkt = np.matmul(Ymkt.T.to_numpy(),Ymkt.to_numpy())/n #variance of common factor
    target = pd.DataFrame(np.matmul(covmkt.to_numpy(),covmkt.T.to_numpy()))/varmkt
    target[np.logical_and(np.eye(P),np.eye(P))] = sample[np.logical_and(np.eye(P),np.eye(P))]
    
    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat = sum(piMat.sum())
    
    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    # diagonal part of the parameter that we call rho 
    rho_diag =  np.sum(np.diag(piMat))
    
    # off-diagonal part of the parameter that we call rho 
    temp = Y*pd.DataFrame([Ymkt for i in range(P)]).T
    ################### added below to correct #########
    temp = temp.iloc[:,P:]                     #########        
    ################### added above to correct #########
    covmktSQ = pd.DataFrame([covmkt[0] for i in range(P)])
    v1 = pd.DataFrame((1/n) * np.matmul(Y2.T.to_numpy(),temp.to_numpy())-np.multiply(covmktSQ.T.to_numpy(),sample.to_numpy()))
    roff1 = (np.sum(np.sum(np.multiply(v1.to_numpy(),covmktSQ.to_numpy())))-np.sum(np.diag(np.multiply(v1.to_numpy(),covmkt.to_numpy()))))/varmkt
    v3 = pd.DataFrame((1/n) * np.matmul(temp.T.to_numpy(),temp.to_numpy()) - varmkt * sample)
    roff3 = (np.sum(np.sum(np.multiply(v3.to_numpy(),np.matmul(covmkt.to_numpy(),covmkt.T.to_numpy())))) - np.sum(np.multiply(np.diag(v3.to_numpy()),(covmkt[0]**2).to_numpy()))) /varmkt**2
    rho_off=2*roff1-roff3
    
    # compute shrinkage intensity
    rhohat = rho_diag + rho_off
    kappahat = (pihat - rhohat) / gammahat
    shrinkage = max(0 , min(1 , kappahat/n))
    
    # compute shrinkage estimator
    sigmahat = shrinkage*target + (1-shrinkage) * sample;
    return sigmahat


# NON-LINEAR SHRINKAGE 6
def GIS(Y,k=None):
    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int

    #Set df dimensions
    N = Y.shape[0]                                           
    p = Y.shape[1]                                               

    #default setting
    if (k is None or math.isnan(k)):
        Y = Y.sub(Y.mean(axis=0), axis=1)                              
        k = 1

    #vars
    n = N-k                                      # adjust effective sample size
    c = p/n                                               # concentration ratio

    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    sample = (sample+sample.T)/2                              #make symmetrical

    #Spectral decomp
    lambda1, u = np.linalg.eigh(sample)            #use Cholesky factorisation 
    #                                               based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)              #reset negative values to 0
    dfu = pd.DataFrame(u,columns=lambda1)   #create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1,inplace = True)              #sort df by column index
    lambda1 = dfu.columns                              #recapture sorted lambda

    #COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2,1/c**2)**0.35)/p**0.35                   #smoothing parameter
    invlambda = 1/lambda1[max(1,p-n+1)-1:p]  #inverse of (non-null) eigenvalues
    dfl = pd.DataFrame()
    dfl['lambda'] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values,min(p,n))]          #like  1/lambda_j
    Lj = pd.DataFrame(Lj.to_numpy())                        #Reset column names
    Lj_i = Lj.subtract(Lj.T)                    #like (1/lambda_j)-(1/lambda_i)
   
    theta = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj)*h**2)).mean(axis = 0)          #smoothed Stein shrinker
    Htheta = Lj.multiply(Lj*h).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj)*h**2)).mean(axis = 0)                    #its conjugate
    Atheta2 = theta**2+Htheta**2                         #its squared amplitude
    
    if p<=n:               #case where sample covariance matrix is not singular
        deltahat_1=(1-c)*invlambda+2*c*invlambda*theta #shrunk inverse eigenvalues (LIS)
        
        delta = 1 / ((1-c)**2*invlambda+2*c*(1-c)*invlambda*theta \
                      +c**2*invlambda*Atheta2)    #optimally shrunk eigenvalues
        delta = delta.to_numpy()
    else: # case where sample covariance matrix is singular
        print('p must be <= n for the Symmetrized Kullback-Leibler divergence')       
        return -1
    
    temp = pd.DataFrame(deltahat_1)
    x = min(invlambda)
    temp.loc[temp[0] < x, 0] = x
    deltaLIS_1 = temp[0]

    temp1 = dfu.to_numpy()
    temp2 = np.diag((delta/deltaLIS_1)**0.5)
    temp3 = dfu.T.to_numpy().conjugate()
    # reconstruct covariance matrix
    #cov_mat = pd.DataFrame(np.matmul(np.matmul(temp1,temp2),temp3))
    cov_mat = np.matmul(np.matmul(temp1,temp2),temp3)
    
    return cov_mat


# NON-LINEAR SHRINKAGE 7
def LIS(Y,k=None):
    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned

    #Set df dimensions
    N = Y.shape[0]                                              #num of columns
    p = Y.shape[1]                                                 #num of rows

    #default setting
    if (k is None or math.isnan(k)):
        Y = Y.sub(Y.mean(axis=0), axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                      # adjust effective sample size
    c = p/n                                               # concentration ratio

    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    sample = (sample+sample.T)/2                              #make symmetrical

    #Spectral decomp
    lambda1, u = np.linalg.eigh(sample)            #use Cholesky factorisation 
    #                                               based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)              #reset negative values to 0
    dfu = pd.DataFrame(u,columns=lambda1)   #create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1,inplace = True)              #sort df by column index
    lambda1 = dfu.columns                              #recapture sorted lambda

    #COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2,1/c**2)**0.35)/p**0.35                   #smoothing parameter
    invlambda = 1/lambda1[max(1,p-n+1)-1:p]  #inverse of (non-null) eigenvalues
    dfl = pd.DataFrame()
    dfl['lambda'] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values,min(p,n))]          #like  1/lambda_j
    Lj = pd.DataFrame(Lj.to_numpy())                        #Reset column names
    Lj_i = Lj.subtract(Lj.T)                    #like (1/lambda_j)-(1/lambda_i)
   
    theta = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj)*h**2)).mean(axis = 0)          #smoothed Stein shrinker
    
    if p<=n:               #case where sample covariance matrix is not singular
         deltahat_1=(1-c)*invlambda+2*c*invlambda*theta #shrunk inverse eigenvalues
         
    else: # case where sample covariance matrix is singular
        print("p must be <= n for Stein's loss")       
        return -1
    
    temp = pd.DataFrame(deltahat_1)
    x = min(invlambda)
    temp.loc[temp[0] < x, 0] = x
    deltaLIS_1 = temp[0]


    temp1 = dfu.to_numpy()
    temp2 = np.diag(1/deltaLIS_1)
    temp3 = dfu.T.to_numpy().conjugate()
    # reconstruct covariance matrix
    #sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1,temp2),temp3))
    sigmahat = np.matmul(np.matmul(temp1,temp2),temp3)
    
    return sigmahat


# NON-LINEAR SHRINKAGE 8
def QIS(Y,k=None):
    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned

    #Set df dimensions
    N = Y.shape[0]                                              #num of columns
    p = Y.shape[1]                                                 #num of rows

    #default setting
    if (k is None or math.isnan(k)):
        Y = Y.sub(Y.mean(axis=0), axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                      # adjust effective sample size
    c = p/n                                               # concentration ratio

    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    sample = (sample+sample.T)/2                              #make symmetrical

    #Spectral decomp
    lambda1, u = np.linalg.eigh(sample)            #use Cholesky factorisation 
    #                                               based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)              #reset negative values to 0
    dfu = pd.DataFrame(u,columns=lambda1)   #create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1,inplace = True)              #sort df by column index
    lambda1 = dfu.columns                              #recapture sorted lambda

    #COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2,1/c**2)**0.35)/p**0.35                   #smoothing parameter
    invlambda = 1/lambda1[max(1,p-n+1)-1:p]  #inverse of (non-null) eigenvalues
    dfl = pd.DataFrame()
    dfl['lambda'] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values,min(p,n))]          #like  1/lambda_j
    Lj = pd.DataFrame(Lj.to_numpy())                        #Reset column names
    Lj_i = Lj.subtract(Lj.T)                    #like (1/lambda_j)-(1/lambda_i)
   
    theta = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj)*h**2)).mean(axis = 0)          #smoothed Stein shrinker
    Htheta = Lj.multiply(Lj*h).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj)*h**2)).mean(axis = 0)                    #its conjugate
    Atheta2 = theta**2+Htheta**2                         #its squared amplitude

    if p<=n:               #case where sample covariance matrix is not singular
         delta = 1 / ((1-c)**2*invlambda+2*c*(1-c)*invlambda*theta \
                      +c**2*invlambda*Atheta2)    #optimally shrunk eigenvalues
         delta = delta.to_numpy()
    else:
        delta0 = 1/((c-1)*np.mean(invlambda.to_numpy())) #shrinkage of null 
        #                                                 eigenvalues
        delta = np.repeat(delta0,p-n)
        delta = np.concatenate((delta, 1/(invlambda*Atheta2)), axis=None)

    deltaQIS = delta*(sum(lambda1)/sum(delta))                  #preserve trace
    
    temp1 = dfu.to_numpy()
    temp2 = np.diag(deltaQIS)
    temp3 = dfu.T.to_numpy().conjugate()
    #reconstruct covariance matrix
    #sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1,temp2),temp3))
    sigmahat = np.matmul(np.matmul(temp1,temp2),temp3)
    
    return sigmahat
