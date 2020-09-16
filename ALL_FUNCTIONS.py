# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 17:01:57 2020

@author: maddi
"""
import numpy as np
import math
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import csv
import astropy.units as u

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun, get_moon
from astropy.constants import R_sun, R_jup
import ephem


''' This function filters out all transits with a depth that will be unobservable from the ground
    - ie those with a depth of less than 1 millimag. '''
# CREATES A LIST, DEPTH_FILTER, containing 
def depth_filter(name, pl_radius, st_radius):
    #DEPTH_FILTER = [['index', 'name', 'depth_percentage', 'above_threshold']]
    INDEXES = []
    NAMES = []
    DEPTH = []
    FACTOR = []
    for i in range(len(name)):
        target_name = name[i]
        r_p = pl_radius[i]
        r_s = st_radius[i]
        depth = ((r_p * R_jup)/ (r_s * R_sun))**2
        depth_perc = depth*100
        F_millimag = -2.5*np.log10(1-0.001) #finds the flux depth corresponding to 1millimag
        if depth_perc > F_millimag: #selects targets with a transit depth of more than 0.1%
            INDEXES.append(i)
            NAMES.append(target_name)
            DEPTH.append(float(depth_perc))
            FACTOR.append(float(depth_perc/F_millimag))
        else:
            pass
    print(f'The database contains {len(name)} entries.')
    print(f'There are {len(INDEXES)} targets with a sufficient transit depth for detection')
    return INDEXES, NAMES, DEPTH, FACTOR


#ideally need to sort DEPTH_FILTER by 4th column (index 3)

def transit_predictor(midpoint, err_mid_plus, err_mid_minus, period, err_period_plus, err_period_minus, start, end, convert):
    
    if type(midpoint) == list:
        midpoint = midpoint[0]
        
    #how many transits will happen in this time interval
    how_many = (end - start)/period
    
    begin = (start - midpoint)/period
   
    if begin < 0:
        print('Going back in time??')
    total = begin + how_many
    transits_included = math.floor(total) - math.floor(begin)
    N = math.floor(begin) + np.linspace(1, transits_included, transits_included)
   
    #calculate the transits and add them to a list
    midpoint_pred = []
    errmid_pred_plus = []
    errmid_pred_minus = []

    
    if len(N) > 1:
        for i in N:
            midpoint_pred.append(midpoint + i*period)
            errmid_pred_plus.append(err_mid_plus + i*err_period_plus)
            errmid_pred_minus.append(err_mid_minus + i*err_period_minus)
            
            mid_before = []
            before_plus = []
            before_minus = []
            mid_after = []
            after_plus = []
            after_minus = []
            
    else: 
        if len(N) == 1:
            midpoint_pred = midpoint + N*period
            errmid_pred_plus = err_mid_plus + N*err_period_plus
            errmid_pred_minus = err_mid_minus + N*err_period_minus
            
            mid_before = []
            before_plus = []
            before_minus = []
            mid_after = []
            after_plus = []
            after_minus = []
            
        elif len(N) == 0:
            num_before = math.floor(begin)
            mid_before = midpoint + num_before*period
            before_plus = err_mid_plus + num_before*err_period_plus
            before_minus = err_mid_minus + num_before*err_period_minus
            
            num_after = num_before + 1
            mid_after = midpoint + num_after*period
            after_plus = err_mid_plus + num_after*err_period_plus
            after_minus = err_mid_minus + num_after*err_period_minus
            
    #print the ephemerides in JD format or calendar date
    #for j in range(len(midpoint_pred)):
     #   if convert == True:
      #      print(jd_to_date(midpoint_pred[j]), '+', errmid_pred_plus[j], '-', errmid_pred_minus[j])
       # else:
        #    print(midpoint_pred[j], '+/-', errmid_pred_plus[j], errmid_pred_minus[j])
    
    return midpoint_pred, errmid_pred_plus, errmid_pred_minus, N, mid_before, before_plus, before_minus, mid_after, after_plus, after_minus



def split_distributions(mid, sigma_plus, sigma_minus, start, end, size):
        #create posterior distribution based on previous detected transit
        
        N=size
        
        sizeplus = (N/(sigma_plus+sigma_minus))*sigma_plus
        sizeminus = (N/(sigma_plus+sigma_minus))*sigma_minus
        if type(sizeminus) == np.ndarray or type(sizeminus) == list:
            sizeminus = sizeminus[0]
            sizeplus = sizeplus[0]
        rounded_minus = round(sizeminus)
        rounded_plus = round(sizeplus)
        
        sample_minus = np.random.normal(loc=mid, scale=sigma_minus, size=int(rounded_minus))
        sample_plus = np.random.normal(loc=mid, scale=sigma_plus, size=int(rounded_plus))
        
        mid_index_minus = []
        for i in range(len(sample_minus)):
            if sample_minus[i] <= mid:
                mid_index_minus.append(i)

        mid_index_plus = []
        for i in range(len(sample_plus)):
            if sample_plus[i] >= mid:
                mid_index_plus.append(i)


        first_half = sample_minus[mid_index_minus]
        second_half = sample_plus[mid_index_plus]
        sample = np.append(first_half, second_half)
        '''
        x = np.linspace((mid - 4*sigma_minus), (mid+4*sigma_plus), 10000)
        kde_og = scipy.stats.gaussian_kde(sample)
        plt.figure(figsize = (20, 8))
        plt.subplot(2, 2, 1)
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
       
        plt.plot(sample, np.zeros(sample.shape), 'k+', ms=8, label = 'sample')
        plt.plot(x, kde_og(x), 'b-', label='KDE')
        plt.axvline(x=start, color='k', linestyle='--')
        plt.axvline(x=end, color='k', linestyle='--')
        plt.xlabel('Julian Days')
        plt.ylabel('PDF')
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()
        '''
        #delete points within observation period and create two new distributions
        indices = []
        for i in range(len(sample)):
            if start <= sample[i] <= end:
                indices.append(i)
                #what if the same index gets listed multiple times??
        new_sample = np.delete(sample, indices)
        if len(new_sample) == 0:
            print('The entire possible window for a transit has been observed')
            return 0, 0
        if len(indices) == 0:
            print('Nothing to refine here')
            return 0, 0

        else:  
            left_points = []
            for m in range(len(new_sample)):
                if new_sample[m] <= start:
                    left_points.append(new_sample[m])
            
            right_points = []
            for n in range(len(new_sample)):
                if new_sample[n] >= end:
                    right_points.append(new_sample[n])
            
            size_left = len(left_points)
            size_right = len(right_points)
            
            
            #find midpoint and errors on both new distributions
            x = np.linspace((mid - 6*sigma_minus), (mid + 6*sigma_plus), 10000)
            
            #plt.subplot(2, 2, 2)
            #plt.xlabel('Julian Days')
            #plt.ylabel('PDF')
            if len(left_points) > 1:
                kde_l = scipy.stats.gaussian_kde(left_points)
                peak_left = scipy.signal.find_peaks(kde_l(x))
                midpoint_left = x[max(peak_left[0])]
           
                
                #plt.plot(x, kde_l(x), 'r-', label='KDE')
                #plt.plot(left_points, np.zeros(len(left_points)), 'k+', ms=8, label='sample')
                #plt.axvline(x=start, color='k', linestyle='--')
                
                values_left, base_l = np.histogram(left_points, bins=10000)
                cumul_left = np.cumsum(values_left)
                #f_left = interp1d(cumul_left, base_l[:-1])
                top_95l = 0.975*(len(left_points))
                bottom_95l = 0.025*(len(left_points))
                
                
                pluserr_left = (np.interp(top_95l, cumul_left, base_l[:-1]) - midpoint_left)/2
                minuserr_left = (midpoint_left - np.interp(bottom_95l, cumul_left, base_l[:-1]))/2
                #errors_left = [pluserr_left, minuserr_left]
                
             
            if len(right_points) > 1:
                kde_r = scipy.stats.gaussian_kde(right_points)
                peak_right = scipy.signal.find_peaks(kde_r(x))
                midpoint_right = x[min(peak_right[0])]
                
                #plt.plot(x, kde_r(x), 'm-', label='KDE')
                #plt.plot(right_points, np.zeros(len(right_points)), 'k+', ms=8, label='sample')
                #plt.axvline(x=end, color='k', linestyle='--')
        
                values_right, base_r = np.histogram(right_points, bins=10000)
                cumul_right = np.cumsum(values_right)
            
                #f_right = interp1d(cumul_right, base_r[:-1])
                top_95r = 0.975*(len(right_points))
                bottom_95r = 0.025*(len(right_points))
                
                
                #xnew = np.linspace(min(cumul_right), max(cumul_right), len(base_r[:-1]))
                #plot cumulative
              
                
                pluserr_right = (np.interp(top_95r, cumul_right, base_r[:-1]) - midpoint_right)/2
                minuserr_right = (midpoint_right - np.interp(bottom_95r, cumul_right, base_r[:-1]))/2
                #errors_right = [pluserr_right, minuserr_right]
                
            
            if len(left_points) <= 1:
                midpoint_left = 0 
                #errors_left = 0
                pluserr_left = 0
                minuserr_left = 0
                size_left = 0
            if len(right_points) <= 1:
                midpoint_right = 0 
                #errors_right = 0
                pluserr_right = 0 
                minuserr_right = 0 
                size_right = 0
            
            #plt.tight_layout()
            #plt.show()
                
            return midpoint_left, midpoint_right, pluserr_right, minuserr_right, pluserr_left, minuserr_left, size_left, size_right, left_points, right_points
        
    
def get_refined(name, midpoint_base, plus_err, minus_err, start, end, period, err_per1, err_per2):
    # calculate all future transits and return transit that happens within start and 
    # end window.
    trans_used1 = transit_predictor(midpoint_base, abs(plus_err), abs(minus_err), period, abs(err_per1), abs(err_per2), start[0], end[0], False)
    
    midtrans_used1 = trans_used1[0]# transits within start and end window
    sigmaplus_used1 = trans_used1[1]
    sigmaminus_used1 = trans_used1[2]

    
    mid_before = trans_used1[4]# transits just before start and just after end 
    before_plus = trans_used1[5]
    before_minus = trans_used1[6]
    mid_after = trans_used1[7]
    after_plus = trans_used1[8]
    after_minus = trans_used1[9]
    
    
    midpoint = []
    plus_errors = []
    minus_errors = []
    size = []
    if len(midtrans_used1) == 0: #no predicted transit midpoint within start and end
        print('You have not observed where the midpoint was predicted to be')
        #chooses the transit before the window or after the window
        dist = split_distributions(mid_before, before_plus, before_minus, start[0], end[0], 1000)
        if dist == None:
            dist = split_distributions(mid_after, after_plus, after_minus, start[0], end[0], 1000)
        #what if they're both None??
        #create new base distributions from previous distribution fragments
        if dist[0] != 0:#append left values
            midpoint.append(dist[0])
            plus_errors.append(dist[4])
            minus_errors.append(dist[5])
            size.append(dist[6])
           
        if dist[1] != 0:#append right values
            midpoint.append(dist[1])
            plus_errors.append(dist[2])
            minus_errors.append(dist[3])
            size.append(dist[7])
            
    
    else: #start and end window does include the midpoint of a predicted transit
        dist = split_distributions(midtrans_used1[0], sigmaplus_used1[0], sigmaminus_used1[0], start[0], end[0], 1000)
    
        
        if dist[0] != 0:#append left values
            midpoint.append(dist[0])
            plus_errors.append(dist[4])
            minus_errors.append(dist[5])
            size.append(dist[6])
        if dist[1] != 0:#append right values
            midpoint.append(dist[1])
            plus_errors.append(dist[2])
            minus_errors.append(dist[3])
            size.append(dist[7])
        
    if type(start) == float: #if there is only one non-detection entry, return the
        #resultant distribtutions.
        return midpoint, plus_errors, minus_errors, size
    else:
        #if there is more than one non-detection recorded for the planet, iterate 
        #through all recorded start and end windows.
        for i in range(len(start))[1:]:
            before_size = []
            after_size = []
            midtrans_used2 = []
            sigmaplus_used2 = []
            sigmaminus_used2 = []
            size_used = []
            for j in range(len(midpoint)): 
                # use midpoints of the new distributions to predict which transits
                # should have been observed in the next non-detection window. 
                trans_used2 = transit_predictor(midpoint[j], abs(plus_errors[j]), abs(minus_errors[j]), period, abs(err_per1), abs(err_per2), start[i], end[i], False)
                if len(trans_used2[0]) >= 1:
                    # if non-detection window includes midpoint predicted using previous
                    # distributions, set this as the new posterior and remove the midpoint
                    # used to predict it from the list. 
                    midtrans_used2 = trans_used2[0]
                    sigmaplus_used2 = trans_used2[1]
                    sigmaminus_used2 = trans_used2[2]
                    size_used = size[j]
                    midpoint.pop(j)
                    plus_errors.pop(j)
                    minus_errors.pop(j)
                    size.pop(j)
                        
                else:
                    # if non-detection window does not include the midpoint of
                    # the transit predicted by this midpoint, append the predicted
                    # transit just before and just after the window to lists.
                    midtrans_used2 = 0
                    
                    mid_before = trans_used2[4]
                    before_plus = trans_used2[5]
                    before_minus = trans_used2[6]
                    before_size = size[j]
                    
                    mid_after = trans_used2[7]
                    after_plus = trans_used2[8]
                    after_minus = trans_used2[9]
                    after_size = size[j]
                    
                if midtrans_used2 != 0:
                    # if next start and end window does include a predicted midpoint, split
                    # this into two new distributions.
                    dist2 = split_distributions(midtrans_used2[0], sigmaplus_used2[0], sigmaminus_used2[0], start[i], end[i], size_used)
            
                    if dist2[0] != 0:#append left values
                        midpoint.append(dist2[0])
                        plus_errors.append(dist2[4])
                        minus_errors.append(dist2[5])
                        size.append(dist2[6])
                    if dist2[1] != 0:#append right values
                        midpoint.append(dist2[1])
                        plus_errors.append(dist2[2])
                        minus_errors.append(dist2[3])
                        size.append(dist2[7])
                        
                else:
                    # no midpoint predicted to be within start and end interval. Find nearest
                    # midpoint before and after the interval to see if 2 sigma tails of
                    # the distributions overlap into interval. Only refines one.
                    dist_bef = [0,0]
                    dist_aft = [0,0]
                
                    if (start[i] - mid_before) <= (2*before_plus):
                        dist_bef = split_distributions(mid_before, before_plus, before_minus, start[i], end[i], before_size) 
                       
                        if dist_bef[0] != 0:#append left values
                            midpoint.append(dist_bef[0])
                            plus_errors.append(dist_bef[4])
                            minus_errors.append(dist_bef[5])
                            size.append(dist_bef[6])
                        if dist_bef[1] != 0:#append right values
                            midpoint.append(dist_bef[1])
                            plus_errors.append(dist_bef[2])
                            minus_errors.append(dist_bef[3])
                            size.append(dist_bef[7])
                        
                    elif (mid_after - end[i]) <= (2*after_minus):
                        dist_aft = split_distributions(mid_after, after_plus, after_minus, start[i], end[i], after_size)
                    
                    
                        if dist_aft[0] != 0:#append left values
                            midpoint.append(dist_aft[0])
                            plus_errors.append(dist_aft[4])
                            minus_errors.append(dist_aft[5])
                            size.append(dist_aft[6])
                        if dist_aft[1] != 0:#append right values
                            midpoint.append(dist_aft[1])
                            plus_errors.append(dist_aft[2])
                            minus_errors.append(dist_aft[3])
                            size.append(dist_aft[7])
                    
                    #remove midpoint if either before or after 
                    if dist_aft[0] != 0 or dist_aft[1] != 0 or dist_bef[0] != 0 or dist_bef[1] != 0:
                            midpoint.pop(j) 
                            plus_errors.pop(j)
                            minus_errors.pop(j)
                            size.pop(j)
            
        
        return midpoint, plus_errors, minus_errors, size  
    
    
    
def target_selector(index, name, midpoint, err_plus, err_minus, period, err_per1, err_per2, start, end):
    
    #read in detection and non-detection data 
    non = pd.read_csv("NON_DETECTIONS.csv")
    df_non = DataFrame(non, columns=['Name', 'Start', 'End'])
    detect_data = pd.read_csv("DETECTIONS.csv")
    df_detect = DataFrame(detect_data, columns=['Planet Name', 'Midpoint', 'Plus Error', 'Minus Error'])
    
    names = []
    ind = []
    midpoints = []
    pluserrs = []
    minuserrs = []
    sizes = []
    
    for i in range(len(midpoint)):
        name_index_non = df_non.index[df_non['Name'] == name[i]].to_list()
        detection_index = df_detect.index[df_detect['Planet Name'] == name[i]].tolist()
        #has there been a non-detection recorded?
        if len(name_index_non) == 0:
            #has there been a detection recorded?
            if len(detection_index) == 0:
                base_mid = [midpoint[i]]
                base_pluserr = [err_plus[i]]
                base_minuserr = [err_minus[i]]
            
            else:
                recent_detection = detection_index[-1]
                if df_detect['Midpoint'][recent_detection] > midpoint[i]:
                    base_mid = [df_detect['Midpoint'][recent_detection]]
                    base_pluserr = [df_detect['Plus Error'][recent_detection]]
                    base_minuserr = [df_detect['Minus Error'][recent_detection]]
                else:
                    base_mid = [midpoint[i]]
                    base_pluserr = [err_plus[i]]
                    base_minuserr = [err_minus[i]]
                    
               
            size = [1000] #if there is only one possible solution, size is set to zero
            
        # refine ephemerides in the case of non-detection    
        else:
            # for each planet, if there is a non-detection listed, refine the ephemeris
            # and create new list of base midpoints to be used to predict future transits.
            base_mid = []
            base_pluserr = []
            base_minuserr = []
            size = []
            
            if len(detection_index) == 0:
                ref = get_refined(df_non['Name'][name_index_non], midpoint[i], abs(err_plus[i]), abs(err_minus[i]), df_non['Start'][name_index_non].tolist(), df_non['End'][name_index_non].tolist(), period[i], abs(err_per1[i]), abs(err_per2[i]))    
            else:
                recent_detection = detection_index[-1]
                if df_detect['Midpoint'][recent_detection] > midpoint[i] and df_non['Start'][name_index_non].tolist()[-1] > df_detect['Midpoint'][recent_detection]:
                    ref = get_refined(df_non['Name'][name_index_non], df_detect['Midpoint'][recent_detection].tolist(), df_detect['Plus Error'][recent_detection].tolist(), df_detect['Minus Error'][recent_detection].tolist(), df_non['Start'][name_index_non].tolist(), df_non['End'][name_index_non].tolist(), period[i], abs(err_per1[i]), abs(err_per2[i]))
                else:
                    ref = get_refined(df_non['Name'][name_index_non], midpoint[i], abs(err_plus[i]), abs(err_minus[i]), df_non['Start'][name_index_non].tolist(), df_non['End'][name_index_non].tolist(), period[i], abs(err_per1[i]), abs(err_per2[i]))
            
            if len(ref[0]) == 0:
                print('This non-detection did not refine the ephemeris')
                #has there been a detection recorded?
                if len(detection_index) == 0:
                    base_mid = [midpoint[i]]
                    base_pluserr = [err_plus[i]]
                    base_minuserr = [err_minus[i]]
            
                else:
                    recent_detection = detection_index[-1]
                    if df_detect['Midpoint'][recent_detection] > midpoint[i]:
                        base_mid = [df_detect['Midpoint'][recent_detection]]
                        base_pluserr = [df_detect['Plus Error'][recent_detection]]
                        base_minuserr = [df_detect['Minus Error'][recent_detection]]
                    else:
                        base_mid = [midpoint[i]]
                        base_pluserr = [err_plus[i]]
                        base_minuserr = [err_minus[i]]
                        
                size = [1000] #if there is only one possible solution, size is set to zero
            
            elif ref[0] != 0:
                if len(ref[0]) > 1:
                    for h in range(len(ref[0])):
                        base_mid.append(ref[0][h])
                        base_pluserr.append(ref[1][h])
                        base_minuserr.append(ref[2][h])
                        size.append(ref[3][h])
                        
                
                else:
                    base_mid.append(ref[0])
                    base_pluserr.append(ref[1])
                    base_minuserr.append(ref[2])
                    size.append(ref[3][0])
                   #base values are now the values obtained from refining the ephemerides
            
        # calculate midpoints of all transits happening in the interval for a specific 
        # planet.
        for j in range(len(base_mid)):  
            transits = transit_predictor(base_mid[j], abs(base_pluserr[j]), abs(base_minuserr[j]), period[i], abs(err_per1[i]), abs(err_per2[i]), start, end, False)
            if len(transits[0]) > 1:
                midpoints.append(transits[0])
                pluserrs.append(transits[1])
                minuserrs.append(transits[2])
                names.append(name[i])
                ind.append(index[i])
                sizes.append(size[j])
            elif len(transits[0]) == 1:
                midpoints.append(transits[0][0])
                pluserrs.append(transits[1][0])
                minuserrs.append(transits[2][0])
                names.append(name[i])
                ind.append(index[i])
                sizes.append(size[j])
                
        
    # for planets with a size not equal to zero, calculate how many of the points 
    # within the window THIS SHOULD BE DONE FOR REST OF THE PLANETS AS WELL
    total = []
    proportion = []

    for p in range(len(midpoints)):
        little_sizes = []
        points = []
        #plt.figure()
        #plt.xlabel('Julian Days')
        #plt.ylabel('PDF')
        #plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        if type(midpoints[p]) == list:
            mmm = midpoints[p]
            ppp = pluserrs[p]
            nnn = minuserrs[p]
        else:
            mmm = [midpoints[p]]
            ppp = [pluserrs[p]]
            nnn = [minuserrs[p]]
        
        for t in range(len(mmm)):
            
            N=sizes[p]        
            if ppp[t] < 0 or nnn[t] < 0:
                ppp[t] = abs(ppp[t])
                nnn[t] = abs(nnn[t])
            
            sizeplus = (N/(ppp[t]+nnn[t]))*ppp[t]
            sizeminus = (N/(ppp[t]+nnn[t]))*nnn[t]
            
            if type(sizeminus) == np.ndarray or type(sizeminus) == list:
                sizeminus = sizeminus[0]
                sizeplus = sizeplus[0]
                
            rounded_minus = round(sizeminus)
            rounded_plus = round(sizeplus)
            sample_minus = np.random.normal(loc=mmm[t], scale=nnn[t], size=int(rounded_minus))
            sample_plus = np.random.normal(loc=mmm[t], scale=ppp[t], size=int(rounded_plus))
                
            mid_index_minus = []
            for y in range(len(sample_minus)):
                if sample_minus[y] <= mmm[t]:
                    mid_index_minus.append(y)
                            
            mid_index_plus = []
            for k in range(len(sample_plus)):
                if sample_plus[k] >= mmm[t]:
                    mid_index_plus.append(k)


            first_half = sample_minus[mid_index_minus]
            second_half = sample_plus[mid_index_plus]
            sample = np.append(first_half, second_half)
            
            indices = []
            for r in range(len(sample)):
                if sample[r] <= start:
                    indices.append(r)
                if sample[r] >= end:
                    indices.append(r)
           

            #what if the same index gets listed multiple times??
            new_sample = np.delete(sample, indices)
            little_sizes.append(len(new_sample))
            if len(mmm) > 1:
                points.append(sample)
            else:
                points = [sample]
            
        '''       
        x = np.linspace((start - 1), (end+1), 10000)
        for i in range(len(points)):
            kde = scipy.stats.gaussian_kde(points[i])
            plt.plot(x, kde(x))
            plt.plot(points[i], np.zeros(len(points[i])), 'k+', ms=8, label='sample')
            plt.hist(points[i], density=True, bins=50)    
            
        plt.axvline(x=start, color='k', linestyle='--')
        plt.axvline(x=end, color='k', linestyle='--')
        plt.tight_layout()
        plt.show()
        '''
        total.append(len(sample))
        
        if len(mmm) > 1:
            sizes[p] = little_sizes
            proportion.append(sum(little_sizes)/len(sample))
        else:
            sizes[p] = len(new_sample)
            proportion.append((len(new_sample)/len(sample)))
       
                
            
    #write the planets and their transits to a csv
    titles = ['Name', 'Midpoint', 'Plus Error', 'Minus Error', 'Size in interval', 'Total Size', 'Proportion Included']
    rows = zip(names, midpoints, pluserrs, minuserrs, sizes, total, proportion)
    with open('observable_targets.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(titles)
        for col in rows:
            writer.writerow(col)
    return names, midpoints, pluserrs, minuserrs, sizes, total, proportion, ind



# FINDS BEGINNING AND END OF ASTRONOMICAL TWILIGHT
def sun_moon_tracking(long, lat, alt, date, utc_offset):
    #location = EarthLocation(long, lat, height) #NOTE: West longitudes are negative
    location = EarthLocation(long*u.deg, lat*u.deg, height = alt*u.m)    
    delta_midnight = np.linspace(-12, 12, 241)*u.hour 
    #gives 1 data point every 6mins - each value in the list has dtype: float64
    utc_midnight = Time(f'{date} 23:59:59.999')
    utc_frame_night = AltAz(obstime=utc_midnight+delta_midnight, location=location) 
    #list of times from 15:59:59 til 07:59:59 and list containing location in m, temp, humidity etc
    utc_moon = get_moon(utc_midnight+delta_midnight)
    utc_moon_altaz = utc_moon.transform_to(utc_frame_night)
    utc_sun_altaz = get_sun(utc_midnight+delta_midnight).transform_to(utc_frame_night)
    #sun and moon have format:  SkyCoord YYYY-MM-DD hh:mm:ss.sss
    #                                    ra(deg), dec(deg), distance(AU)
    #sun_altaz has format:      SkyCoord YYYY-MM-DD hh:mm:ss.sss
    #                                    az(deg), alt(deg), distance(m)
    
    #utc_sun_indexlist = []
    #for i in range(0, len(utc_sun_altaz)):
        #if utc_sun_altaz[i].alt.degree < -12: #ie IF astronomical twilight or astronomical night
            #utc_sun_indexlist.append(i)
        #else:
            #pass
    #utc_ast_twi_beg = utc_sun_altaz[utc_sun_indexlist[0]].obstime 
    #utc_ast_twi_end = utc_sun_altaz[utc_sun_indexlist[-1]].obstime 
    
    #local_midnight = utc_midnight - utc_offset*u.hour #this works for all
    #local_ast_twi_beg = utc_sun_altaz[utc_sun_indexlist[0]].obstime + utc_offset*u.hour
    #local_ast_twi_end = utc_sun_altaz[utc_sun_indexlist[-1]].obstime + utc_offset*u.hour
    
    # THIS STUFF WORKS NOW (mostly) #
    #print(f'ALL TIMES ARE GIVEN AS UTC TIMES')
    #print(f'Local midnight occurs at {local_midnight} UTC')
    #print(f'Astronomical twilight and night occurs between {utc_ast_twi_beg} and {utc_ast_twi_end} UTC')
    #print(f'Local astronomical twilight and night occurs between {local_ast_twi_beg} and {local_ast_twi_end} local time')
    
    #plt.figure(figsize = (18, 4))
    #plt.plot(delta_midnight, utc_sun_altaz.alt, color='r', label='Sun')
    #plt.plot(delta_midnight, utc_moon_altaz.alt, color=[0.75]*3, ls='--', label='Moon')
    #plt.xlabel('Hours from UTC Midnight')
    #plt.hlines(0, delta_midnight[0], delta_midnight[-1], label='Local Horizon')
    #plt.fill_between(delta_midnight, -20*u.deg, 90*u.deg, utc_sun_altaz.alt < -0*u.deg, color='0.5', zorder=0)
    #plt.fill_between(delta_midnight, -20*u.deg, 90*u.deg, utc_sun_altaz.alt < -12*u.deg, color='0.25', zorder=0)
    #plt.fill_between(delta_midnight, -20*u.deg, 90*u.deg, utc_sun_altaz.alt < -18*u.deg, color='k', zorder=0)
    #grey area indicates when sun is below the horizon
    #dark grey area indicates when the Sun is between -12 and -18 deg - ie ASTRONOMICAL TWILIGHT 
    #black area indicates when the sun is below -18 deg - ie ASTRONOMICAL NIGHT
    #plt.show()
    return delta_midnight, utc_moon, utc_moon_altaz, utc_sun_altaz

'''THIS FUNCTION SUPERCEDES sun_moon_tracking ATM'''
def find_astrotwi(long, lat, date):
        location = ephem.Observer()
        location.lat = str(lat)
        location.long = str(long)
        location.date = str(date)
        location.horizon = '-12'
        UTC_beg = location.previous_setting(ephem.Sun(), use_center=True)
        UTC_end = location.next_rising(ephem.Sun(), use_center=True)
        return UTC_beg, UTC_end


''' FINDS WHICH TARGETS ARE ABOVE +20(deg) WHEN THE SUN IS BELOW -12(deg) '''
def observability(index, target_name, ra, dec, long, lat, alt, date, utc_offset, SM0, SM1, SM2, SM3):
    location = EarthLocation(long*u.deg, lat*u.deg, height = alt*u.m)   
    #print(f'The RA and DEC of {target_name} in degrees is {ra}, {dec}')
    Tar_Coords_1 = SkyCoord(ra = ra*u.degree, dec = dec*u.degree)
    
    delta_midnight = SM0
    utc_midnight = Time(f'{date} 23:59:59.999')
    utc_frame_night = AltAz(obstime=utc_midnight+delta_midnight, location=location) 
    #list of times from 15:59:59 til 07:59:59 and list containing location in m, temp, humidity etc
    #frame_night is a list containing date and time in form 'yyyy-mm-dd hh:mm:ss.sss'
    Tar_altaz_window = Tar_Coords_1.transform_to(utc_frame_night)
    Tar_airmass_window = Tar_altaz_window.secz #AIRMASS values contained within this list have dype float64           
    
    sun_altaz = SM3  
    moon = SM1
    moon_altaz = SM2
    
    DEGREE = []
    count = 0
    above = 0
    tar_indexlist = []
    for i in range(0, len(Tar_altaz_window)):
        a = Tar_altaz_window[i].alt.degree #altitude (deg) of target
        DEGREE.append(a) 
        if sun_altaz[i].alt.degree < -12: 
        #ie IF astronomical twilight or night, start counting
            count = count + 1
            if a > 20:
                tar_indexlist.append(i)
                above = above + 1
            else:
                pass
        else:
            pass
    if above > 0:
        FLAG = 1
        INDEX = index
    else:
        FLAG = 0
        INDEX = 'NaN'
        
    if len(tar_indexlist) == 0:
        target_start = 0 #need something else here
        target_end = 0 #need something else here
        VIS_TIME = 0 
        
    else:
        target_start = sun_altaz[tar_indexlist[0]].obstime
        target_end = sun_altaz[tar_indexlist[-1]].obstime
        vis_time = sun_altaz[tar_indexlist[-1]].obstime - sun_altaz[tar_indexlist[0]].obstime
        VIS_TIME = vis_time*24 #number of hours target is visible for
        #print(f'The target ({target_name}) can be observed for {VIS_TIME}hours')
        #print(f'{target_name} is visible {((above/count)*100)} % of the night')
        
   # if pd.isna(trandur.item()):
        #print('There is no entry in the NASA database for the length of the transit')
    #else:
        #print(f'The transit of {target_name} is {trandur.item()*24}hours long')
    #print(f'The target is +20 degrees above the local horizon when the Sun is at < -12 degrees {(above/count)*100}% of the time.')    
    
    if FLAG == 1:
        Tar_Sep_Moon = []
        for i in range(0, len(moon)):
            Moon_RA = moon[i].ra.degree
            Moon_DEC = moon[i].dec.degree
            Moon_coords = SkyCoord(ra = Moon_RA*u.degree, dec = Moon_DEC*u.degree)
            separation = Tar_Coords_1.separation(Moon_coords).degree
            Tar_Sep_Moon.append(separation) 
            #print(separation)
            #if separation < 5:
             #   FLAG = 0
            #else:
    else:
        pass
    #print(Moon_coords[0])
    '''    
    # PLOTTING GRAPHS #
    plt.figure(figsize = (20, 8))
    plt.subplot(2, 2, 1)
    plt.plot(delta_midnight, Tar_airmass_window)
    plt.ylim(1, 4)
    plt.xlabel('Hours from UTC Midnight')
    plt.ylabel('Airmass [Sec(z)]')
    
    plt.subplot(2, 2, 2)
    plt.plot(delta_midnight, DEGREE)
    plt.ylim(0, 90)
    plt.xlabel('Hours from UTC Midnight')
    plt.ylabel('Altitude [deg]')
    
    plt.subplot(2, 2, 3)
    plt.plot(delta_midnight, Tar_Sep_Moon)
    plt.xlabel('Hours from UTC Midnight')
    plt.ylabel('Angular Separation from Moon [deg]')
    
    plt.subplot(2, 2, 4)
    plt.plot(delta_midnight, sun_altaz.alt, color='r', label='Sun')
    plt.plot(delta_midnight, moon_altaz.alt, color=[0.75]*3, ls='--', label='Moon')
    plt.scatter(delta_midnight, Tar_altaz_window.alt, c=Tar_altaz_window.az.degree, label=f'{target_name}', lw=0, s=8, cmap='viridis')
    plt.hlines(0, delta_midnight[0], delta_midnight[-1], label='Local Horizon')
    plt.fill_between(delta_midnight, -20*u.deg, 90*u.deg, sun_altaz.alt < -0*u.deg, color='0.5', zorder=0)
    plt.fill_between(delta_midnight, -20*u.deg, 90*u.deg, sun_altaz.alt < -12*u.deg, color='0.25', zorder=0)
    plt.fill_between(delta_midnight, -20*u.deg, 90*u.deg, sun_altaz.alt < -18*u.deg, color='k', zorder=0)
    plt.colorbar().set_label('Azimuth [deg]')  
    #grey area indicates when sun is below the horizon
    #dark grey area indicates when the Sun is between -12 and -18 deg - ie ASTRONOMICAL TWILIGHT 
    #black area indicates when the sun is below -18 deg - ie ASTRONOMICAL NIGHT
    
    plt.legend(loc='upper left')
    plt.xticks((np.arange(9)*2-8)*u.hour)
    plt.ylim(-20*u.deg, 90*u.deg)
    plt.xlabel('Hours from UTC Midnight')
    plt.ylabel('Altitude [deg]')
    plt.show()
    '''
    return target_name, FLAG, target_start, target_end, INDEX, (above/count), VIS_TIME

''' FILTERS OUT TARGETS BASED ON TOTAL UNCERTAINTY TIME '''
def unc_filter(index, target_name, plus_err, minus_err):
    new_index = []
    for i in range(len(target_name)):
        if type(plus_err[i]) == list:
            ppp = plus_err[i]
            nnn = minus_err[i]
        else:
            ppp = [plus_err[i]]
            nnn = [minus_err[i]]
        for j in range(len(ppp)):
            #total_unc is total 2sigma uncertainty - hence the 2*
            total_unc = abs(2*ppp[j]) + abs(2*nnn[j])
            
        if total_unc > 0.0208: #30mins
            new_index.append(index[i])
    return new_index

''' FINDS WHETHER A TARGET COULD BE OR CAN NEVER BE SEEN FROM THE USERS LOCATION '''
def dec_visible(index, dec, lat):
    VIS_INDEX = []
    NEG_INDEX = []    
    for i in index: 
        if lat > 0:
            if dec[i] - lat < -90:
                NEG_INDEX.append(i)
                #print('target is never observable')
            else:
                VIS_INDEX.append(i)
        elif lat <= 0:
            if dec[i] - lat > +90:
                NEG_INDEX.append(i)
                #print('target is never observable')
            else:
                VIS_INDEX.append(i)
    return VIS_INDEX, NEG_INDEX
    