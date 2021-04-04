#!/usr/bin/env python2
# -*- coding: utf-8 -*-


#crowdflower processor script.

import sys
import pandas as pd
import numpy as np
import scipy.stats as sp
import os
from matplotlib import pyplot as plt
import seaborn as sns

def getCenter(questionDF, grouper, center='median'):
    """groups by the task id and then pulls the central tendency"""
    from scipy import stats
    if center=='mode':
        return questionDF.groupby(grouper).agg(lambda x: stats.mode(x)[0][0]).reset_index()
    elif center=='mean':
        return questionDF.groupby(grouper).mean().reset_index()
    else:
        return questionDF.groupby(grouper).median().reset_index()
    
def taskCountAboveThreshold(taskDF,smlMeasure,threshold,grouper, percentile=False):
    """takes a measure of SML and for each grouped item (typically a job), it returns
    the count of tasks in that job with an smlMeasure above the threshold in the function.
    The measure is at the job level. Also returned are total task counts and proportions.
    If percentile is specified, it will convert the threshold to a percentile and run things that way"""
    if percentile:
        threshold = np.percentile(taskDF[smlMeasure], threshold)
    thresTaskCount = taskDF[taskDF[smlMeasure]>threshold][[grouper, smlMeasure]].groupby(grouper).count()
    allTaskCount = taskDF[[grouper, smlMeasure]].groupby(grouper).count()
    outdf = thresTaskCount.join(allTaskCount,lsuffix='a',rsuffix='b').reset_index()
    outdf.rename(columns={smlMeasure+'a':smlMeasure+'_threshold'+str(threshold), smlMeasure+'b':'allTaskCount'}, inplace=True)
    outdf[smlMeasure+'_thresholdProp'+str(threshold)] = outdf[smlMeasure+'_threshold'+str(threshold)]/outdf.allTaskCount
    return outdf

def seabornMultiHist(df, title, xlabel, ylabel):
    """creates multiple histogram plot used in figure 1"""
    plt.style.use('ggplot')
    df.plot(kind='hist',alpha=.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("hist_percentiles.png")
    plt.show()

def plotSingleSML(OESdetail, smlMeasure, percentile1, sml_label, perc1_label):
    """plotting function for single SML measure"""
    plt.style.use('ggplot')
    plt.scatter(OESdetail[percentile1], OESdetail[smlMeasure])
    plt.title(sml_label+' vs. '+perc1_label)
    plt.xlabel(perc1_label)
    plt.ylabel(sml_label)
    plt.savefig(smlMeasure+"singlePlot.png",dpi=300)
    plt.show()

def plotStackedSML(OESdetail, smlMeasure, percentile1, percentile2, perc1_label, perc2_label, sml_label, color='red'):
    """plotting function to stack 2 SML measures"""
    plt.style.use('ggplot')
    plt.figure(1)
    plt.rcParams["figure.figsize"] = [6.0,16.0]
    plt.subplot(211)
    plt.scatter(OESdetail[percentile1],OESdetail[smlMeasure],color=color)
    plt.title(sml_label+" Score vs. "+perc1_label)
    plt.xlabel("Occupational "+perc1_label)
    plt.ylabel(sml_label+" Score")
    plt.subplot(212)
    plt.scatter(OESdetail[percentile2],OESdetail[smlMeasure], color=color)
    plt.title(sml_label+" vs. "+perc2_label)
    plt.xlabel("Occupational "+perc2_label)
    plt.ylabel(sml_label+" Score")
    plt.tight_layout()
    plt.savefig(smlMeasure+".png",dpi=300)
    plt.show()


if __name__ == '__main__':
    cf_dir, cf_file, cf_phys_file = sys.argv[1:4]
    os.chdir(cf_dir)
    cf = pd.read_csv(cf_file)
    cf_phys = pd.read_csv(cf_phys_file) #physical questions asked in separate run
    
    #filter to people who understand the task
    cfu = cf[cf.understand>2]
    cfpu = cf_phys[cf_phys.understand>2]
    qfields = ['q'+str(item+1) for item in list(range(21))] + ['dwa_id']
    pqfields = ['q'+str(item+1) for item in [21,22]] + ['dwa_id']
    qs = cfu[qfields] # only the questions and task id
    pqs = cfpu[pqfields]
    qs['variance']=qs[list(qs)[:-1]].var(axis=1) #calculate row-wise variance 
    #not doing the variance filter for the physical questions (only 2 qs)
    qs2 = qs[qs.variance!=0] #drop the zero variance rows. These are people who answered one thing.
    centerQ = getCenter(qs2, 'dwa_id', center='median') #return the median answer for what's left
    centerQp = getCenter(pqs, 'dwa_id', center='median')
    centerQp[['q22','q23']] = 6.0 - centerQp[['q22','q23']] #map physical to bad for ML
    centerQ['qD'] = centerQ[['q15','q16','q17','q18']].max(axis=1) #max of the data questions
    qfields.insert(-1,'qD')
    centerQ = pd.merge(centerQ, centerQp, on='dwa_id')
    qfields = qfields+pqfields[0:2] #now we've merged in the physical-ness
    
    
    #okay now we have the ratings for each activity. Let's join them to the tasks.
    task_file = sys.argv[4]
    tasks = pd.read_excel(task_file) #it's the Tasks to DWAs.xlsx file
    tasksDWAs = tasks.merge(centerQ, left_on=['DWA ID'], right_on=['dwa_id'], how='left')
    #average over all of the DWAs in a task to get the average task values.
    #we omit the dwa_id here. It's grouping at the task level.
    taskScores = getCenter(tasksDWAs[[item for item in qfields if item[0]=='q'] + ['Task ID']],'Task ID','mean')
    #merge back and store these to csv
    onlyTasks = tasks[['O*NET-SOC Code','Title','Task ID','Task']].drop_duplicates()
    taskScores2 = taskScores.merge(onlyTasks, on='Task ID')

    taskScores2.to_csv('Tasks_Scores.csv', index=False) #save the task-level scores
    
    #Now with the task data, we can aggregate to the job level
    jobTaskRatingFile, jobDataFile = sys.argv[5:7]
    jobs = pd.read_excel(jobTaskRatingFile) #Task Ratings.xlsx from O*NET db
    jobsData = pd.read_excel(jobDataFile).rename(columns={"Title": "Title_jobsData"}) #Task Statements.xlsx from O*NET db
    #get the importance values out for each of the tasks
    jobTask = jobs[jobs['Scale ID']=='IM'][['O*NET-SOC Code','Title','Task ID','Task','Data Value']]
    jobTaskScores = pd.merge(jobTask, taskScores2, on=['Task ID','O*NET-SOC Code','Title','Task'])
    jobTaskScores['Data Value'] = jobTaskScores['Data Value']/5.0 #scale to 0-1
    jobTaskScores['weight'] = jobTaskScores['Data Value'] / jobTaskScores.groupby('O*NET-SOC Code')['Data Value'].transform('sum')
    JTS = jobTaskScores.copy() #prevent overwriting
    
    #weighted average calculations for the job
    for q in list(JTS)[5:-1]:
        JTS['w'+q] = JTS[q]*JTS['weight']
    JTSfields = [item for item in list(JTS) if item[0:2]=='wq']
    wJTS = JTS.groupby(['O*NET-SOC Code','Title'])[JTSfields].sum().reset_index()
    #join the calcs to the data we have for the job
    jobScores = pd.merge(wJTS, jobsData, how='left',on='O*NET-SOC Code').drop_duplicates()

    jobScores.to_csv('Job_Scores.csv', index=False) #save jobScores for later

    #variance calculations
    JTSV = jobTaskScores.copy() #separate copy for the variances
    vJTS = JTSV.merge(wJTS, on=['O*NET-SOC Code','Title'],how='left')
    for q in [k for k in list(vJTS) if k[0]=='q']: #for all the original question values
        vJTS['dV'+q] = vJTS['weight']*(vJTS[q]-vJTS['w'+q])**2 #dV is for sq. deviation value
    vJTSfields = [item for item in list(vJTS) if item[0:2]=='dV']
    #wvJTS = np.sqrt(vJTS.groupby(['O*NET-SOC Code','Title'])[vJTSfields].sum()).reset_index()
    wvJTS = vJTS.groupby(['O*NET-SOC Code','Title'])[vJTSfields].sum().reset_index()
    jobVarianceScores = pd.merge(wvJTS, jobsData, how='left',on='O*NET-SOC Code').drop_duplicates()
    jobVarianceScores.to_csv('JobVariance_Scores.csv', index=False) #save variance scores for later
    #merge it all together
    allscores = pd.merge(jobScores,wvJTS,on=['O*NET-SOC Code','Title'])

    allscores.to_csv('JobScores_mean_variance.csv') #save allscores
    
    #categorizing questions as SML or Measurability
    SML = ['q3','q5','q6','q7','q8','q9','q10','q11','q12','q13','q14','q19','q20','q21','q22','q23'] #SML without measurement / data fields
    measurability = ['q1','q2','q4','qD'] # Measurement and data fields - maximum is applied twice to increase "weight"
    #mean and variance for each of SML, measurability.
    #this treats each question equally. In future iterations, weights may be applied.
    allscores['mSML'] = (1/float(len(SML+measurability)))*allscores[['w'+ item for item in SML+measurability]].sum(axis=1)
    allscores['vmSML'] = (1/float(len(SML+measurability)))*allscores[['dV'+ item for item in SML+measurability]].sum(axis=1)
    allscores['sdmSML'] = np.sqrt(allscores['vmSML'])
    
    allscores.to_csv('allscores_SML.csv',index=False)

    #task-level SML on the basis of averages. Used to generate figure 1.
    cQ = centerQ.copy() # for activity-level info
    cQ['mSML']=(1/float(len(SML+measurability)))*cQ[[item for item in SML+measurability]].sum(axis=1)
    taskSML = JTS.copy() #using the unweighted tasks for reorganization
    taskSML['SML'] =  (1/float(len(SML)))*taskSML[[item for item in SML]].sum(axis=1)
    taskSML['measure'] = (1/float(len(measurability)))*taskSML[[item for item in measurability]].sum(axis=1)
    taskSML['mSML'] = (1/float(len(SML+measurability)))*taskSML[[item for item in SML+measurability]].sum(axis=1)
    snipper = lambda x: x[:x.find('.')]
    taskSML['occ code'] = taskSML['O*NET-SOC Code'].apply(snipper)
    #90th-75th-50th percentile version. Reorganization measures
    perc90 = taskCountAboveThreshold(taskSML,'mSML',90,'Title',percentile=True)
    perc75 = taskCountAboveThreshold(taskSML,'mSML',75,'Title',percentile=True)
    perc50 = taskCountAboveThreshold(taskSML,'mSML',50,'Title',percentile=True)
    
    #plot it now
    plotin = pd.DataFrame()
    plotin['90th Percentile'] = perc90['mSML_thresholdProp3.85']
    #plotin['75th Percentile'] = perc75['mSML_thresholdProp3.675']
    plotin['50th Percentile'] = perc50['mSML_thresholdProp3.45']
    seabornMultiHist(plotin, '', \
                     "Proportion of Tasks in Occupation with SML Above Percentile",'')
                         
    # #BLS Wage Data 2016 - used to calculate figure 2.
    # OESfile = sys.argv[7]
    # wagesOES = pd.read_excel(OESfile)
    # allscores['occ code']=allscores['O*NET-SOC Code'].apply(snipper)
    # #for SML
    # occScoresSML = allscores[['mSML','vmSML','occ code']].groupby('occ code').mean().reset_index()
    # OES = pd.merge(wagesOES, occScoresSML, on=['occ code'])
    # OES['sdmSML']=np.sqrt(OES['vmSML']) #variation within tasks
    # OESdetail = OES[(OES.group=='detailed')&(OES.naics_title=='Cross-industry')\
    #                 &(OES.area_title=='U.S.')&(OES.a_median.isin(['*','#'])==False)]
    # OESdetail['a_median'] = OESdetail['a_median'].astype(float)
    # OESdetail['log_median'] = np.log(OESdetail['a_median'])
    # OESdetail['log_median_percentileWage'] = \
    #     OESdetail['log_median'].apply(lambda x: sp.percentileofscore(OESdetail['log_median'],x))
    # OESdetail['wagebill'] = OESdetail['tot_emp']*OESdetail['a_mean']
    # OESdetail['wagebill_percentile'] = OESdetail['wagebill'].apply(lambda x: sp.percentileofscore(OESdetail['wagebill'],x))
    #
    # OESdetail.to_csv("OESdetail_SML.csv", index=False)

    #plotStackedSML(OESdetail,'sdmSML','log_median_percentileWage','wagebill_percentile',\
    #	'Occupational Log Median Wage Percentile','Occupational Wage Bill Percentile','sdSML',color='blue')