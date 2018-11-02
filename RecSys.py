# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:36:53 2018

@author: Josh
"""
from surprise import SVD
from surprise import NMF
from surprise import KNNBasic
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def setDF(perf, alg, row):
    df.loc[row,['Algorithm']] = alg
    df.loc[row,['RMSE Fold 1']] = perf['rmse'][0]
    df.loc[row,['RMSE Fold 2']] = perf['rmse'][1]
    df.loc[row,['RMSE Fold 3']] = perf['rmse'][2]
    df.loc[row,['RMSE Mean']] = ((perf['rmse'][0] + perf['rmse'][1] + perf['rmse'][2])/3)
    df.loc[row,['MAE Fold 1']] = perf['mae'][0]
    df.loc[row,['MAE Fold 2']] = perf['mae'][1]
    df.loc[row,['MAE Fold 3']] = perf['mae'][2]
    df.loc[row,['MAE Mean']] = ((perf['mae'][0] + perf['mae'][1] + perf['mae'][2])/3)

#file_path = os.path.expanduser('restaurant_ratings')
reader = Reader(line_format='user item rating timestamp',sep='\t')
data = Dataset.load_from_file('restaurant_ratings.txt',reader=reader)

data.split(n_folds=3)

#Starting dataframe to store needed values
df = pd.DataFrame([],
     index = [0,1,2,3,4,5,6,7],
     columns = ['Algorithm','RMSE Fold 1','RMSE Fold 2','RMSE Fold 3','RMSE Mean','MAE Fold 1','MAE Fold 2','MAE Fold 3','MAE Mean'])
'''
#SVD algorithm
algo = SVD()
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'SVD',0)

print '\n'
#PMF algorithm
algo = SVD(biased=False)
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'PMF',1)

print '\n'
#NMF algorithm
algo = NMF()
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'NMF',2)


print '\n'
#User based collaborative filtering algorithm
algo = KNNBasic(sim_options = {'user_based': True})
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'User-Based',0)

print '\n'
#Item based collaborative filtering algorithm
algo = KNNBasic(sim_options = {'user_based': False})
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'Item-Based',4)

print '\n'
print df[['Algorithm', 'RMSE Fold 1', 'MAE Fold 1']]


print '\n'
print df[['Algorithm', 'RMSE Fold 2', 'MAE Fold 2']]


print '\n'
print df[['Algorithm', 'RMSE Fold 3', 'MAE Fold 3']]


print '\n'
print df[['Algorithm', 'RMSE Mean', 'MAE Mean']]

#User based collaborative filtering algorithm w/ MSD similarity
print '\n'
algo = KNNBasic(sim_options = {'name':'MSD', 'user_based': True})
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'User-Based MSD',1)

#User based collaborative filtering algorithm w/ Cosine similarity
print '\n'
algo = KNNBasic(sim_options = {'name':'cosine', 'user_based': True})
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'User-Based Cosine',2)

#User based collaborative filtering algorithm w/ Pearson similarity
print '\n'
algo = KNNBasic(sim_options = {'name':'pearson', 'user_based': True})
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'User-Based Pearson',3)

#Item based collaborative filtering algorithm w/ MSD similarity
print '\n'
algo = KNNBasic(sim_options = {'name':'MSD', 'user_based': False})
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'Item-Based MSD',5)

#Item based collaborative filtering algorithm w/ Cosine similarity
print '\n'
algo = KNNBasic(sim_options = {'name':'cosine', 'user_based': False})
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'Item-Based Cosine',6)

#Item based collaborative filtering algorithm w/ Pearson similarity
print '\n'
algo = KNNBasic(sim_options = {'name':'pearson', 'user_based': False})
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)
setDF(perf,'Item-Based Pearson',7)

userDF = df[:4]
itemDF = df[4:]

print userDF,'\n'
print itemDF

sns.set(font_scale=1)
g = sns.factorplot(x="Algorithm", y="MAE Mean",
                    data=userDF, saturation=.5,
                    kind="bar", ci=None, aspect=2)
(g.set_axis_labels("", "MAE Mean")
    .set_xticklabels(["User-Based", "User-Based MSD","User-Based Cosine","User-Based Pearson"])
    .set_titles("{col_name} {col_var}")
    .despine(left=True))  
#plt.figure(figsize=(100,5))
g.fig.suptitle('Performance on Mean MAE, User-Based');
plt.savefig('image1.png')

sns.set(font_scale=1)
g = sns.factorplot(x="Algorithm", y="RMSE Mean",
                    data=userDF, saturation=.5,
                    kind="bar", ci=None, aspect=2)
(g.set_axis_labels("", "RMSE Mean")
    .set_xticklabels(["User-Based", "User-Based MSD","User-Based Cosine","User-Based Pearson"])
    .set_titles("{col_name} {col_var}")
    .despine(left=True))  
#plt.figure(figsize=(100,5))
g.fig.suptitle('Performance on Mean RMSE, User-Based');
plt.savefig('image2.png')

#

sns.set(font_scale=1)
g = sns.factorplot(x="Algorithm", y="MAE Mean",
                    data=itemDF, saturation=.5,
                    kind="bar", ci=None, aspect=2)
(g.set_axis_labels("", "MAE Mean")
    .set_xticklabels(["Item-Based", "Item-Based MSD","Item-Based Cosine","Item-Based Pearson"])
    .set_titles("{col_name} {col_var}")
    .despine(left=True))  
#plt.figure(figsize=(100,5))
g.fig.suptitle('Performance on Mean MAE, Item-Based');
plt.savefig('image3.png')

sns.set(font_scale=1)
g = sns.factorplot(x="Algorithm", y="RMSE Mean",
                    data=itemDF, saturation=.5,
                    kind="bar", ci=None, aspect=2)
(g.set_axis_labels("", "RMSE Mean")
    .set_xticklabels(["Item-Based", "Item-Based MSD","Item-Based Cosine","Item-Based Pearson"])
    .set_titles("{col_name} {col_var}")
    .despine(left=True))  
#plt.figure(figsize=(100,5))
g.fig.suptitle('Performance on Mean RMSE, Item-Based');
plt.savefig('image4.png')
'''
newDF = pd.DataFrame([],
     index = [1,2,3,4,5,6,7,8,9,10],
     columns = ['K','RMSE'])

for p in range(1,11):
    #User based collaborative filtering algorithm
    algo = KNNBasic(k=p, sim_options = {'user_based': True})
    perf = evaluate(algo,data,measures=['RMSE'])
    newDF.loc[p,['K']] = p
    newDF.loc[p,['RMSE']] = ((perf['rmse'][0] + perf['rmse'][1] + perf['rmse'][2])/3)

print newDF
sns.set(style="darkgrid")  
sns.barplot(x='K',y='RMSE',data=newDF)
plt.savefig('image5.png')

for p in range(1,11):
    #Item based collaborative filtering algorithm
    algo = KNNBasic(k=p, sim_options = {'user_based': False})
    perf = evaluate(algo,data,measures=['RMSE'])
    newDF.loc[p,['K']] = p
    newDF.loc[p,['RMSE']] = ((perf['rmse'][0] + perf['rmse'][1] + perf['rmse'][2])/3)

print newDF
sns.set(style="darkgrid")  
sns.barplot(x='K',y='RMSE',data=newDF)
plt.savefig('image6.png')





