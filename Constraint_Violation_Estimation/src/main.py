# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:28:53 2021

@author: t.urita
"""
import numpy as np
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# load dataset and plot class count
data = np.loadtxt("data.csv", delimiter = ",")
ans = np.loadtxt("answer.csv", delimiter = ",")

sns.countplot(ans).set_xticklabels(["f1, f2", "f1", "f2", "feasible"])

# split dataset
K = 10
skf = StratifiedKFold(n_splits = K)
idx = [*skf.split(data, ans)]

# generate classifier
knc = KNeighborsClassifier()
rf = RandomForestClassifier(random_state = 2021)
mdls = [knc, rf]

# calc and show acc
accs = [*map(lambda mdl: cross_val_score(mdl, data, ans, cv = idx).mean(), mdls)]

print("-----  4 classes classification  -----")
print("K Nearest Neighbor accuracy : {}".format(accs[0]))
print("     Random Forest accuarcy : {}".format(accs[1]))

# f1 classification
ans_f1 = np.ones_like(ans)
mask_f1 = (ans == 0) | (ans == 1)
ans_f1[mask_f1] = 0
sns.countplot(ans_f1).set_xticklabels(["f1", "feasible"])

idx_f1 = [*skf.split(data, ans_f1)]

accs_f1 = [*map(lambda mdl: cross_val_score(mdl, data, ans_f1, cv = idx_f1).mean(), mdls)]

print("\n------  f1 is feasible or not  -------")
print("K Nearest Neighbor accuracy : {}".format(accs_f1[0]))
print("     Random Forest accuarcy : {}".format(accs_f1[1]))

#f2 classification
ans_f2 = np.ones_like(ans)
mask_f2 = (ans == 0) | (ans == 2)
ans_f2[mask_f2] = 0
sns.countplot(ans_f2).set_xticklabels(["f2", "feasible"])

idx_f2 = [*skf.split(data, ans_f1)]

accs_f2 = [*map(lambda mdl: cross_val_score(mdl, data, ans_f2, cv = idx_f2).mean(), mdls)]

print("\n------  f2 is feasible or not  -------")
print("K Nearest Neighbor accuracy : {}".format(accs_f2[0]))
print("     Random Forest accuarcy : {}".format(accs_f2[1]))