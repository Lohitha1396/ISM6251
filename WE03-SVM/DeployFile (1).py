#!/usr/bin/env python
# coding: utf-8

# # DEPLOYING THE MODEL

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

Lawnmover_model = pickle.load(open("C:/Users/18137/Box/DSP/SVM/pickle.pkl", "rb"))

print("\n*******************************************")
print("* Lawn Mover Ownership Prediction Model *")
print("*********************************************\n")
Income = float(input("Enter the Income: "))
Lot_Size= float(input("Enter the Lotsize: "))
df = pd.DataFrame({'Income': [Income], 'Lot_Size': [Lot_Size]})
result = Lawnmover_model.predict(df)
probability = Lawnmover_model.predict_proba(df)
Ownership = ('Owner', 'Nonowner')
print(f"\n The Lawn mover Ownership Prediction Model indicates the probability of Ownership being {probability[0][1]:.4f}, and it belongs to the: {Ownership[result[0]]} category of Ownership.\n")


# In[ ]:




