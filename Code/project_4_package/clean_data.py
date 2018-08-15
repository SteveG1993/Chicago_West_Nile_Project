#This handles the preparation of the data before we model

import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib        as mtl
import matplotlib.pyplot as plt
import requests
import collections
import time
import statsmodels.api as sm



from datetime                        import timedelta
from datetime                        import date
from datetime                        import datetime


from sklearn.preprocessing           import StandardScaler

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_selection       import SelectPercentile
from sklearn.feature_selection       import SelectFromModel

from sklearn.model_selection         import train_test_split
from sklearn.model_selection         import GridSearchCV

from sklearn.ensemble                import BaggingClassifier
from sklearn.ensemble                import RandomForestClassifier
from sklearn.ensemble                import ExtraTreesClassifier
from sklearn.ensemble                import GradientBoostingClassifier

from sklearn.linear_model            import LogisticRegression
from sklearn.linear_model            import LinearRegression

from sklearn.metrics                 import confusion_matrix
from sklearn.metrics                 import classification_report
from sklearn.metrics                 import roc_auc_score
from sklearn.metrics                 import accuracy_score
from sklearn.metrics                 import f1_score


from sklearn.tree                    import DecisionTreeClassifier

from sklearn.neighbors               import KNeighborsClassifier



from sklearn.pipeline                import Pipeline


from sklearn.svm                     import SVC




# This is the CodeSum dictionary.
dictonary = {'+FC': 'TORNADO/WATERSPOUT',
        'FC': 'FUNNEL CLOUD',
        'TS': 'THUNDERSTORM',
        'GR': 'HAIL',
        'RA': 'RAIN',
        'DZ': 'DRIZZLE',
        'SN': 'SNOW',
        'SG': 'SNOW  GRAINS',
        'GS': 'SMALL HAIL &/OR SNOW PELLETS',
        'PL':  'ICE PELLETS',
        'IC':  'ICE CRYSTALS',
        'FG+': 'HEAVY FOG (FG & LE.25 MILES VISIBILITY)',
        'FG':  'FOG',
        'BR':  'MIST',
        'UP':  'UNKNOWN PRECIPITATION',
        'HZ':  'HAZE',
        'FU':  'SMOKE',
        'VA':  'VOLCANIC ASH',
        'DU':  'WIDESPREAD DUST',
        'DS':  'DUSTSTORM',
        'PO':  'SAND/DUST WHIRLS',
        'SA':  'SAND',
        'SS':  'SANDSTORM',
        'PY':  'SPRAY',
        'SQ':  'SQUALL',
        'DR':  'LOW DRIFTING',
        'SH':  'SHOWER',
        'FZ':  'FREEZING',
        'MI':  'SHALLOW',
        'PR':  'PARTIAL',
        'BC':  'PATCHES',
        'BL':  'BLOWING',
        'VC':  'VICINITY',
        '-': 'LIGHT',
        '+':   'HEAVY',
        'NO SIGN': 'MODERATE',
        'VCTS': 'VICINITY THUNDERSTORM',
        'TSRA': 'THUNDERSTORM RAIN',
        'BCFG': 'PATCHES FOG',
        'MIFG': 'SHALLOW FOG',
        'VCFG': 'VICINITY FOG' }



# Function to convert weather codes to words
def get_codes(string,k):
    l = string.split()
    if k in l:
        return 1
    else:
        return 0

weather = pd.read_csv('../Assets/input/weather.csv')
train   = pd.read_csv('../Assets/input/train.csv')


print('Train Rows, Columns before',train.shape)
print('Weather Rows,Columns before',weather.shape)
w2 = weather[weather['Station']==1]
df = pd.merge(train,w2,on='Date',how='inner')
print('rows,columns after merge',df.shape)

#train['Date']=train['Date'].map(datetime.toordinal)

df['Date_Number']= pd.to_datetime(df['Date']).map(datetime.toordinal)


# Dropping Water1 and Depth from df_weather
df.drop(['Water1'],axis=1,inplace=True)
df.drop(['Depth'],axis=1,inplace=True)
df.drop(['SnowFall'],axis=1,inplace=True)

# Filling in empty spaces in CodeSum
df['CodeSum'] = df['CodeSum'].apply(lambda x: 'nothing' if x == ' ' else x)

# Filling in "-" in Sunrise and Sunset
df['Sunrise'] = df['Sunrise'].apply(lambda x: np.NaN if x == '-' else x)
df['Sunset'] = df['Sunset'].apply(lambda x: np.NaN if x == '-' else x)

# Creating features from the weather codes in the CodeSum feature.
for k in dictonary.keys():
    df[dictonary[k]] = df['CodeSum'].apply(lambda x: get_codes(x,k))
