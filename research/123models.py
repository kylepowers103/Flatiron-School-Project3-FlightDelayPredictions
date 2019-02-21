# subscription_id = "<subscription_id>"
# resource_group = "myrg"
# workspace_name = "myws"
# workspace_region = "eastus2"


file_total=[]
files=["On_time2015_00.csv","On_time2015_01.csv","On_time2015_02.csv","On_time2015_03.csv","On_time2015_04.csv",'On_time2016_00.csv','On_time2016_01.csv','On_time2016_02.csv','On_time2016_03.csv','On_time2016_04.csv','On_time2017_00.csv','On_time2017_01.csv','On_time2017_02.csv','On_time2017_03.csv','On_time2017_04.csv']
for file in files:
    new=pd.read_csv('compressed/'+file)#,usecols=['Month', 'DayofMonth', 'DayOfWeek','Origin', 'Dest', 'UniqueCarrier', 'FlightNum', 'Origin', 'Dest', 'DepTime', 'DepDelay', 'TaxiOut', 'WheelsOff', 'WheelsOn', 'TaxiIn', 'ArrTime', 'ArrDelay', 'Cancelled', 'Diverted', 'AirTime', 'Flights', 'Distance', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'DivAirportLandings'])
    file_total.append(new)
# frame = [df,df2, df3, df4]
merged_df = pd.concat(file_total, axis = 0)
merged_df.describe()
# frame = [df,df2, df3, df4]
merged_df = pd.concat(file_total, axis = 0)
merged_df.describe()




# "{ "subscription_id": "7fa66662-a863-41de-8826-4a7c2f6ca7b2", "resource_group": "cloud-shell-storage-eastus", "workspace_name": "jan14newestflighttest" }"

from azureml.core import Workspace

ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group, 
                      location = workspace_region)
ws.get_details()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

# Matplotlib for visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Set default font size
plt.rcParams['font.size'] = 24

from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Imputing missing values and scaling values
from sklearn.preprocessing import Imputer, MinMaxScaler

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#%%
import pandas as pd
df = pd.read_csv("./dataframe.csv")
df.head() # shape = (278220, 35)


df=df[['DepDelay','hour', 'pressure', 'humidity', 'temperature', 'wind_speed', 'description', 'Origin', 'Dest', 'DepTime', 'Distance', 'ArrTime','AirTime']]
X=df[['hour', 'pressure', 'humidity', 'temperature', 'wind_speed', 'description', 'Origin', 'Dest', 'DepTime', 'Distance', 'ArrTime','AirTime']]
y = df['DepDelay']
# 'DepTime' 'ArrTime' 'AirTime']



#%%
# features.shape
test1234=df


#%%
def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
#     y = df['DepDelay']
    # Dont want to remove correlations between Energy Star Score
    y = df['DepDelay']
    x = df.drop(columns = ['DepDelay'])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = df.drop(columns = drops)
#     x = df.drop(columns = ['hour', 'pressure', 'humidity', 'temperature', 'wind_speed',
#        'description', 'Origin', 'Dest', 'DepTime', 'Distance', 'ArrTime',
#        'AirTime'])
    
    # Add the score back in to the data
    x['DepDelay'] = y
               
    return x


#%%
features123 = remove_collinear_features(df, 0.6);




#%%
categorical_subset = features123[['Origin','Dest','description']]
numeric_subset=features123[['hour', 'pressure', 'humidity', 'temperature', 'wind_speed', 'Distance']]
# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features1234 = pd.concat([numeric_subset, categorical_subset], axis = 1)



features = features1234 #(278220, 13)
targets = y #pd.DataFrame(y)  #278220,)


#%%
### changed to calculate just with depdelay
###Convert y to set of classes
y_ontime = y[(y <= 15) & (y >= 0)] # 0 <= y <= 15 is on time by FAA standards
y_early = y[y < 0] # y < 0 is flight left early
y_delay = y[(y > 15) & (y <= 60)] # 15 < y <= 60 we'll consider as delayed
y_sig_delay = (y[y > 60]) # 60 < y we'll considered significantly delayed

y_early = ["y_early"]*len(y_early)
y_ontime = ['y_ontime']*len(y_ontime)
y_delay = ['y_delay']*len(y_delay)
y_sig_delay = ['y_sig_delay']*len(y_sig_delay)
y = pd.DataFrame(y_early + y_ontime + y_delay + y_sig_delay)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
# y = column(y, warn=True)
y_new = le.transform(y)
y_new.shape #(278220,)
y_new
targets=y_new


#%%
# Split into 70% training and 30% testing set
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)


#%%
# Create the scaler object with a range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
scaler.fit(X)

# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)
# y = scaler.transform(y)
# y_test = scaler.transform(y_test)
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# #%%
# # Function to calculate mean absolute error
# def mae(y_true, y_pred):
#     return np.mean(abs(y_true - y_pred))

# # Takes in a model, trains the model, and evaluates the model on the test set
# def fit_and_evaluate(model):
    
#     # Train the model
#     model.fit(X, y)
    
#     # Make predictions and evalute
#     model_pred = model.predict(X_test)
#     model_mae = mae(y_test, model_pred)
    
#     # Return the performance metric
#     return model_mae


#%%
lr = LinearRegression()
lr_mae = fit_and_evaluate(lr)

print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)
#$Linear Regression Performance on the test set: MAE = 18.8622
# Linear Regression Performance on the test set: MAE = 0.4349
# print(lr_mae.classes_)



# random_forest = RandomForestRegressor(random_state=60)
# random_forest_mae = fit_and_evaluate(random_forest)

# print('Random Forest Regression Performance on the test set: MAE = %0.4f' % random_forest_mae)
# # Random Forest Regression Performance on the test set: MAE = 0.0023
# # random_forest_mae.feature_importances_




# gradient_boosted = GradientBoostingRegressor(random_state=60)
# gradient_boosted_mae = fit_and_evaluate(gradient_boosted)

# print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)
# #Gradient Boosted Regression Performance on the test set: MAE = 0.2731


# #%%
# knn = KNeighborsRegressor(n_neighbors=5)
# knn_mae = fit_and_evaluate(knn)
# print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)



