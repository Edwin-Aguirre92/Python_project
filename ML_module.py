import numpy as np
from pandas.core.dtypes.missing import notnull
from sklearn.linear_model import LinearRegression
import pandas as pd
import time 




if __name__ == "__main__":
    ## read the dataframe
    df=pd.read_csv('Concrete_Data_Yeh.csv')

    print('The dataframe has a shape of :', df.shape)
    ### converting the dataframe into a numpy object (C object in python), also this is my trarget variable or dependent variable
    y=df.loc[:,'csMPa'].to_numpy()

    #### converting the dataframe into a numpy object because scikit learn uses numpy objects. Further, these dataframe or numpy object represents 
    ##the independent variables or features
    X=df.iloc[:,:df.shape[1]-1].to_numpy()

    print(f"The shape of the target feature is:",{y.shape}, "while the shape of the features are:", {X.shape})

    ### using scikit learn to train linear regressor algorithm

    #First you have to instantiate the LinearRegression class(basically an object of a class), then you call the fit method. 
    #The fit method basically is the method used in all the algorithm to learn from data.The first input is the independent variables, while the second input is the dependent variable

    start = time.process_time()

    LR=LinearRegression().fit(X,y)

    print('The time it took the algorithm to learn from data is ', round(time.process_time() - start,2), "seconds")

    ##After it learns this then you can call the prediction of it, using the predict method 

    #getting the first row with all the columns as a data sample
    data_sample=X[0,:]

    predictions_linear_regression=LR.predict(data_sample.reshape(1,-1))

    true_value = y[0]
    ##You can observe that his value is is not good because the distribution of the variable is not normal
    print("The true values is:", true_value, "the prediction values based on the linear regression algorithm:",predictions_linear_regression )


    ## This is not a good correlation value because the dependent variable is not normally distributed 
    print('The R^2 value is:', LR.score(X,y))


