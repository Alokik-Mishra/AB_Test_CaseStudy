#!/usr/bin/env/ python
# Case Study - Alokik Mishra
# Supplmentary .py file
# 4/09/2018


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import statsmodels.discrete.discrete_model as sm
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

class ABTest(object):
    
    """
    The purpose of the object is to work through the 
    case study
    """
    
    def __init__(self, name, raw_data):
        self.name = name
        self.raw_data = raw_data
        self.status = "Class initiated"
        self.data = None
        self.long_df = None
        self.numeric_df = None      
        
        
    def read(self):
        """
        Currently only reads excel files
        Creates the data method which presents the data as 
        a pandas dataframe
        """
        self.data = pd.read_excel(self.raw_data)
        self.status += " & Data read as pandas df"
    
    
    def clean_date(self, date_var, yearfirst = True, new_name = 'Date_int'):
        """
        Formats the date variable, and creates a alternative
        integer date variable
        
        Inputs: 
            date_var - specifies the date column
            yearfirst (optional) - default is True specifies the formatting for the date
            new_name (optional) - default is 'Date_int', name of interger version of
                    the date variable.
        
        Updates the dataframe in the date method.
        """
        data = self.data
        data[date_var] = pd.to_datetime(data[date_var], yearfirst = yearfirst)
        data[new_name] = ((data[date_var] - data[date_var].min()).dt.days)
        self.data = data
        self.status += " & date converted to int called :" + new_name
        
        
    def reshape_long(self, value_cols, id_names = None, value_names = None, make_dummy = False, val_if_true = None):
        """
        Reshapes the dataframe using the id and value 
        column positions given.
        
        Input:  
            value_cols = Value colunms (the ones to be 'stacked')
            id_names (optional) = the index var new name
            val_names (optional) = the value new name
            make_dummy (optional) = converts the stacked variable into a dummy 
                                    (must specify val_if_true)
            val_if_true (optional) = the value used as boolean for dummy, 
                                     must be specified if make_dummy = True
            
        Creates long data frame accessible from the .long_df method 
        """
        df = self.data
        df_val = [i for i in list(df) if i in value_cols]
        df_id = [i for i in list(df) if i not in df_val]

        long_df = pd.melt(df, id_vars = df_id, value_vars = df_val, var_name = id_names,
                         value_name = value_names)
        
        if make_dummy:
            long_df[id_names] = long_df[id_names] == val_if_true
            
        self.long_df = long_df
        self.status += " & long version of the dataframe can be called through .long_df"
    
    
    def make_weekend(self, date_var, name = "Weekend"):
        """
        Creates a dummy variable equal to one if its the weekend
        
        Inputs:
            date_var = the variable used to determine date 
                        (must be in a dt compatible format)
            name (optional) = A name for the new weekend variable (default is 'Weekend') 
        """
        df = self.long_df
        df[name] = ((pd.DatetimeIndex(df[date_var]).dayofweek) // 5 == 1).astype(float)
        self.long_df = df
        self.status += " & weekend variable has been added with the name: " + name
    
    
    def exclude_include(self, df, var_name, var_value, 
                        exclude = True, drop_cols = None):
        """
        Changes data frame, byt either dropping rows based on 
        a conditon and dropping columns if needed
        
        Inputs:
            df = a pandas dataframe
            var_name = the columns based on which to drop rows
            var_value = the value on which to either keep or drop rows
            exclude (optional) = by default drops all rows that mee the criteria
                                if changed to False, will keep only those rows
            drop_cols (optional) = A list of column names to drop
        Output:
            new_df = A new dataframe with the changes specified
        """
        if exclude:
            new_df = df.loc[df[var_name] != var_value, :]
        else:
            new_df = df.loc[df[var_name] == var_value, :]        
        if drop_cols is not None:
            new_df = new_df.drop(drop_cols, axis = 1)
            
        return new_df
    
    
    def bar_graph(self, x, y, hue, axis_labels = None, title = None, 
                  save_name = None, fig_size = None):
        """
        Creates condtional bar plots
        
        Inputs:
            x = Categorical variable for the x-axis
            y = Continous or ordinal var for the y-axis
            hue = The conditional variable
            axis_labels (optional) = [X-axis label, y-axis label]
            title (optional) = string with plot title
            save_name (optional) = A string which is used as the filename to save the plot
                                    (if not specified the plot is not saved)
            fig_size (optional) = a tuple with the figure size for notebooks
        Shows the plot
        """
        df = self.long_df.copy()        
        df[x] = df[x].astype('category')
        if fig_size is not None:
            plt.figure(figsize=fig_size)
        ax = sns.barplot(x=x, y=y, hue = hue, data=df)
        if axis_labels is not None:
            ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1])
        if title is not None:
            ax.set_title(title)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()    
        
        
    def make_df_numeric(self, use_long = True):
        """
        Takes a dataframe and converts all non-numeric (int, categorical, float)
        variables into dummies.
        Input:
            use_long (optional) = Default is true. Uses the .long_df if true, 
                                    or the .data df if false
                                    
        The new dataframe can be called through the .numeric_df method
        """
        if use_long:
            df = self.long_df
        else:
            df = self.data
        non_numerics = list(df.select_dtypes(include=['object']))
        dummy_df = pd.get_dummies(df[non_numerics])
        numeric_only = df.select_dtypes(exclude=['object'])
        numeric_df = numeric_only.join(dummy_df)
        bools = list(numeric_df.select_dtypes(include=['bool']))
        if len(bools) > 0:
            for i in bools:
                numeric_df[i] = numeric_df[i].astype(int)
        self.numeric_df = numeric_df
        self.status += " & numeric-only version of the dataframe can be called through .numeric_df"
    
    
    def calc_rates(self, calc_name, value_var, grouping_vars, 
                   constraint_var = None, constraint_val = None,
                   use_numeric = True):
        """
        Creates tables with aggregated summation computations based on
        certain conditonals.
        
        Inputs:
            calc_name = string with the name of the resulting computed value
            value_var = string with the name of the column with raw values
            grouping_vars = list of vars to group the value_var by. The last 
                            variable becomes the column index, and all preceeding it
                            are rpw indices.
            constraint_var (optional) = an optional list of varaible names to use if the
                            to constrain the summations to certain conditions
                            (must specify constraint_val list if used)
            constraint_val (optional) = the list of vlues to constrain the summations,
                            each corresponding to the index of the variable name in the 
                            constraint_var list
            use_numeric (optional) = default is true, uses the .numeric_df as the input,
                             if False, uses the .long_df
        Output:
            step2: A pandas dataframe of the groupby table created
        """        
        if use_numeric:
            df = self.numeric_df.copy()
        else:
            df = self.long_df.copy()
        
        if constraint_var is not None:
            val_counter = 0
            for i in constraint_var:
                df = df.loc[df[i] == constraint_val[val_counter], :]
                val_counter += 1
        
        unstack_var = grouping_vars[-1]
        step1 = df.groupby(grouping_vars)[value_var].sum().unstack(unstack_var)
        step2 = pd.DataFrame(step1)
        step2[calc_name] = step2.iloc[:, 1] / (step2.iloc[:, 1] + step2.iloc[:, 0])
        
        return step2
    
    
    def timeplot(self, data, col_key, lab_key, x, y, split_var,
                 x_lab = None,
                 y_lab = None, title = None, save_name = None):
        """
        Used for plotting the variables across time
        
        Inputs:
            data = a gropuby dataframe resulting from .calc_rates
            split_var = a string with the variable to conditon the plot
            col_key = dictionary with keys as the different values of the split_var
                    and values as sring with colors to split
            lab_key = dictionary with keys as the different values of the split_var
                    and values as strings with labels for the split
             x = x-axis variable
             y = y-axis variable
             x_lab (optional) = string with x-axis label
             y_lab (optional) = string with y-axis label
             title (optional) = string with plot title
             save_name (optional) = a string with the file name to save the plot,
                     if not specified, the plot is not saved.
        
        Shows the plot
        """
        col_dict = col_key
        lab_dict = lab_key
        fig,ax = plt.subplots()
        for key,grp in data.groupby([split_var]):
            ax = grp.plot(ax = ax, kind = 'line', x = x, y = y, c = col_dict[key], label = lab_dict[key])
        plt.legend(loc='best')
        if title is not None:
            plt.title(title)
        if x_lab is not None:
            plt.xlabel(x_lab)
        if y_lab is not None:
            plt.ylabel(y_lab)
        if save_name is not None:
            plt.savefig(save_name)
            self.status += " & a figure titled " + save_name + " has been saved."
        plt.show()
    
    
    def estplot(self, data, x, y, hue = None, col = None,
                lowess = False, axis_labels = None,
               title = None, save_name = None, custom = False, col_wrap = None,
               height = None, x_lim = None):
        """
        Creates plots with estimated fits
        
        Inputs:
            x = Categorical variable for the x-axis
            y = Continous or ordinal var for the y-axis
            hue = The conditional variable
            col = Another conditional variable for the subplots
            lowess = default is False, uses ols fit, if True uses locally
                    weighted nonparametric fit
            axis_labels (optional) = [X-axis label, y-axis label]
            title (optional) = string with plot title
            save_name (optional) = A string which is used as the filename to save the plot
                     (if not specified the plot is not saved)
            custom (optional) = default is False. If true creates customisations to 
                     the aesthetic, much specify col_wrap, height, and x_lim.
            col_wrap (optional) = how many columsn to allow for subplots
            height (optional) = height of the plots
            x_lim (optional) = tuple in the from (x min value, x max value)
        
        Shows the plot
        """
        if custom:
            ax = sns.lmplot(x = x, y = y, hue = hue, col = col,
                        lowess = lowess, data = data, col_wrap = col_wrap,
                           height = height)
            ax = ax.set(xlim=x_lim)
        else:
            ax = sns.lmplot(x = x, y = y, hue = hue, col = col,
                        lowess = lowess, data = data)
        if title is not None:
            ax.fig.suptitle(title)
            ax.fig.subplots_adjust(top = 0.9)
        if axis_labels is not None:
            ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1])
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()
        
        
        
        
class AB_analysis(ABTest):
    
    """
    An object inheriting the ABTest object
    Primarily used for the 'individual level' section
    of the analysis.
    """
  
    def __init__(self, name, raw_data):
        self.name = name
        self.raw_data = raw_data
        self.status = "Class initiated"
        self.data = None
        self.long_df = None
        self.numeric_df = None 
        self.inarrays = False
        self.X_array = None
        self.y_array = None
        
        
    def expand(self, repeat_col):
        """
        Expands each observation based on the value of a specified column.
        Must have a .numeric_df version to implement.
        
        Inputs:
            repeat_col = string with column name used to determine how many times
                    repeat each observation.
        
        Created a .expanded version of the dataframe.
        """
        df = self.numeric_df
        self.expanded = pd.DataFrame(np.repeat(df.values, df[repeat_col], axis=0), columns=df.columns)
        self.status += " & expanded version has been added, accessible with the method .expanded"
        
        
    def order_t(self, treatment_var):
        """
        Places a variable in the first column of the expanded dataframe
        
        Inputs:
            treatment_var = string with variable to regarrange into first position
        
        Updates the .expanded dataframe
        """
        t = []
        non_t = []
        features = list(self.expanded)
        for i in features:       
            if i == treatment_var:
                t.append(i)
            else:
                non_t.append(i)
        order = t + non_t
        self.expanded = self.expanded.loc[:,order]
        self.treament_ordered = True
        self.status += " & the dataframe has been reordred to make " + treatment_var + " the first column"
      
    
    def make_arrays(self, y_col, drop_cols = None):
        """
        Created X (features) and y (labels) arrays from the .expanded dataframe
        
        Inputs:
            y_cols = The variable name of the label variable
            drop_cols (optional) = list of column names to drop
        
        Creates feature ndarray accessible through .X_array, label array 
        accessible through .y_array, list of feature names accessible
        through .features.
        """
        expanded_df = self.expanded
        if drop_cols is not None:
            expanded_df.drop(drop_cols, axis = 1, inplace = True)
        X = np.asarray(expanded_df.loc[:, expanded_df.columns != y_col]).astype(int)
        y = np.asarray(expanded_df[y_col]).astype(int)
        #y[y == 0] = -1
        y = y.flatten()
        self.X_array = X
        self.y_array = y
        self.features = list(expanded_df.loc[:, expanded_df.columns != y_col])
        self.inarrays = True
        self.status += " & arrays have been added with feature ndarray accessible through .X_array, label array accessible through .y_array, list of feature names accessible through .features"
    
    
    def downsample(self, majority_label, seed = 123, replace = False):
        """
        Creates a downsampled version of the data, where the majority class 
        is downsampled to the number of the minority class. Takes a random sample
        for the majority class of the same shape as the minority class and appends
        it to the minority class, to avoid predictive models overweighting the
        majority class.
        If the data has been converted to arrays using .make_arrays uses .X_array
        and .y_array, otherwise uses .expanded dataframe.
        
        Inputs:
            majority_label = the value of the majority class (o or 1)
            seed (optional) = default is 123, integer to preserve the randomisation
            replace (optional) = default is False, if true, the sampling of the majority
                    class is done with replacement.
        
        Created a .df_downsampled dataframe if .make_arrays has not been used or a 
        .df_downsampled ndarray if it has.
        """
        if self.inarrays:
            X = self.X_array
            y = self.y_array
            df = np.concatenate([X,y.reshape(y.shape[0], 1)], axis = 1)
            df_maj = df[y == majority_label]
            df_min = df[y != majority_label]


            df_maj_resample = resample(df_maj, 
                                         replace=replace,   
                                         n_samples=df_min.shape[0],     
                                         random_state=seed)
            df_downsampled = np.concatenate([df_maj_resample, df_min], axis = 0)
            self.df_downsampled = df_downsampled
        else:
            df = self.expanded
            df_majority = df[df.Purchase == majority_label]
            df_minority = df[df.Purchase != majority_label]
            df_majority_downsampled = resample(df_majority, 
                                             replace=False,
                                             n_samples=df_minority.shape[0],
                                             random_state=123) 
            df_downsampled = pd.concat([df_majority_downsampled, df_minority])
            self.df_downsampled = df_downsampled
        self.status += " & downsampled data has been added accessible through .df_downsampled"
    
    
    def make_day_indicator(self, day, date_int_var = 'Date_int', 
                           new_var_name = 'Abnormal_day', use_bootstrap = False):
        """
        Creates an indicator variable for any specified day, using either the .df_downsampled
        dataframe of the .bootstrapped dataframe. Can only be used if make_arrays has not been
        implemented.
        
        Inputs:
            day = integer with the day to make the indtor variable from, 
                    note this is not the 'Date' but the ordered number of the date
            date_int_var (optional) = default is 'Date_int', name of the variable to
                    determine the day
            new_var_name (optional) = default is 'Abnormal_day', the new indicator 
                    variables name, must be specified if this is used more than once.
            use_bootstrap (optional) = default is False. If ture uses the .bootstrapped
                    dataframe, otherwise uses .df_downsampled
        
        Updates the relevant dataframe with the new variable
        """
        if use_bootstrap:
            df = self.bootstrapped.copy()
        else:
            df = self.df_downsampled.copy()
        df[new_var_name] = 0
        df.loc[df[date_int_var] == day,new_var_name] = 1
        self.indcator_day = day
        if use_bootstrap:
            self.bootstrapped = df
        else:
            self.df_downsampled = df
        self.status += " & an indicator variable for the variable " + date_int_var + " value of " + str(day) + "has been added with the name: " + new_var_name
    
    
    def bootstrap(self, majority_label, seeds):
        """
        Resamples the downsampled dataframe using new seeds and appends.
        Only works if make_arrays has not been implemented.
        Inputs:
            majority_label = the value of the majority class (o or 1)
            seeds = a list of ineger seeds to resample. The lenght of the list 
                    determines how many time bootstrapping is done.
        Creates a .bootstrapped version of the dataframe.
        """
        main = []
        for i in seeds:
            self.downsample(majority_label = majority_label, seed = i)
            main.append(self.df_downsampled)
        combined = pd.concat(main, axis = 0)
        self.bootstrapped = combined
        self.status += " & a bootstrapped version of the data has been added accessible through the .bootstrapped method"
    
    
    '''
    NO LONGER USED (SCIKIT IS USED FOR ALL INTERACTION VARS)
    def treatment_interaction(self, y_col = None, drop_cols = None):
        if self.inarrays:
            X = self.df_downsampled[:,:-1]
            k = X.shape[1]
            X_interaction = X
            features = self.features
            for i in range(1,k):
                product = X[:,0] * X[:,i]
                X_interaction = np.concatenate([X_interaction, product.reshape(product.shape[0],1)], axis = 1)
                features.append(str("Treatment * " + features[i]))
        else:
            X = self.df_downsampled.loc[:, self.df_downsampled.columns != y_col]
            if drop_cols is not None:
                X.drop(drop_cols, axis = 1, inplace = True)
            k = X.shape[1]
            X_interaction = X
            features = list(X)
            for i in features[1:]:
                name = "Treatmentx"+str(i)
                X_interaction[name] = X_interaction["Treatment"]*X_interaction[i]
            features = list(X_interaction)                   
        return (features, X_interaction)
    '''    
    
    
    def logistic_with_interactions(self, treatment_var, interaction_vars,
                                    y_var, other_vars = None, use_bootstrapped = False):
        """
        Uses statamodels packages to do logisitic regressions on the treatment variable
        and up to two interaction terms and optionally other covariates. Uses either the downsampled
        or the bootstrapped data.
        Inputs:
              treatment_var = string with treatment var name
              interaction_vars = a list of up to two variable to interact with the treatment
              y_var = outcome variable (labels)
              other_vars = list of other covariates to add to the specification.
              use_bootstrapped = default = False, uses the .df_downsampled data, if True uses
                      .bootstrapped
        Outputs:
            The summary table with the regression results.
        """
        if use_bootstrapped:
            df = self.bootstrapped.copy()
        else:
            df = self.df_downsampled.copy()
       
        if len(interaction_vars) == 2:
            for i in range(len(interaction_vars)-1):                
                df[interaction_vars[i]+'x'+interaction_vars[i+1]] = df[interaction_vars[i]] * df[interaction_vars[i + 1]]
                interaction_vars.append(interaction_vars[i]+'x'+interaction_vars[i+1])
        treatment_ints = [treatment_var]
        for i in interaction_vars:
            df['Tx'+i] = df[treatment_var] * df[i]
            treatment_ints.append('Tx'+i)
        if other_vars is not None:    
            X = df[[*treatment_ints, *interaction_vars, *other_vars]] 
        else:
            X = df[[*treatment_ints, *interaction_vars]]
        y = df[[y_var]]
        LogitSM = sm.Logit(np.asarray(y.astype(int)),X.astype(int))
        
        return LogitSM.fit().summary()
    
    
    def fit_and_predict(self, predictor, y_col, test_size = 0.2, 
                    use_bootstrapped = True, drop_cols = None,
                    save_name = None, new_sample = None, use_interactions = False,
                    subsample = None, subsample_raw = True):
        """
        All in one function to split the data, take an optional subsample, optionally
        do feature expansion through interaction terms, train and fit the data using a scikit
        classifier on a 80% sample of the data, predict the data on a 20% testing sample,
        optionally also predicting on a new sample (can use simulated data here).
        
        Inputs:
            predictor = scikit classfier to use
            y_col = variable name of the label 
            test_size (optional) = default is 0.2, the fraction to split into the testing set
            use_bootstrapped (optional) = default is True, uses bootstrapped data, 
                    otherwise uses df_downsampled.
            drop_cols (optional) = a list of columns to drop before using the model
            save_name (optional) = string corresponding to the out file saving the
                    confusion matrix, if not specified the matrix is not saved
            new_sample (optional) = name of external data with the same structure as the
                    festures in the fit (including ommiting the 'drop_cols') on which
                    to use the weights from the fit to predict the classes
            subsample_raw (optional) = default is True, is true assumes the external sample
                    is raw in excel format, and new_sample should be string, if false, takes
                    in a dataframe, and new_sample should be name of the dataframe object.
            use_interactions (optional) = defualt is False, if true, uses interaction terms
                    as features
            subsample (optional) = a decimal value corresponding to the fraction of the data
                   to use for the predictive model. This is useful if the model is farily complex
                   or if using the use_interaction option, to limit the computational complexity.
        
        Outputs:
            A tuple with the following order:
            [0] = predictions on the testing set
            [1] = a list of feature names (pre interaction terms)
            [2] = accuracy score on the testing set
            [3] = dataframe with the new sample including predicted classes 
                    (only active if using new_sample)                 
        """
        if use_bootstrapped:
            df = self.bootstrapped
        else:
            df = self.df_downsampled
        if subsample is not None:
            df = df.sample(frac = subsample, replace = False, axis = 0)
        if drop_cols is not None:
            df = df.drop(drop_cols, axis = 1)
        X = df.loc[:,df.columns != y_col].astype(int)
        features = list(X)
        if use_interactions:
            X = PolynomialFeatures(degree=3, interaction_only=True).fit_transform(X)
        y = np.asarray(df.loc[:,df.columns == y_col]).astype(int)
        y = y.reshape(y.shape[0])
        x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                            test_size = test_size)
        model = predictor.fit(x_train, y_train)
        predictions = model.predict(x_test)
        score = model.score(x_test, y_test)
        conf = metrics.confusion_matrix(y_test, predictions)
        
        new_sample_predicted = None
        if new_sample is not None:
            if subsample_raw:
                new = pd.read_excel(new_sample)
                new_orig = new.copy()
                if use_interactions:
                    new = PolynomialFeatures(degree=3, interaction_only=True).fit_transform(new)
                predictions_newsample = model.predict(new)
                predictions_newsample = pd.DataFrame({'Predicted':predictions_newsample.flatten()})
                new_sample_predicted = pd.concat([new_orig, predictions_newsample], axis = 1)
            else:
                new_orig = new_sample.copy()
                if use_interactions:
                    new_sample = PolynomialFeatures(degree=3, interaction_only=True).fit_transform(new_sample)
                predictions_newsample = model.predict(new_sample)
                predictions_newsample = pd.DataFrame({'Predicted':predictions_newsample.flatten()})
                new_sample_predicted = pd.concat([new_orig, predictions_newsample], axis = 1)
        
        plt.figure(figsize=(9,9))
        sns.heatmap(conf, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size = 15)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()
        
        return (predictions, features, score, new_sample_predicted)