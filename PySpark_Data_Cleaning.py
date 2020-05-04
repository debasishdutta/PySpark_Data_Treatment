#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### USER INPUT SECTION ###

# Global Variables #
global_source_name = "C:/Users/debas/Downloads/Python Code Library/PySpark Model Codes/Datasets/Outlier_Missing_Value_Treatment_Dataset.csv" 
global_source_format = "csv"
global_treatment_var_list = ["Account_Length",
"Area_Code",
"International_Plan",
"Voice_Mail_Plan",
"Number_Vmail_Messages",
"Total_Day_Minutes",
"Total_Day_Calls",
"Total_Day_Charge",
"Total_Eve_Minutes",
"Total_Eve_Calls",
"Total_Eve_Charge",
"Total_Night_Minutes",
"Total_Night_Calls",
"Total_Night_Charge",
"Total_Intl_Minutes",
"Total_Intl_Calls",
"Total_Intl_Charge",
"Customer_Service_Calls"]
global_outlier_capping_pct = 0.99
global_outlier_flooring_pct = 0.01
global_outlier_iqr_multiplier = 1.5


# In[ ]:


### ENVIORNMENT SET UP ###

# Initialize PySpark Engine #
import findspark
findspark.init()
    
# Initiate A Spark Session On Local Machine With 4 Physical Cores #
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ML_Outlier_MissingValue_Pyspark_V1').master('local[4]').getOrCreate()


# In[ ]:


### RAW DATA IMPORT ###

from pyspark.sql.types import *
from time import *

def data_import(source_name, source_format):
    
    import_start_time = time()
    
    print("\nSpark Session Initiated Successfully. Kindly Follow The Log For Further Output\n")
    
    df = spark.read.format(source_format).option("header","true").option("inferSchema","true").load(source_name)
    
    import_end_time = time()
    import_elapsed_time = (import_end_time - import_start_time)/60
    print("\nTime To Perform Data Import: %.3f Minutes\n" % import_elapsed_time)
    
    return(df)


# In[ ]:


### USER DEFINED FUNCTION: OUTLIER TREATMENT USING CAPPING & FLOORING ###

from pyspark.sql.functions import when
from time import *

def outlier_capping_flooring(df,var_list,flooring_pct,capping_pct):
    
    outlier_start_time = time()  
 
  # Creating A Copy of Source Data Frame
    raw_data_treated = df
  
  # Performing Outlier Treatment
    for var in var_list:
        lower_bound = raw_data_treated.approxQuantile(var,[flooring_pct],0)[0]
        upper_bound = raw_data_treated.approxQuantile(var,[capping_pct],0)[0]
        raw_data_treated = raw_data_treated.withColumn(var,
                                                       when(raw_data_treated[var] < lower_bound, lower_bound)
                                                       .otherwise(raw_data_treated[var]))
        raw_data_treated = raw_data_treated.withColumn(var,
                                                       when(raw_data_treated[var] > upper_bound, upper_bound)
                                                       .otherwise(raw_data_treated[var]))
    outlier_end_time = time()
    outlier_elapsed_time = (outlier_end_time - outlier_start_time)/60
    print("\nTime To Perform Outlier Treatment Using Capping & Flooring: %.3f Minutes\n" % outlier_elapsed_time)  
    
    return(raw_data_treated)


# In[ ]:


### USER DEFINED FUNCTION: OUTLIER TREATMENT USING IQR METHOD ###

from pyspark.sql.functions import when
from time import *

def outlier_iqr(df,var_list, multiplier):
    
    outlier_start_time = time()  
 
  # Creating A Copy of Source Data Frame
    raw_data_treated = df
  
  # Performing Outlier Treatment
    for var in var_list:
        quartile_1 = raw_data_treated.approxQuantile(var,[0.25],0)[0]
        quartile_3 = raw_data_treated.approxQuantile(var,[0.75],0)[0]
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * multiplier)
        upper_bound = quartile_3 + (iqr * multiplier)
        raw_data_treated = raw_data_treated.withColumn(var,
                                                       when(raw_data_treated[var] < lower_bound, lower_bound)
                                                       .otherwise(raw_data_treated[var]))
        raw_data_treated = raw_data_treated.withColumn(var,
                                                       when(raw_data_treated[var] > upper_bound, upper_bound)
                                                       .otherwise(raw_data_treated[var]))
    outlier_end_time = time()
    outlier_elapsed_time = (outlier_end_time - outlier_start_time)/60
    print("\nTime To Perform Outlier Treatment Using IQR Method: %.3f Minutes\n" % outlier_elapsed_time)  
    
    return(raw_data_treated)


# In[ ]:


### USER DEFINED FUNCTION: MISSING VALUE TREATMENT USING MEAN IMPUTATION ###

from pyspark.sql.functions import mean

def missing_value_mean(df, var_list):
    
    missing_value_start_time = time()  
  
  # Performing Outlier Treatment
    mean_dict = df[var_list].select(*[mean(c).alias(c) for c in df[var_list].columns]).toPandas().to_dict('r')[0]
    raw_data_treated = df.fillna(mean_dict)
    
    missing_value_end_time = time()
    missing_value_elapsed_time = (missing_value_end_time - missing_value_start_time)/60
    print("\nTime To Perform Missing Value Treatment Using Mean Imputation: %.3f Minutes\n" % missing_value_elapsed_time)  
    
    return(raw_data_treated)    


# In[ ]:


### USER DEFINED FUNCTION: MISSING VALUE TREATMENT USING MEDIAN IMPUTATION ###

import pandas as pd

def missing_value_median(df, var_list):
    
    missing_value_start_time = time()  
  
  # Performing Outlier Treatment
    temp_df = pd.DataFrame()
    final_df = pd.DataFrame()
    
    for col_name in var_list:
        temp_df.loc[0,"Column_Name"] = col_name
        temp_df.loc[0,"Column_Median"] = df.approxQuantile(col_name,[0.5],0)
        final_df = final_df.append(temp_df)
    final_df = final_df.transpose()
    final_df.columns = final_df.iloc[0]
    final_df = final_df[1:]
    final_df.index.name = None
    final_df.reset_index(drop=True, inplace=True)
    median_dict = final_df.to_dict('r')[0]
    
    raw_data_treated = df.fillna(median_dict)
        
    missing_value_end_time = time()
    missing_value_elapsed_time = (missing_value_end_time - missing_value_start_time)/60
    print("\nTime To Perform Missing Value Treatment Using Median Imputation: %.3f Minutes\n" % missing_value_elapsed_time)  
    
    return(raw_data_treated)


# In[ ]:


# Data Import #
raw_data = data_import(global_source_name, global_source_format)

# Outlier Treatment Using Capping & Flooring #
treated_data_1 = outlier_capping_flooring(raw_data,
                                        global_treatment_var_list,
                                        global_outlier_flooring_pct,
                                        global_outlier_capping_pct)

# Outlier Treatment Using IQR Method #
treated_data_2 = outlier_iqr(raw_data,
                             global_treatment_var_list,
                             global_outlier_iqr_multiplier)

# Missing Value Treatment Using Mean Imputation #
treated_data_3 = missing_value_mean(treated_data_1,
                                    global_treatment_var_list)

# Missing Value Treatment Using Median Imputation #
treated_data_4 = missing_value_median(treated_data_2,
                                      global_treatment_var_list)

