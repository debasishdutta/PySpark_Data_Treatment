# PySpark_Data_Treatment
Outlier &amp; Missing Value Treatment In Hadoop Big Data Environment

Disclaimer: This code snippet is developed with an intension for generic use only. I hereby declare that the code was not part of any of my professional development work. Also, no sensitive or confidential data sources are used for this development.

Description: The repository consists of below list of missing value and outlier treatments for predictive model development:

	1. Outlier Treatment : Capping & Flooring, IQR
	2. Missing Value Treatment: Mean Imputation, Median Imputation

Note:
1. This scripts taking raw csv faile as a input source, however Hive integration is possible with very minimal changes 
2. User Input Section allows to set all parameters as well as data source specification. Users dont need to edit anything except this    section. This is a fully automated end to end code
3. Final output is outlier treated and missing imputeted data frame

Compatibility: The code is developed and tested on Zeppelin Notebook in Hadoop Big Data Environment using Python 3.7
