# Assignment 3 Question 1
# Joseph Krosel

import pandas
import numpy

# Gets the csv
data_from_crime_file = pandas.read_csv('crime.csv')

# Gets column
violent_crimes_per_pop_columns_data = data_from_crime_file['ViolentCrimesPerPop'].tolist()

# Calculates information
mean_of_column = numpy.mean(violent_crimes_per_pop_columns_data)
median_of_column = numpy.median(violent_crimes_per_pop_columns_data)
standard_deviation_of_column = numpy.std(violent_crimes_per_pop_columns_data)
minimum_of_column = numpy.min(violent_crimes_per_pop_columns_data)
maximum_of_column = numpy.max(violent_crimes_per_pop_columns_data)

print(mean_of_column,
      median_of_column,
      standard_deviation_of_column,
      minimum_of_column,
      maximum_of_column)

# Question 1: Compare the mean and median. Does the
# distribution look symmetric or skewed? Explain briefly

# Answer:The distribution looks skewed because the mean
# and median are not the same values

# Question 2: If there are extreme values (very large or very small),
# which statistic is more affected: mean or median? Explain why.

# Answer: Yes there are extreme values because the max and min
# are at opposite ends of the scale. With the minimum being 0.02
# and the max at 1. This means the mean is more effected because
# it uses each magnitude of all data points allowing extreme values
# to skew averages