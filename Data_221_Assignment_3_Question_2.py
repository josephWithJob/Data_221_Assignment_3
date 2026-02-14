# Assignment 3 Question 1
# Joseph Krosel

# FIX THE LABEL NAMES
import pandas
import matplotlib.pyplot as plt

# Gets the csv's
data_from_crime_file = pandas.read_csv('crime.csv')

# Gets column
violent_crimes_per_pop_columns_data = data_from_crime_file['ViolentCrimesPerPop'].tolist()

# Histogram
plt.hist(violent_crimes_per_pop_columns_data, bins=20, edgecolor="white")
plt.title("Histogram of Violent Crimes")
plt.xlabel("Rates of Violent Crimes Per Pop")
plt.ylabel("Frequency of Violent Crimes")
plt.show()

# Boxplot
plt.boxplot(violent_crimes_per_pop_columns_data)
plt.xlabel("Rates of Violent Crimes Per Pop")
plt.ylabel("Frequency of Violent Crimes")
plt.show()

# Need 5-7 sentences
# Question 1: What the histogram shows about how the data values are spread
# Answer: Judging on the size of the bars in the histogram, the data is
# generally right skewed. As the first four bins reach above 30 in frequency,
# while the other six do not. Columns 5-7 stay at 30, and the rest drop bellow
# reaching a minimum of 14.

# Question 2: What the box plot shows about the median
# Answer: The box plot shows the median is at 0.4 from the orange
# horizontal lines spanning the width of the box.

# Question 3: Whether the box plot suggests the presence of outliers
# Answer: The box plot shows outliers because of the far stretched
# whiskers. Revealing they have data far outside the IQR.