import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. LOAD DATA & INITIAL INSPECTION
# ------------------------------------------------------------
 
# Load the raw dataset into a DataFrame
df = pd.read_csv('data/stud.csv')
 
# Preview the first 5 rows to get a feel for the columns and values
# print(df.head())
 
# Check the dataset's dimensions (rows, columns)
# print(df.shape)
 
# Count missing values per column — helps decide if imputation is needed
# print(df.isna().sum())
 
# Count fully duplicated rows — duplicates can bias the analysis/model
# print(df.duplicated().sum())
 
# Check Null and Dtypes
# Quick summary of column dtypes, non-null counts, and memory usage
# print(df.info())
 
# Number of unique values per column — useful to spot categorical vs
# continuous features and to sanity-check category cardinality
# print(df.nunique())
 
# Standard descriptive statistics (mean, std, min, max, quartiles)
# for all numeric columns
# print(df.describe())
 
# --------------------------------------------------------
# Inspect the unique categories within each categorical feature
# to confirm there are no unexpected/typo'd category labels
# --------------------------------------------------------
 
# print("Categories in 'gender' variable:     ", end=" ")
# print(df['gender'].unique())
 
# print("Categories in 'race_ethnicity' variable:  ", end=" ")
# print(df['race_ethnicity'].unique())
 
# print("Categories in'parental level of education' variable:", end=" ")
# print(df['parental_level_of_education'].unique())
 
# print("Categories in 'lunch' variable:     ", end=" ")
# print(df['lunch'].unique())
 
# print("Categories in 'test preparation course' variable:     ", end=" ")
# print(df['test_preparation_course'].unique())
 
 
# # ------------------------------------------------------------
# # 2. SEPARATE NUMERICAL & CATEGORICAL FEATURES
# # ------------------------------------------------------------
 
# define numerical & categorical columns
# Any column with dtype 'O' (object) is treated as categorical;
# everything else (int/float) is treated as numerical
# numeric_features = [feature for feature in df.columns if df[feature].dtype != 'str']
# categorical_features = [feature for feature in df.columns if df[feature].dtype == 'str']
 
# print columns
# print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
# print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))
 
# Quick peek at the first 2 rows again before feature engineering
# print(df.head(2))
 
 
# ------------------------------------------------------------
# 3. FEATURE ENGINEERING — TOTAL & AVERAGE SCORE
# ------------------------------------------------------------
 
# Sum the three subject scores into a single 'total score' column
df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
 
# Convert the total into a per-subject average (0-100 scale)
df['average'] = df['total score'] / 3
 
# print(df.head())
 
# --------------------------------------------------------
# Count how many students achieved a perfect (100) score
# and how many scored very poorly (<= 20) in each subject
# --------------------------------------------------------
 
# reading_full = df[df['reading_score'] == 100]['average'].count()
# writing_full = df[df['writing_score'] == 100]['average'].count()
# math_full = df[df['math_score'] == 100]['average'].count()
 
# print(f'Number of students with full marks in Maths: {math_full}')
# print(f'Number of students with full marks in Writing: {writing_full}')
# print(f'Number of students with full marks in Reading: {reading_full}')
 
# reading_less_20 = df[df['reading_score'] <= 20]['average'].count()
# writing_less_20 = df[df['writing_score'] <= 20]['average'].count()
# math_less_20 = df[df['math_score'] <= 20]['average'].count()
 
# print(f'Number of students with less than 20 marks in Maths: {math_less_20}')
# print(f'Number of students with less than 20 marks in Writing: {writing_less_20}')
# print(f'Number of students with less than 20 marks in Reading: {reading_less_20}')
 
 
# ------------------------------------------------------------
# 4. UNIVARIATE DISTRIBUTION — AVERAGE & TOTAL SCORE
# ------------------------------------------------------------
 
# Distribution of the 'average' score overall, and split by gender,
# to check for normality/skew and any gender-based gap
# fig, axs = plt.subplots(1, 2, figsize=(15, 7))
# plt.subplot(121)
# sns.histplot(data=df, x='average', bins=30, kde=True, color='g')
# plt.subplot(122)
# sns.histplot(data=df, x='average', kde=True, hue='gender')
# plt.show()
 
# Same idea, but for the raw 'total score' (sum of all 3 subjects)
# fig, axs = plt.subplots(1, 2, figsize=(15, 7))
# plt.subplot(121)
# sns.histplot(data=df, x='total score', bins=30, kde=True, color='g')
# plt.subplot(122)
# sns.histplot(data=df, x='total score', kde=True, hue='gender')
# plt.show()
 
 
# ------------------------------------------------------------
# 5. AVERAGE SCORE vs LUNCH TYPE (overall, and by gender)
# ------------------------------------------------------------
 
# Does having standard vs. free/reduced lunch correlate with
# performance? Shown overall, then separately for each gender.
# plt.subplots(1, 3, figsize=(25, 6))
# plt.subplot(141)
# sns.histplot(data=df, x='average', kde=True, hue='lunch')
# plt.subplot(142)
# sns.histplot(data=df[df.gender == 'female'], x='average', kde=True, hue='lunch')
# plt.subplot(143)
# sns.histplot(data=df[df.gender == 'male'], x='average', kde=True, hue='lunch')
# plt.show()
 
 
# ------------------------------------------------------------
# 6. AVERAGE SCORE vs PARENTAL LEVEL OF EDUCATION
# ------------------------------------------------------------
 
# Does parental education level relate to student performance?
# Shown overall, then split by gender.
# plt.subplots(1, 3, figsize=(25, 6))
# plt.subplot(141)
# ax = sns.histplot(data=df, x='average', kde=True, hue='parental_level_of_education')
# plt.subplot(142)
# ax = sns.histplot(data=df[df.gender == 'male'], x='average', kde=True, hue='parental_level_of_education')
# plt.subplot(143)
# ax = sns.histplot(data=df[df.gender == 'female'], x='average', kde=True, hue='parental_level_of_education')
# plt.show()
 

# ------------------------------------------------------------
# 7. AVERAGE SCORE vs RACE/ETHNICITY
# ------------------------------------------------------------
 
# Distribution of average score across race/ethnicity groups,
# again overall and split by gender
# plt.subplots(1, 3, figsize=(25, 6))
# plt.subplot(141)
# ax = sns.histplot(data=df, x='average', kde=True, hue='race_ethnicity')
# plt.subplot(142)
# ax = sns.histplot(data=df[df.gender == 'female'], x='average', kde=True, hue='race_ethnicity')
# plt.subplot(143)
# ax = sns.histplot(data=df[df.gender == 'male'], x='average', kde=True, hue='race_ethnicity')
# plt.show()
 
 
# ------------------------------------------------------------
# 8. SCORE SPREAD PER SUBJECT — VIOLIN PLOTS
# ------------------------------------------------------------
 
# Violin plots show the full distribution shape (median, quartiles,
# density) for each subject score independently
# plt.figure(figsize=(18, 8))
# plt.subplot(1, 4, 1)
# plt.title('MATH SCORES')
# sns.violinplot(y='math_score', data=df, color='red', linewidth=3)
# plt.subplot(1, 4, 2)
# plt.title('READING SCORES')
# sns.violinplot(y='reading_score', data=df, color='green', linewidth=3)
# plt.subplot(1, 4, 3)
# plt.title('WRITING SCORES')
# sns.violinplot(y='writing_score', data=df, color='blue', linewidth=3)
# plt.show()
 
 
# ------------------------------------------------------------
# 9. CATEGORICAL FEATURE COMPOSITION — PIE CHARTS
# ------------------------------------------------------------
 
# One pie chart per categorical feature, showing the proportion
# of students in each category (gender, race/ethnicity, lunch,
# test preparation course, parental education)
# plt.rcParams['figure.figsize'] = (30, 12)
 
# plt.subplot(1, 5, 1)
# size = df['gender'].value_counts()
# labels = 'Female', 'Male'
# color = ['red', 'green']
 
# plt.pie(size, colors=color, labels=labels, autopct='.%2f%%')
# plt.title('Gender', fontsize=20)
# plt.axis('off')
 
# plt.subplot(1, 5, 2)
# size = df['race_ethnicity'].value_counts()
# labels = 'Group C', 'Group D', 'Group B', 'Group E', 'Group A'
# color = ['red', 'green', 'blue', 'cyan', 'orange']
 
# plt.pie(size, colors=color, labels=labels, autopct='.%2f%%')
# plt.title('Race_Ethnicity', fontsize=20)
# plt.axis('off')
 
# plt.subplot(1, 5, 3)
# size = df['lunch'].value_counts()
# labels = 'Standard', 'Free'
# color = ['red', 'green']
 
# plt.pie(size, colors=color, labels=labels, autopct='.%2f%%')
# plt.title('Lunch', fontsize=20)
# plt.axis('off')
 
# plt.subplot(1, 5, 4)
# size = df['test_preparation_course'].value_counts()
# labels = 'None', 'Completed'
# color = ['red', 'green']
 
# plt.pie(size, colors=color, labels=labels, autopct='.%2f%%')
# plt.title('Test Course', fontsize=20)
# plt.axis('off')
 
# plt.subplot(1, 5, 5)
# size = df['parental_level_of_education'].value_counts()
# labels = 'Some College', "Associate's Degree", 'High School', 'Some High School', "Bachelor's Degree", "Master's Degree"
# color = ['red', 'green', 'blue', 'cyan', 'orange', 'grey']
 
# plt.pie(size, colors=color, labels=labels, autopct='.%2f%%')
# plt.title('Parental Education', fontsize=20)
# plt.axis('off')
 
# plt.tight_layout()
# plt.grid()
# plt.show()
 
 
# ------------------------------------------------------------
# 10. GENDER — COUNT PLOT + PIE CHART
# ------------------------------------------------------------
 
# Bar chart of raw gender counts (with count labels on top of bars),
# followed by a pie chart of the same distribution as a percentage
# f, ax = plt.subplots(1, 2, figsize=(20, 10))
# sns.countplot(x=df['gender'], data=df, palette='bright', ax=ax[0], saturation=0.95)
# for container in ax[0].containers:
#     ax[0].bar_label(container, color='black', size=20)
 
# plt.pie(x=df['gender'].value_counts(), labels=['Male', 'Female'], explode=[0, 0.1], autopct='%1.1f%%', shadow=True, colors=['#ff4d4d', '#ff8000'])
# plt.show()
 
 
# ------------------------------------------------------------
# 11. AVERAGE PERFORMANCE BY GENDER — GROUPED BAR CHART
# ------------------------------------------------------------
 
# Compute the mean of every numeric column, grouped by gender
# gender_group = df.groupby('gender').mean(numeric_only=True)
# plt.figure(figsize=(10, 8))
 
# X = ['Total Average', 'Math Average']
 
# # Index 0/1 assumes 'female' sorts before 'male' alphabetically
# # in the groupby result — double check this ordering matches
# # gender_group.index before relying on it
# female_scores = [
#     gender_group.loc["female", "average"],
#     gender_group.loc["female", "math_score"]
# ]

# male_scores = [
#     gender_group.loc["male", "average"],
#     gender_group.loc["male", "math_score"]
# ]
 
# X_axis = np.arange(len(X))
 
# # Side-by-side bars comparing male vs. female averages
# plt.bar(X_axis - 0.2, male_scores, 0.4, label='Male')
# plt.bar(X_axis + 0.2, female_scores, 0.4, label='Female')
 
# plt.xticks(X_axis, X)
# plt.ylabel("Marks")
# plt.title("Total average v/s Math average marks of both the genders", fontweight='bold')
# plt.legend()
# plt.show()
 
 
# ------------------------------------------------------------
# 12. RACE/ETHNICITY — COUNT PLOT + PIE CHART
# ------------------------------------------------------------
 
# Same count + pie chart pattern as gender, but for race/ethnicity groups
# f, ax = plt.subplots(1, 2, figsize=(20, 10))
# sns.countplot(x=df['race_ethnicity'], data=df, palette='bright', ax=ax[0], saturation=0.95)
# for container in ax[0].containers:
#     ax[0].bar_label(container, color='black', size=20)
 
# plt.pie(x=df['race_ethnicity'].value_counts(), labels=df['race_ethnicity'].value_counts().index, explode=[0.1, 0, 0, 0, 0], autopct='%1.1f%%', shadow=True)
# plt.show()
 
 
# ------------------------------------------------------------
# 13. MEAN SCORE PER SUBJECT, GROUPED BY RACE/ETHNICITY
# ------------------------------------------------------------
 
# For each race/ethnicity group, compare the mean math, reading,
# and writing scores side by side in three separate bar charts
# Group_data2 = df.groupby('race_ethnicity')
# f, ax = plt.subplots(1, 3, figsize=(20, 8))
# sns.barplot(x=Group_data2['math_score'].mean().index, y=Group_data2['math_score'].mean().values, palette='mako', ax=ax[0])
# ax[0].set_title('Math_score', color='#005ce6', size=20)
 
# for container in ax[0].containers:
#     ax[0].bar_label(container, color='black', size=15)
 
# sns.barplot(x=Group_data2['reading_score'].mean().index, y=Group_data2['reading_score'].mean().values, palette='flare', ax=ax[1])
# ax[1].set_title('Reading_score', color='#005ce6', size=20)
 
# for container in ax[1].containers:
#     ax[1].bar_label(container, color='black', size=15)
 
# sns.barplot(x=Group_data2['writing_score'].mean().index, y=Group_data2['writing_score'].mean().values, palette='coolwarm', ax=ax[2])
# ax[2].set_title('Writing_score', color='#005ce6', size=20)
 
# for container in ax[2].containers:
#     ax[2].bar_label(container, color='black', size=15)
 
 
# # ------------------------------------------------------------
# # 14. PARENTAL LEVEL OF EDUCATION — DISTRIBUTION & MEAN SCORES
# # ------------------------------------------------------------
 
# # How many students fall into each parental education category
# plt.rcParams['figure.figsize'] = (15, 9)
# plt.style.use('fivethirtyeight')
# sns.countplot(df['parental_level_of_education'], palette='Blues')
# plt.title('Comparison of Parental Education', fontweight=30, fontsize=20)
# plt.xlabel('Degree')
# plt.ylabel('count')
# plt.show()
 
# # Mean of every numeric score column, grouped by parental education,
# # shown as a horizontal bar chart for easy comparison
# df.groupby('parental level of education').agg('mean').plot(kind='barh', figsize=(10, 10))
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()
plt.savefig("reports/PARENTAL LEVEL OF EDUCATION — DISTRIBUTION & MEAN SCORES.png")
 
 
# ------------------------------------------------------------
# 15. LUNCH TYPE — DISTRIBUTION
# ------------------------------------------------------------
 
# # How many students have standard vs. free/reduced lunch
# plt.rcParams['figure.figsize'] = (15, 9)
# sns.countplot(df['lunch'], palette='PuBu')
# plt.title('Comparison of different types of lunch', fontweight="bold", fontsize=20)
# plt.xlabel('types of lunch')
# plt.ylabel('count')
# plt.show()
 
 
# ------------------------------------------------------------
# 16. PARENTAL EDUCATION vs TEST PREP / LUNCH — COUNT PLOTS
# ------------------------------------------------------------
 
# # Left: for each parental education level, how many students did/
# # didn't complete the test preparation course
# # Right: for each parental education level, how many have standard
# # vs. free/reduced lunch
# f, ax = plt.subplots(1, 2, figsize=(20, 8))
# sns.countplot(x=df['parental_level_of_education'], data=df, palette='bright', hue='test_preparation_course', saturation=0.95, ax=ax[0])
# ax[0].set_title('Students vs test preparation course ', color='black', size=25)
# for container in ax[0].containers:
#     ax[0].bar_label(container, color='black', size=20)
 
# sns.countplot(x=df['parental_level_of_education'], data=df, palette='bright', hue='lunch', saturation=0.95, ax=ax[1])
# for container in ax[1].containers:
#     ax[1].bar_label(container, color='black', size=20)
 
 
# # ------------------------------------------------------------
# # 17. SUBJECT SCORES vs LUNCH, SPLIT BY TEST PREPARATION COURSE
# # ------------------------------------------------------------
 
# # For each subject, compare average score across lunch types,
# # further split by whether the student completed test prep —
# # checks whether test prep narrows or widens the lunch-based gap
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 2, 1)
# sns.barplot(x=df['lunch'], y=df['math_score'], hue=df['test_preparation_course'])
# plt.subplot(2, 2, 2)
# sns.barplot(x=df['lunch'], y=df['reading_score'], hue=df['test_preparation_course'])
# plt.subplot(2, 2, 3)
# sns.barplot(x=df['lunch'], y=df['writing_score'], hue=df['test_preparation_course'])
 
 
# # ------------------------------------------------------------
# # 18. OUTLIER DETECTION — BOX PLOTS
# # ------------------------------------------------------------
 
# Box plots for each score column to visually spot outliers
# (points outside the whiskers) and compare spread/median
# plt.subplots(1, 4, figsize=(16, 5))
# plt.subplot(141)
# sns.boxplot(df['math_score'], color='skyblue')
# plt.subplot(142)
# sns.boxplot(df['reading_score'], color='hotpink')
# plt.subplot(143)
# sns.boxplot(df['writing_score'], color='yellow')
# plt.subplot(144)
# sns.boxplot(df['average'], color='lightgreen')
# plt.show()
 
 
# ------------------------------------------------------------
# 19. MULTIVARIATE RELATIONSHIPS — PAIR PLOT
# ------------------------------------------------------------
 
# Grid of scatter plots (and diagonal distributions) for every
# pair of numeric features, colored by gender — useful for
# spotting correlations and clusters at a glance
# sns.pairplot(df, hue='gender')
# plt.show()