import matplotlib.pyplot as plt
import seaborn as sns
from preparation import data_cleaning, remove_outliers
from dython.nominal import associations

data_cleaning()

# CORR MATRIX
def corr_matrix():
  associations_df = associations(df_train, nominal_columns='all', plot=False)
  corr_matrix = associations_df['corr']
  plt.figure(figsize=(20, 8))
  plt.gcf().set_facecolor('#FFFDD0') 
  sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
  plt.title('Correlation Matrix including Categorical Features')
  plt.show()
  
  x = df_train[:1000]
  sns.pairplot(x[['price', 'milage', 'model_year', 'engine', 'fuel_type']], hue='fuel_type')
  plt.suptitle('Pair Plot of Selected Features Colored by Fuel Type', y=1.02)
  plt.show()

# BRAND AVG PRICE
def brand_price():
  plt.figure(figsize=(12, 6))
  sns.barplot(x='brand', y='price', data=df_train[:10000], errorbar=None)
  plt.title('Average Price by Car Brand')
  plt.xlabel('Brand')
  plt.ylabel('Average Price')
  plt.xticks(rotation=90)  
  plt.show()

# HISTOGRAM MILES
def histo_miles():
  plt.figure(figsize=(10, 6))
  plt.hist(df_train['milage'], bins=30, edgecolor='black')
  plt.title('Histograma de kilometraje')
  plt.xlabel('Kilómetros')
  plt.ylabel('Frecuencia')
  plt.grid(True)
  plt.show()

# TRAMSMITION AVG PRICE
def transmition_price():
  plt.figure(figsize=(14, 6))
  sns.barplot(x='transmission', y='price', data=df_train, errorbar=None)
  plt.title('Average Price by Transmission Type')
  plt.xlabel('Transmission')
  plt.ylabel('Average Price')
  plt.xticks(rotation=45)
  plt.show()

# COUNTPLOT FUEL_TYPE
def countplot_fuel():
  plt.figure(figsize=(10, 6))
  sns.countplot(x='fuel_type', data=df_train)
  plt.title('Count of Cars by Fuel Type')
  plt.xlabel('Fuel Type')
  plt.ylabel('Count')
  plt.xticks(rotation=45)
  plt.show()

# COUNTPLOT TRANSMITION
def countplot_transmition():
  plt.figure(figsize=(10, 6))
  sns.countplot(x='transmission', data=df_train)
  plt.title('Count of Cars by Transmission Type')
  plt.xlabel('Transmission')
  plt.ylabel('Count')
  plt.xticks(rotation=90)
  plt.show()

# HISTOGRAM ACCIDENT
def histo_accident():
  plt.figure(figsize=(10, 6))
  sns.barplot(x='accident', y='price', data=df_train, errorbar=None)
  plt.title('Average Price by Accident History')
  plt.xlabel('Accident History')
  plt.ylabel('Average Price')
  plt.xticks(rotation=45)
  plt.show()


