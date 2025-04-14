import pandas as pd

df_train = pd.read_csv("train.csv")
df_train.head()
df_train.isnull().sum()
columns= df_train.columns

# DATA CLEANING

def data_cleaning():
  # CLEAN TITLE
  df_train['clean_title'] = df_train['clean_title'].fillna('No')
  
  # ACCIDENT
  df_train['accident'] = df_train['accident'].replace('None reported', 'No')
  df_train['accident'] = df_train['accident'].fillna('No')
  df_train['accident'] = df_train['accident'].replace('At least 1 accident or damage reported', 'Yes')
  
  # MILAGE
  df_train = df_train.dropna(subset=['milage'])
  
  # MODEL YEAR
  df_train = df_train.dropna(subset=['model_year'])
  
  # TYPE FUEL
  df_train['fuel_type'] = df_train['fuel_type'].replace('Plug-In Hybrid', 'Hybrid')
  
  df_train = df_train.dropna(subset=['fuel_type'])
  df_train = df_train[~df_train['fuel_type'].str.contains('–', case=False, na=False)]
  df_train = df_train[~df_train['fuel_type'].str.contains('not supported', case=False, na=False)]
  
  # TRANSMISION
  df_train['transmission'] = df_train['transmission'].replace('8-SPEED AT', '8-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('8-SPEED A/T', '8-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('8-Speed Automatic', '8-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('Automatic', 'A/T')
  df_train['transmission'] = df_train['transmission'].replace('10-Speed Automatic', '10-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('6-Speed Automatic', '6-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('9-Speed Automatic', '9-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('CVT Transmission', 'CVT')
  df_train['transmission'] = df_train['transmission'].replace('1-Speed Automatic', '1-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('6-Speed Manual', '6-Speed M/T')
  df_train['transmission'] = df_train['transmission'].replace('7-Speed Automatic', '7-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('5-Speed Automatic', '5-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('7-Speed Manual', '7-Speed M/T')
  df_train['transmission'] = df_train['transmission'].replace('4-Speed Automatic', '4-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('6 Speed Mt', '6-Speed M/T')
  df_train['transmission'] = df_train['transmission'].replace('8-Speed Manual', '8-Speed M/T')
  df_train['transmission'] = df_train['transmission'].replace('2-Speed Automatic', '2-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('7-Speed DCT Automatic ', '7-Speed A/T')
  df_train['transmission'] = df_train['transmission'].replace('6-Speed Manual', '6-Speed M/T')
  df_train['transmission'] = df_train['transmission'].replace('Automatic CVT', 'CVT')
  
  df_train = df_train[~df_train['transmission'].str.contains('SCHEDULED FOR OR IN PRODUCTION', case=False, na=False)]
  df_train = df_train[~df_train['transmission'].str.contains('CVT-F', case=False, na=False)]
  df_train = df_train[~df_train['transmission'].str.contains('Single-Speed Fixed Gear', case=False, na=False)]
  df_train = df_train[~df_train['transmission'].str.contains('Manual', case=False, na=False)]
  df_train = df_train[~df_train['transmission'].str.contains('6 Speed At/Mt', case=False, na=False)]
  df_train = df_train[~df_train['transmission'].str.contains('F', case=False, na=False)]
  df_train = df_train[~df_train['transmission'].str.contains('2', case=False, na=False)]
  df_train = df_train[~df_train['transmission'].str.contains('Manual', case=False, na=False)]
  df_train = df_train[~df_train['transmission'].str.contains('–', case=False, na=False)]
  df_train = df_train[~df_train['transmission'].str.contains('Variable', case=False, na=False)]
  
  # EXT COL
  df_train['ext_col'] = df_train['ext_col'].replace('Agate Black Metallic', 'Black Black Metallic')
  df_train['ext_col'] = df_train['ext_col'].replace('BLACK', 'Black')
  df_train['ext_col'] = df_train['ext_col'].replace('BLU ELEOS', 'Blue')
  df_train['ext_col'] = df_train['ext_col'].replace('BLUE', 'Blue')
  df_train['ext_col'] = df_train['ext_col'].replace('Blu', 'Blue')
  df_train['ext_col'] = df_train['ext_col'].replace('Carrara White Metallic', 'White Metallic')
  df_train['ext_col'] = df_train['ext_col'].replace('DB Black Clearcoat', 'Black')
  df_train['ext_col'] = df_train['ext_col'].replace('Daytona Gray Pearl Effect w/ Black Roof', 'Daytona Gray Pearl Effect')
  df_train['ext_col'] = df_train['ext_col'].replace('GT SILVER', 'Silver')
  df_train['ext_col'] = df_train['ext_col'].replace('Isle of Man Green Metallic', 'Green Metallic')
  df_train['ext_col'] = df_train['ext_col'].replace('MANUFAKTUR Diamond White Bright', 'Diamond White Bright')
  df_train['ext_col'] = df_train['ext_col'].replace('Phantom Black Pearl Effect / Black Roof', 'Phantom Black Pearl Effect')
  
  df_train = df_train[~df_train['ext_col'].str.contains('C / C', case=False, na=False)]
  df_train = df_train[~df_train['ext_col'].str.contains('Custom Color', case=False, na=False)]
  df_train = df_train[~df_train['ext_col'].str.contains('Go Mango!', case=False, na=False)]
  df_train = df_train[~df_train['ext_col'].str.contains('Grigio Nimbus', case=False, na=False)]
  df_train = df_train[~df_train['ext_col'].str.contains('Tan', case=False, na=False)]
  df_train = df_train[~df_train['ext_col'].str.contains('-', case=False, na=False)]
  
  # INT COL
  colores = df_train['int_col'].str.split('/', n=1)
  df_train['int_col'] = colores.str[0].str.strip()
  df_train['int_col_second'] = colores.str[1].fillna('No').str.strip()
  
  df_train['int_col'] = df_train['int_col'].replace('AMG Black', 'Black')
  df_train['int_col'] = df_train['int_col'].replace('Adrenaline Red', 'Red')
  df_train['int_col'] = df_train['int_col'].replace('Agave Green', 'Green')
  df_train['int_col'] = df_train['int_col'].replace('Almond Beige', 'Beige')
  df_train['int_col'] = df_train['int_col'].replace('BEIGE', 'Beige')
  df_train['int_col'] = df_train['int_col'].replace('BLACK', 'Black')
  df_train['int_col'] = df_train['int_col'].replace('Beluga Hide', 'Beluga')
  df_train['int_col'] = df_train['int_col'].replace('Bianco Polar', 'White')
  df_train['int_col'] = df_train['int_col'].replace('Black Onyx', 'Black')
  df_train['int_col'] = df_train['int_col'].replace('Black w', 'Black')
  df_train['int_col'] = df_train['int_col'].replace('Camel Leather', 'Camel')
  df_train['int_col'] = df_train['int_col'].replace('Blk', 'Black')
  df_train['int_col'] = df_train['int_col'].replace('Canberra Beige', 'Beige')
  df_train['int_col'] = df_train['int_col'].replace('Charles Blue', 'Blue')
  df_train['int_col'] = df_train['int_col'].replace('Carbon Black', 'Black')
  df_train['int_col'] = df_train['int_col'].replace('Ebony Black', 'Black')
  df_train['int_col'] = df_train['int_col'].replace('Graphite w', 'Graphite')
  df_train['int_col'] = df_train['int_col'].replace('Gray w', 'Gray')
  df_train['int_col'] = df_train['int_col'].replace('Jet Black', 'Black')
  df_train['int_col'] = df_train['int_col'].replace('ORANGE', 'Orange')
  df_train['int_col'] = df_train['int_col'].replace('Obsidian Black', 'Black')
  df_train['int_col'] = df_train['int_col'].replace('Oyster W', 'Oyster')
  df_train['int_col'] = df_train['int_col'].replace('Parchment.', 'Parchment')
  df_train['int_col'] = df_train['int_col'].replace('Pearl Beige', 'Beige')
  df_train['int_col'] = df_train['int_col'].replace('Pimento Red w', 'Red')
  df_train['int_col'] = df_train['int_col'].replace('Sakhir Orange', 'Orange')
  df_train['int_col'] = df_train['int_col'].replace('Sand Beige', 'Beige')
  df_train['int_col'] = df_train['int_col'].replace('Silk Beige', 'Beige')
  df_train['int_col'] = df_train['int_col'].replace('Titan Black', 'Black')
  df_train['int_col'] = df_train['int_col'].replace('Very Light Cashmere', 'Cashmere')
  df_train['int_col'] = df_train['int_col'].replace('WHITE', 'White')
  df_train['int_col'] = df_train['int_col'].replace('Whisper Beige', 'Beige')
  
  df_train = df_train[~df_train['int_col'].str.contains('Cocoa', case=False, na=False)]
  df_train = df_train[~df_train['int_col'].str.contains('Ebony.', case=False, na=False)]
  df_train = df_train[~df_train['int_col'].str.contains('Tan', case=False, na=False)]
  df_train = df_train[~df_train['int_col'].str.contains('-', case=False, na=False)]
  
  df_train['int_col_second'] = df_train['int_col_second'].replace('Ebony Accents', 'Ebony')
  df_train['int_col_second'] = df_train['int_col_second'].replace('Ebony/Ebony', 'Ebony')
  df_train['int_col_second'] = df_train['int_col_second'].replace('Saddle Brown', 'Saddle')
