import matplotlib
from preprocessing.data_cleanner import DataCleanner

matplotlib.use('TkAgg')

# Initialization and data cleaning
cleaner = DataCleanner("data/immoweb-dataset.csv")
cleaner.send_output_file("data/data_cleanned.csv")

# Convert -1 values to NaN so they are not included in the correlation
df = cleaner.to_real_values() 