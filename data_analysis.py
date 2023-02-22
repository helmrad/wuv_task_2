import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random

# Reading 'xlsx' files requires openpyxl
# Install openpyxl 3.1 rather than 3.1.1

data_path = 'data//'

# AidData ChinesePublicDiplomacy.csv: provides country-year aggregates for each public diplomacy measure
# https://www.aiddata.org/data/chinas-public-diplomacy-dashboard-dataset-version-2);
fname_aidd = 'data1_AidData.csv'
fname_cari = 'data2_CARI.xlsx'
fname_bank = 'data3_WorldBank.txt'
# SAIS CARI debt relief: http://www.sais-cari.org/debt-relief
# World Bank: https://databank.worldbank.org/source/world-development-indicators

data_aidd = pd.read_csv(data_path + fname_aidd, sep=',')
data_canc = pd.read_excel(data_path + fname_cari, sheet_name='Cancellation', engine='openpyxl')
data_rest = pd.read_excel(data_path + fname_cari, sheet_name='Restructuring', engine='openpyxl')
data_bank = pd.read_csv(data_path + fname_bank, sep='\t')

##################
# Clean datasets #
##################

# AidData
# Rename column 'year' to 'Year' & 'receiving country' to 'Country'
data_aidd = data_aidd.rename(columns={'year': 'Year', 'receiving_country': 'Country'})

# CARI cancellation
# Drop empty columns in CARI cancellation, as well as 'ID' and 'Sources' columns
data_canc = data_canc[['Country', 'Year', 'USD Millions']]
# Drop empty rows in CARI cancellation
data_canc = data_canc[data_canc.Country.isna()==False]
# Convert 'Year' column from float to integer
data_canc['Year'] = data_canc['Year'].astype(int)
# Sum up the amount of several cancellations per country per year (Zambia, 2019)
data_canc = data_canc.groupby(['Country', 'Year']).sum().reset_index()
# This converts missing numbers to zeros, reverse that
data_canc['USD Millions'] = data_canc['USD Millions'].replace(0, np.nan)

# CARI restructuring
# Rename column '# of loans'
data_rest = data_rest.rename(columns={'# of loans (0 for duplicates, figures represent "at least")': 'Number of loans'})
# Retain only columns 'Country', 'Year', 'USD Millions', '# of loans'
data_rest = data_rest[['Country', 'Year', 'USD Millions', 'Number of loans']]
# Drop empty rows
data_rest = data_rest[data_rest.Country.isna()==False]
# Drop last row with no indication of year
data_rest = data_rest[:-1]
# Convert 'Year' column from float to integer
data_rest['Year'] = data_rest['Year'].astype(int)
# Remove asterisks from values
data_rest['USD Millions'] = data_rest['USD Millions'].astype(str).str.replace('*', '')
data_rest['USD Millions'] = data_rest['USD Millions'].astype(float)
# # Divide into two data series: amount of restructuring, number of restructurings
# data_rest_amt = data_rest[['Country', 'Year', 'USD Millions']]
# data_rest_no  = data_rest[['Country', 'Year', 'Number of loans']]

# World Bank
# Rename column 'Country Name' to 'Country'
data_bank = data_bank.rename(columns={'Country Name': 'Country'})
# Drop columns 'Country Code' and 'Series Code'
data_bank = data_bank.drop(['Country Code', 'Series Code'], axis=1)
# Drop last five rows which are descriptive
data_bank = data_bank[:-5]
# Rename year columns from 'YYYY [YRYYYY]' to 'YYYY' (i.e., retain first 4 digits)
cols_bank = data_bank.columns.to_list()
years = [int(col[0:4]) for col in cols_bank[2:]]
cols_bank_re = cols_bank[:2] + years
# Build a dictionary that maps old to new column names, as pd library requires a dictionary to rename columns
d_cols = {}
for col_old, col_new in zip(cols_bank, cols_bank_re):
    d_cols[col_old] = col_new
data_bank = data_bank.rename(columns=d_cols)
# Convert columns with numerical values to type float
for col in years:
    data_bank[col] = data_bank[col].replace('..', np.nan).astype('float')

##################
# Merge datasets #
##################

# Merge datasets according to the formatting of the World Bank data
data = deepcopy(data_bank)

# Harmonize country names across datasets
# Get country naming of each data source
countries_aidd = list(data_aidd.Country.unique())
countries_canc = list(data_canc.Country.unique())
countries_rest = list(data_rest.Country.unique())
countries_bank = list(data_bank.Country.unique())

# Take world bank country names as basis
# See whether any country name is not contained in the bank's list
countries = list(set(countries_rest).difference(countries_bank))
# ROC (Taiwan) is not part of the bank's list
countries_re_rest = {'ROC': 'Republic of China'}
# Rename
data_rest['Country'] = data_rest['Country'].replace(countries_re_rest)

countries = list(set(countries_canc).difference(countries_bank))
countries_re_canc = {'ROC': 'Republic of China',
                     'DRC': 'Congo, Dem. Rep.',
                     'CAR': 'Central African Republic',
                     'The Gambia': 'Gambia, The',
                     'Cape Verde': 'Cabo Verde',
                     'Sao Tome & Principe': 'Sao Tome and Principe',
                     "Cote D'Ivoire": "Cote d'Ivoire"}
data_canc['Country'] = data_canc['Country'].replace(countries_re_canc)

countries = list(set(countries_aidd).difference(countries_bank))
countries_re_aidd = {'North Korea': "Korea, Dem. People's Rep.",
                     'Laos': 'Lao PDR',
                     'Micronesia': 'Micronesia, Fed. Sts.',
                     'Brunei': 'Brunei Darussalam',
                     'Kyrgyzstan': 'Kyrgyz Republic',
                     'South Korea': 'Korea, Rep.'}
data_aidd['Country'] = data_aidd['Country'].replace(countries_re_aidd)

cols = data.columns.to_list()
# Merge each indicator of the AidData dataset with the main dataset
aidd_indicators = data_aidd.columns.to_list()[2:]
for col_name in aidd_indicators:
    # Reshape the data according to the world bank format: columns become rows and years become columns
    indicator = data_aidd.pivot(index='Country', columns='Year', values=col_name).reset_index()
    indicator.insert(1, 'Series Name', col_name)
    cols_indicator = indicator.columns.to_list()
    # Extract common columns, based on which the two datasets will be merged
    cols_common = [c for c in cols_indicator if c in cols]
    # Merge the main dataset with the indicator
    data = pd.merge(data, indicator, on=cols_common, how='outer')

# Merge the CARI cancellation dataset with the main dataset
# Reshape the data according to the world bank format: columns become rows and years become columns
indicator = data_canc.pivot(index='Country', columns='Year', values='USD Millions').reset_index()
indicator.insert(1, 'Series Name', 'Debt Cancellation [US$ Millions]')
cols_indicator = indicator.columns.to_list()
# Extract common columns, based on which the two datasets will be merged
cols_common = [c for c in cols_indicator if c in cols]
# Merge the main dataset with the indicator
data = pd.merge(data, indicator, on=cols_common, how='outer')

# Merge the CARI restructuring dataset with the main dataset
# First, the total amount of debt restructuring per country per year
# Reshape the data according to the world bank format: columns become rows and years become columns
indicator = data_rest.pivot(index='Country', columns='Year', values='USD Millions').reset_index()
indicator.insert(1, 'Series Name', 'Debt Restructuring [US$ Millions]')
cols_indicator = indicator.columns.to_list()
# Extract common columns, based on which the two datasets will be merged
cols_common = [c for c in cols_indicator if c in cols]
# Merge the main dataset with the indicator
data = pd.merge(data, indicator, on=cols_common, how='outer')

# Second, the number of loans restructured per country per year
# Reshape the data according to the world bank format: columns become rows and years become columns
indicator = data_rest.pivot(index='Country', columns='Year', values='Number of loans').reset_index()
indicator.insert(1, 'Series Name', 'Number of loans restructured')
cols_indicator = indicator.columns.to_list()
# Extract common columns, based on which the two datasets will be merged
cols_common = [c for c in cols_indicator if c in cols]
# Merge the main dataset with the indicator
data = pd.merge(data, indicator, on=cols_common, how='outer')

# Save dataset
data.to_csv(path_or_buf=data_path+'task2_dataset.csv', sep=',', index=False)

# Read dataset
data = pd.read_csv(data_path+'task2_dataset.csv', sep=',')


###########################
# CPIA debt policy rating #
###########################

# CPIA criteria
# https://thedocs.worldbank.org/en/doc/69484a2e6ae5ecc94321f63179bfb837-0290032022/original/CPIA-Criteria-2021.pdf, pp. 12-14
# Rating 1 a. "the country has recently engaged or in the near future will likely engage in debt restructuring negotiations"
# In other words, likely future, current, or recent debt restructuring should drag the rating down
# On the other hand, it is unclear whether this part of the 2021 rating has been applied throughout 2000-2020

# Methodology:
# a) contrast the amount of debt cancellation received with the mean CPIA debt policy rating for each country
# b) evaluate the correlation of debt cancellation time series with the CPIA rating time series (Pearson coefficient)

# Get a list of countries who received debt cancellation from China AND for which exists a CPIA rating
countries_canc = list(data.loc[data['Series Name'] == 'Debt Cancellation [US$ Millions]'].Country.unique())
countries_cpia = list(data.loc[data['Series Name'] == 'CPIA debt policy rating (1=low to 6=high)'].Country.unique())
countries = list(set(countries_canc) & set(countries_cpia))

# For each country, calculate Pearson correlation coefficient between CPIA and amt of debt cancellation
# Also, for each country get the average debt rating and the total amount of debt cancellation
r_cpia_canc = {}
cpia_avg = {}
canc_tot = {}
for country in countries:
    # Get a time series of cpia ratings for the country under scrutiny
    cpia = data.loc[data['Country'] == country].loc[data['Series Name'] == 'CPIA debt policy rating (1=low to 6=high)']
    cpia = cpia.iloc[0, 2:]
    cpia_mean = cpia.mean()
    # If all ratings for that country are missing, skip it
    if cpia.isnull().all():
        pass
    # If cpia ratings are populated, continue
    else:
        # CPIA ratings started being assigned only in the 2000s, so to calculate the correlation over the entire time series:
        # Fill earlier, non-populated CPIA ratings with the first ever assigned rating
        # This shouldn't matter as the correlation method considers differences rather than absolute values
        cpia = cpia.fillna(cpia[cpia.first_valid_index()]).astype(float)
        # Get a time series of debt cancellations for the country under scrutiny
        canc = data.loc[data['Country'] == country].loc[data['Series Name'] == 'Debt Cancellation [US$ Millions]']
        # Fill empty fields of debt cancellation amount with 0
        canc = canc.iloc[0, 2:].fillna(0).astype(float)
        # Cumulate received amount of debt cancellation, as this is necessary for a meaningful calculation of the correlation
        canc = canc.cumsum()
        # Calculate the Pearson coefficient
        r = cpia.corr(canc, method='pearson')
        # If the coefficient is not-a-number, this means that the debt policy rating has been constant
        # In this case, the coefficient can be set to 0, as there appears to be no correlation
        if np.isnan(r):
            r_cpia_canc[country] = 0
            cpia_avg[country] = cpia_mean
            canc_tot[country] = canc.iloc[-1]
        else:
            r_cpia_canc[country] = r
            cpia_avg[country] = cpia_mean
            canc_tot[country] = canc.iloc[-1]

# Visualize total debt cancellation vs cpia rating
fig, ax = plt.subplots()
for country in canc_tot.keys():
    color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
    ax.scatter(canc_tot[country], cpia_avg[country], label=country, c=color)
ax.legend(fontsize=5)
ax.set_xlabel('Total amount of debt cancellation received by a country [million USD]', fontsize=12)
ax.set_ylabel('CPIA debt policy rating (higher rating, better policy)', fontsize=12)

# Based on this visualization, there doesn't seem to be a trend that would, across countries, generally push the CPIA rating
# into a certain direction, based on the amount of debt cancellation received. However, it is possible that such a trend would
# appear if we could control for other variables that might influence the CPIA rating, and isolate the influence of the amount of
# debt cancellation

# Visualize correlation coefficient by country
# Between 1 and 0: positive correlation (series a increases, series b increases)
# 0: no correlation
# Between 0 and -1: negative correlation (series a increases, series b decreases)
# Calculate mean and std of the correlation coefficient and plot the results
r_countries = list(r_cpia_canc.keys())
r_values = list(r_cpia_canc.values())
r_mean = np.nanmean(r_values)
r_std = np.nanstd(r_values)

fig, ax = plt.subplots()
x_axis = np.arange(0, len(r_cpia_canc), 1)
ax.set_axisbelow(True)
ax.grid(color='gray', linestyle='dashed', alpha=0.5)
ax.scatter(x_axis, r_values)
ax.scatter(np.mean(x_axis), r_mean, c='red', label='mean', marker='*', s=100)
ax.scatter(np.mean(x_axis), r_std, c='green', label='std', marker='*', s=100)
ax.legend()
ax.set_xticks(x_axis)
ax.set_xticklabels(r_countries, rotation=90)
ax.set_xlabel('Country', fontsize=12)
ax.set_ylabel('Pearson correlation coefficient', fontsize=12)

# The average correlation for a country is mildly positive (i.e. amt of debt cancellation increases, debt policy rating increases)
# This is not what was expected in light of the criteria for setting the CPIA rating explained above. However, the interpretation
# of the mean becomes less expressive in light of the significant standard deviation
# An interesting observation is also that the results appear to cluster in three regions:
# Little correlation, strong negative correlation (smaller than -0.5), strong positive correlation (larger than 0.5).
# This could be coincidence, or that Chinese debt cancellation simply doesn't matter much in the CPIA process,
# but it could also hint at a third factor that is at play, possibly related to political orientation, which may decide whether
# debt cancellation by China influences the CPIA rating to the positive or to the negative.

# Limitation of the method and how to improve on it
# The temporal resolution of this method is not very granular. It seems intuitive that a case of debt cancellation has the most
# influence on the CPIA rating in the years surrounding the event. However, the Pearson correlation coefficient considers a very
# long time horizon. For example, if a country receives debt cancellation in 2003, and the CPIA rating increases in 2017, the
# Pearson coefficient would still capture this and be tilted towards a positive value, even though it seems unlikely in practice
# that a debt cancellation from 2003 influences the CPIA rating in 2017. This could be mitigated by extracting time intervals
# surrounding the events of debt cancellation by a few years, and then calculating the correlation on that.
# In addition, as mentioned previously, an isolation of the effect of debt cancellation on the CPIA rating by controlling for
# other variables that may affect the debt rating would also be desirable.


###############################
# Number of government visits #
###############################

# Get a list of countries who received debt cancellation from China AND for which exists data on government visits
countries_canc = list(data.loc[data['Series Name'] == 'Debt Cancellation [US$ Millions]'].Country.unique())
countries_govt = list(data.loc[data['Series Name'] == 'government_visits'].Country.unique())
countries = list(set(countries_canc) & set(countries_govt))

# As the two data streams are mutually exclusive in their country coverage, there is no analysis to do

#############
# NB Viewer #
#############

# Jupyter notebook can be started via command line
# Figure out how to update the environment used in the jupyter notebook
# Try uploading the executed Jupyter notebook onto my Github

# Set up NBviewer
