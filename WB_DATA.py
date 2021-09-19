import requests, re, copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA

def get_wb_data(format,indicator,country):

    page = 0
    last_page = 0

    data_list = []

    #We assume that there will be at least one page, also if there would be something like page = 0 then it would throw an error code 120 and it just wouldn't make any sense.
    while True:
        page += 1

        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

        param = {
           #'date': "2000:2001",
           'format': format,
           'page': page
        }

        response = requests.request(
           "GET",
           url,
           params=param
        )

        #We don't want to catch it because if the URL is incorrect or some parameters are, then it'll simply return what we have in a while true loop. We can just skip it in except but we don't want incomplete data for now when API is working properly.
        if(response.status_code != 200):
            raise RuntimeError(f"Received response code {response.status_code}, please check the URL and try again.")

        data = (response.json())

        if(last_page == 0):
            last_page = data[0]['pages']

        data_list = data_list + data[1]

        if(data[0]['page'] == last_page):
            break

    #print(data_list)
    return data_list

def cleanse_data(input_list):

    clean_list = []

    for x in input_list:
        x_dict = {
            'indicator id': x['indicator']['id'].upper(),
            'indicator value': x['indicator']['value'].upper(),
            'country value': x['country']['value'].upper(),
            'country iso code': x['countryiso3code'].upper(),
            'date': x['date'],
            'value': x['value'],
            'unit': x['unit'],
            'obs status': x['obs_status'],
            'decimal': x['decimal'],
            'source': 'API'
        }
        clean_list.append(x_dict)

    df = pd.DataFrame(clean_list)

    #remove rows where date == 0 and swap NaN to 0 if there are any. + Drop duplicates where the combination of country iso code and date repeats itself.
    df = df[(df.date != 0)].fillna(0) 
    df_cleaned = df[(df.date != 0)].fillna(0).drop_duplicates(subset=['country iso code', 'date'], keep=False)

    #Changing values for the years like 2020 in jpn (please see the 'NY.GDP.MKTP.CN' indicators with jpn as country code to see that 2020 value is missing which affects forecast).
    #In short, if value is missing: copy value from the previous year. ARIMA doesn't like missing data for a last row in a series.
    for x in range(0,len(df_cleaned)):
        if(int(df_cleaned['value'][x]) == 0 and re.search("^20[1-2][0-9]",str(df_cleaned['date'])) == None):
            if(x < len(df_cleaned)-1):
                df_cleaned['value'][x] = df_cleaned['value'][x+1]

    df_plot = df_cleaned[['date','value']]

    #Remove empty rows from the beginning (chronologically speaking)
    for x in reversed(range(0,len(df_cleaned))):
        if(df_cleaned['value'][x] == 0):
            df_cleaned.drop(index=df_cleaned.index[x], axis=0, inplace=True)
        else:
            break

    df_cleaned = df_cleaned.set_index(['date'])

    return df_plot, df_cleaned

def forecast(time_series, chart_name):
    #creating copy of input as there were some issues when I'd plot the df used in the ARIMA() with the forecast + 
    #reversing df to keep the years in the date column in the right order and avoid one AttributeError down the path with index.year.

    time_series = time_series.iloc[::-1]

    time_series_copy = time_series.copy()
    time_series_copy = time_series_copy.set_index('date')

    time_series["date"] = pd.to_numeric(time_series["date"])
    time_series = time_series.set_index('date')

    #Used (1,1,2) as a safe parameters. I didn't tune it much if anything.
    model = ARIMA(time_series_copy, order=(1,1,2)) 
    model_fit = model.fit()

    prediction = model_fit.forecast(10,alpha=0.05)
    prediction_series = pd.DataFrame(prediction)
    prediction_series['date'] = prediction_series.index.year

    prediction_series = prediction_series[['date','predicted_mean']]

    #Renaming the column so it could be merged without any major headache, just in case
    prediction_series = prediction_series.rename({'predicted_mean': 'value'}, axis='columns')

    prediction_series = prediction_series.set_index(['date'])

    time_series.index = list(time_series.index)

    plt.figure(figsize=(13,5), dpi=80)
    plt.xticks(rotation = 60)
    plt.plot(time_series, label='API')
    plt.plot(prediction_series, label='Forecast')
    plt.legend(loc="upper left")
    #plt.show()
    plt.savefig(f'Chart {chart_name}.png')

    #concat df from api with df from ARIMA / Add 'source' column and reverse it to
    prediction_series = prediction_series.iloc[::-1]
    prediction_series['source'] = 'Forecast'

    return prediction_series

def merge_results(prediction_series, api_data, fname):

    #Concat these 2 series and fill blank "cells" with the data from api rows
    df_list = [prediction_series,api_data]
    complete_df = pd.concat(df_list)
    complete_df = complete_df.iloc[::-1]
    complete_df = complete_df.ffill()
    print(complete_df)
    complete_df.to_csv(f"Output {fname}.csv")

#Removing any special chars for the file names
def remove_forbidden_chars(name):
    restricted_char_list = ['/',':','\\','*','?','"','<','>','|']
    for x in restricted_char_list:
        if x in name:
            name = name.replace(x,'_')
    return name

def main():
    #Main parameters which we need to define.
    #Loading casese from csv file as it's faster this way.
    input_list = pd.read_csv('INPUT.csv')

    for x in range(0,len(input_list)):

        api_format = "json"
        indicators = input_list.Indicators[x]
        country = input_list.Country[x]
        time_date = datetime.now().strftime("%d/%m/%Y")
        file_details = remove_forbidden_chars(f"{indicators} - {country} - {time_date}")

        wb_df = get_wb_data(api_format, indicators, country)
        cleaned_df, output_df = cleanse_data(wb_df)
        prediction_series = forecast(cleaned_df, file_details)

        merge_results(prediction_series, output_df, file_details)

if __name__ == "__main__":
    main()