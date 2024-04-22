import pandas as pd
import numpy as np


class DataPreprocessor:
    INVALID_NUMBER = -9999

    # Define translation dictionary
    TRANSLATION_DICT = {
        "index": "Index",
        "Data": "Date",
        "Hora": "Hour",
        "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "Precip_mm",
        "RADIACAO GLOBAL (Kj/m²)": "Global Radiation (Kj/m²)",
        "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "Atm_pressure_mb",
        "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "Air_temp_C",
        "TEMPERATURA DO PONTO DE ORVALHO (°C)": "Dew_Point_Temp_C",
        "UMIDADE RELATIVA DO AR, HORARIA (%)": "Rel_Humidity_percent",
        "VENTO, DIREÇÃO HORARIA (gr) (° (gr))": "Wind_dir_deg",
        "VENTO, RAJADA MAXIMA (m/s)": "Gust",
        "VENTO, VELOCIDADE HORARIA (m/s)": "Wind_Speed",
        "region": "Brazilian_Geopolitical_Regions",
        "state": "State",
        "station": "Station_Name",
        "station_code": "Station_Code",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "height": "Elevation"
    }

    def __init__(self, dataset, D):
        self.dataset = dataset
        self.D = D
        self.select_col = [
            "index", "Data", "Hora", "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
            "RADIACAO GLOBAL (Kj/m²)",
            "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)",
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
            "TEMPERATURA DO PONTO DE ORVALHO (°C)",
            "UMIDADE RELATIVA DO AR, HORARIA (%)",
            "VENTO, DIREÇÃO HORARIA (gr) (° (gr))",
            "VENTO, RAJADA MAXIMA (m/s)",
            "VENTO, VELOCIDADE HORARIA (m/s)",
            "region", "state", "station", "station_code", "latitude", "longitude", "height"
        ]

        # Translate selected column names
        self.select_col = [self.TRANSLATION_DICT.get(col, col) for col in self.select_col]

    def preprocess(self):
        # Read the dataset and translate column headers
        self.dataset.rename(columns=self.TRANSLATION_DICT, inplace=True)
        self.dataset = self.dataset[self.select_col].copy()  # Selecting desired columns
        self.dataset.replace(self.INVALID_NUMBER, np.nan, inplace=True)  # Convert invalid rows to NaN
        self.dataset.dropna(inplace=True)  # Remove invalid rows
        self.dataset["Date"] = pd.to_datetime(self.dataset["Date"], format='ISO8601')  # Convert to datetime
        self.dataset["Hour"] = pd.DatetimeIndex(pd.to_datetime(self.dataset["Hour"], format='%H:%M')).hour

    def deriv(self, data):
        '''
        Determines derivatives up to Dth for sequential hourly data
        '''
        data = data.copy()
        diff_set = []  # store diff
        for d in range(1, self.D + 1):
            for col in data.columns:
                if col not in ["Index","Date", "Hour", "Wind_dir_deg",
                               "Brazilian_Geopolitical_Regions", "State", "Station_Name",
                               "Station_Code", "Latitude", "Longitude", "Elevation"]:
                    # diff for consecutive hours
                    diff_col_name = col + '_diff_' + str(d)
                    diff_col = data[col].diff(periods=d)
                    diff_col.loc[data['Hour'].diff(periods=d) != pd.Timedelta(hours=d)] = np.nan
                    diff_set.append(diff_col.rename(diff_col_name))
        return pd.concat([data] + diff_set, axis=1)

    def seq_prep(self, data):
        '''
        For D + 1 derivatives
        '''
        data = data.reset_index(drop=True)
        drop_portion = []

        # Find sequences with fewer than D + 1 entries
        for i in range(len(data) - self.D):
            if not all(data.index[j] == data.index[j + 1] - 1 for j in range(i, i + self.D)):
                drop_portion.extend(range(i, i + self.D + 1))

        # Drop sequences with fewer than D + 1 entries
        data = data.drop(drop_portion)
        return data


root = './tmp/data/Weather/'
# region = 'central_west'
region = 'north'

# Read CSV dataset
dataset = pd.read_csv(root + region + '.csv')

# Create DataPreprocessor object
preprocessor = DataPreprocessor(dataset, 8)

# Preprocess data
preprocessor.preprocess()
preprocessor.dataset.to_csv(root + region + '_processed.csv', index=False)
