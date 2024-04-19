import pandas as pd
import numpy as np

class DataPreprocessor:
    # Define translation dictionary
    translation_dict = {
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
        self.select_col = [self.translation_dict.get(col, col) for col in self.select_col]

    def preprocess(self):
        # Read the dataset and translate column headers
        translate_d = self.dataset.rename(columns=self.translation_dict)

        # Selecting desired columns
        data2 = translate_d[
            self.select_col].copy()  # Ensure a copy to avoid SettingWithCopyWarning

        # Convert to datetime
        data2["Date"] = pd.to_datetime(data2["Date"])
        data2["Hour"] = pd.to_datetime(data2["Hour"], format='%H:%M').dt.time
        data2["Hour"] = data2.apply(lambda row: pd.Timestamp.combine(row['Date'], row['Hour']), axis=1)

        # Remove rows with -9999  or NaN values in any column
        data2.loc[data2["Hour"] == -9999, "Hour"] = np.nan
        data2 = data2.replace(-9999, np.nan).dropna()
        data2.dropna(inplace=True)

        # Compute derivatives
        data2 = self.deriv(data2)

        # Verify at least D + 1 entries in a row
        data2 = self.seq_prep(data2)

        return data2

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

# Read CSV dataset
dataset = pd.read_csv('./central_west.csv')

# Create DataPreprocessor object
preprocessor = DataPreprocessor(dataset, 8)

# Preprocess data
processed_data = preprocessor.preprocess()


processed_data.head(200).to_csv('practice3.csv', index=False)
