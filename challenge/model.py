import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime


class DelayModel:

    def __init__(self) -> None:
        """Inicializa el modelo DelayModel."""
        self._model = None  # Model should be saved in this attribute.

    def preprocess(
            self,
            data: pd.DataFrame,
            target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        añade columnas calculadas como 'high_season',
        'period_day', 'min_diff', y 'delay'. También codifica las variables categóricas
        necesarias para el modelo.

        Parameters:
        data (pd.DataFrame): El DataFrame original.
        target_column (str): Nombre de la columna objetivo.

        Returns:
        Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]: Features.
        """
        try:
            # nuevas columnas calculadas
            data['high_season'] = data['Fecha-I'].apply(self._is_high_season)
            data['period_day'] = data['Fecha-I'].apply(self._get_period_day)
            data['min_diff'] = data.apply(self._get_min_diff, axis=1)

            # columna de delay basada en un umbral de 15 minutos
            threshold_in_minutes = 15
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)


            training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'delay']], random_state=111)
            x = pd.concat([
                pd.get_dummies(training_data['OPERA'], prefix='OPERA'),
                pd.get_dummies(training_data['TIPOVUELO'], prefix='TIPOVUELO'),
                pd.get_dummies(training_data['MES'], prefix='MES')],
                axis=1
            )

            top_features = [
                "OPERA_Latin American Wings", "MES_7", "MES_10", "OPERA_Grupo LATAM", "MES_12",
                "TIPOVUELO_I", "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air"
            ]
            x = x[top_features]

            if target_column:
                y = training_data[[target_column]]
                return x, y
            else:
                return x
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None

    def fit(
            self,
            features: pd.DataFrame,
            target: pd.DataFrame
    ) -> None:
        """
        Entrena el modelo XGBoost con los datos preprocesados.

        Parameters:
        features (pd.DataFrame): Características de entrenamiento.
        target (pd.DataFrame): Variable objetivo.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

            # Ajustar el peso de las clases
            n_y0 = y_train[y_train == 0].count()
            n_y1 = y_train[y_train == 1].count()
            scale = float(n_y0 / n_y1)

            self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
            self._model.fit(X_train, y_train)

            # Guardar el modelo
            self._model.save_model("modelo.xgb")
        except Exception as e:
            print(f"Error during model training: {e}")

    def predict(
            self,
            features: pd.DataFrame
    ) -> List[int]:
        """
        Predice el retraso utilizando el modelo entrenado.

        Parameters:
        features (pd.DataFrame): Características a predecir.

        Returns:
        List[int]: Lista de predicciones (0 o 1).
        """
        try:
            pred = self._model.predict(features)
            return pred.tolist()
        except Exception as e:
            print(f"Error during prediction: {e}")
            return []

    # Métodos privados para funciones auxiliares

    def _is_high_season(self, fecha: str) -> int:
        """Determina si una fecha está en temporada alta."""
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        season_ranges = [
            (datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año),
             datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)),
            (datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año),
             datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)),
            (datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año),
             datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)),
            (datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año),
             datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año))
        ]
        for start, end in season_ranges:
            if start <= fecha <= end:
                return 1
        return 0

    def _get_period_day(self, date: str) -> str:
        """Determina el período del día (mañana, tarde, noche) basado en la hora."""
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        if datetime.strptime("05:00", '%H:%M').time() <= date_time <= datetime.strptime("11:59", '%H:%M').time():
            return 'mañana'
        elif datetime.strptime("12:00", '%H:%M').time() <= date_time <= datetime.strptime("18:59", '%H:%M').time():
            return 'tarde'
        else:
            return 'noche'

    def _get_min_diff(self, row: pd.Series) -> float:
        """Calcula la diferencia en minutos entre las fechas de salida y llegada."""
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        return (fecha_o - fecha_i).total_seconds() / 60

class FlightDataVisualizer:
    """
    Clase para manejar la visualización de los datos de vuelos y las tasas de retrasos.
    """

    @staticmethod
    def plot_flight_distribution(data: pd.DataFrame) -> None:
        """Genera gráficos sobre la distribución de vuelos por aerolínea, día, mes, etc."""
        flights_by_airline = data['OPERA'].value_counts()
        plt.figure(figsize=(10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)
        plt.title('Flights by Airline')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Airline', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()

        flights_by_day = data['DIA'].value_counts()
        plt.figure(figsize=(10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=flights_by_day.index, y=flights_by_day.values, color='lightblue', alpha=0.8)
        plt.title('Flights by Day')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Day', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()

        flights_by_month = data['MES'].value_counts()
        plt.figure(figsize=(10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=flights_by_month.index, y=flights_by_month.values, color='lightblue', alpha=0.8)
        plt.title('Flights by Month')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Month', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()

        flights_by_day_in_week = data['DIANOM'].value_counts()
        days = [
            flights_by_day_in_week.index[2],
            flights_by_day_in_week.index[5],
            flights_by_day_in_week.index[4],
            flights_by_day_in_week.index[1],
            flights_by_day_in_week.index[0],
            flights_by_day_in_week.index[6],
            flights_by_day_in_week.index[3]
        ]
        values_by_day = [
            flights_by_day_in_week.values[2],
            flights_by_day_in_week.values[5],
            flights_by_day_in_week.values[4],
            flights_by_day_in_week.values[1],
            flights_by_day_in_week.values[0],
            flights_by_day_in_week.values[6],
            flights_by_day_in_week.values[3]
        ]
        plt.figure(figsize=(10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=days, y=values_by_day, color='lightblue', alpha=0.8)
        plt.title('Flights by Day in Week')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Day in Week', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()

        flights_by_type = data['TIPOVUELO'].value_counts()
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 2))
        sns.barplot(x=flights_by_type.index, y=flights_by_type.values, alpha=0.9)
        plt.title('Flights by Type')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Type', fontsize=12)
        plt.show()

        flight_by_destination = data['SIGLADES'].value_counts()
        plt.figure(figsize=(10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=flight_by_destination.index, y=flight_by_destination.values, color='lightblue', alpha=0.8)
        plt.title('Flight by Destination')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Destination', fontsize=12)
        plt.xticks(rotation=90)

        plt.show()

    @staticmethod
    def plot_delay_rate(data: pd.DataFrame) -> None:
        """Genera gráficos sobre la tasa de retraso por diversas categorías."""
        def get_rate_from_column(data, column):
            delays = {}
            for _, row in data.iterrows():
                if row['delay'] == 1:
                    if row[column] not in delays:
                        delays[row[column]] = 1
                    else:
                        delays[row[column]] += 1
            total = data[column].value_counts().to_dict()

            rates = {}
            for name, total in total.items():
                if name in delays:
                    rates[name] = round(total / delays[name], 2)
                else:
                    rates[name] = 0

            return pd.DataFrame.from_dict(data=rates, orient='index', columns=['Tasa (%)'])

        destination_rate = get_rate_from_column(data, 'SIGLADES')
        destination_rate_values = data['SIGLADES'].value_counts().index
        plt.figure(figsize=(20, 5))
        sns.set(style="darkgrid")
        sns.barplot(x=destination_rate_values, y=destination_rate['Tasa (%)'], alpha=0.75)
        plt.title('Delay Rate by Destination')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Destination', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()

        airlines_rate = get_rate_from_column(data, 'OPERA')
        airlines_rate_values = data['OPERA'].value_counts().index
        plt.figure(figsize=(20, 5))
        sns.set(style="darkgrid")
        sns.barplot(x=airlines_rate_values, y=airlines_rate['Tasa (%)'], alpha=0.75)
        plt.title('Delay Rate by Airline')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Airline', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()

        month_rate = get_rate_from_column(data, 'MES')
        month_rate_value = data['MES'].value_counts().index
        plt.figure(figsize=(20, 5))
        sns.set(style="darkgrid")
        sns.barplot(x=month_rate_value, y=month_rate['Tasa (%)'], color='blue', alpha=0.75)
        plt.title('Delay Rate by Month')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Month', fontsize=12)
        plt.xticks(rotation=90)
        plt.ylim(0, 10)
        plt.show()

        days_rate = get_rate_from_column(data, 'DIANOM')
        days_rate_value = data['DIANOM'].value_counts().index

        sns.set(style="darkgrid")
        plt.figure(figsize=(20, 5))
        sns.barplot(x=days_rate_value, y=days_rate['Tasa (%)'], color='blue', alpha=0.75)
        plt.title('Delay Rate by Day')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Days', fontsize=12)
        plt.xticks(rotation=90)
        plt.ylim(0, 7)
        plt.show()

        high_season_rate = get_rate_from_column(data, 'high_season')
        high_season_rate_values = data['high_season'].value_counts().index

        plt.figure(figsize=(5, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=["no", "yes"], y=high_season_rate['Tasa (%)'])
        plt.title('Delay Rate by Season')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('High Season', fontsize=12)
        plt.xticks(rotation=90)
        plt.ylim(0, 6)
        plt.show()

        flight_type_rate = get_rate_from_column(data, 'TIPOVUELO')
        flight_type_rate_values = data['TIPOVUELO'].value_counts().index
        plt.figure(figsize=(5, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=flight_type_rate_values, y=flight_type_rate['Tasa (%)'])
        plt.title('Delay Rate by Flight Type')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Flight Type', fontsize=12)
        plt.ylim(0, 7)
        plt.show()