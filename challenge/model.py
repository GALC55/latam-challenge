import pandas as pd

from typing import Tuple, Union, List
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
import xgboost as xgb

class DelayModel:

    def __init__(
            self
    ):
        self._model = None  # Model should be saved in this attribute.

    def preprocess(
            self,
            data: pd.DataFrame,
            target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:

        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'])
        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'])

        def is_high_season(date):
            if ((date.month == 12 and date.day >= 15) or
                    (date.month == 1) or
                    (date.month == 2 and date.day <= 28) or
                    (date.month == 3 and date.day <= 3) or
                    (date.month == 7 and 15 <= date.day <= 31) or
                    (date.month == 9 and 11 <= date.day <= 30)):
                return 1
            return 0

        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        # Calcular la diferencia en minutos
        data['min_diff'] = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60
        # Redondear a 1 decimal
        data['min_diff'] = data['min_diff'].round(1)

        # Definir la función para determinar el período del día
        def get_period_of_day(date):
            hour = date.hour
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 19:
                return 'afternoon'
            else:
                return 'night'

        # Aplicar la función de período del día
        data['period_day'] = data['Fecha-I'].apply(get_period_of_day)

        # Crear la columna delay
        data['delay'] = data['min_diff'].apply(lambda x: 1 if x > 15 else 0)
        data = data.drop(columns=['Fecha-I', 'Fecha-O','Vlo-I','Vlo-O','min_diff','Ori-I','Des-I','Emp-I','Ori-O','Des-O','Emp-O'])

        # Identificar las columnas categóricas
        categorical_features = data.select_dtypes(include=['object']).columns

        # Crear un DataFrame con solo las columnas categóricas
        categorical_data = data[categorical_features]
        # Imprimir el DataFrame con las columnas categóricas
        print(categorical_features)

        onehot_encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' para evitar la trampa de la variable ficticia
        onehot_encoded = onehot_encoder.fit_transform(categorical_data)

        # Paso 3: Convertir la matriz resultante a un DataFrame
        onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['DIANOM', 'TIPOVUELO', 'OPERA', 'SIGLAORI', 'SIGLADES', 'period_day']))

        # Paso 4: Concatenar las nuevas columnas al DataFrame original
        data_with_onehot = pd.concat([data, onehot_encoded_df], axis=1)

        # Paso 5: (Opcional) Eliminar las columnas categóricas originales
        data_with_onehot = data_with_onehot.drop(columns=['DIANOM', 'TIPOVUELO', 'OPERA', 'SIGLAORI', 'SIGLADES', 'period_day'])

        # Imprimir el DataFrame resultante
        print(data_with_onehot.head())

        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        # Determinar si se debe retornar el DataFrame con o sin objetivo
        if target_column:
            x = data_with_onehot.drop(columns=[target_column])
            y = data_with_onehot[target_column]
            return x, y
        else:
            return data_with_onehot

    def fit(
            self,
            features: pd.DataFrame,
            target: pd.DataFrame
    ) -> None:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        # Entrenar el modelo
        modelo = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            eval_metric='logloss'
        )
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        df = pd.DataFrame(data=pred)
        print(df)
        from sklearn.metrics import accuracy_score
        # Evaluar el modelo
        accuracy = accuracy_score(y_test, pred)
        print(f'Accuracy: {accuracy:.2f}')
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        return

    def predict(
            self,
            features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        return



