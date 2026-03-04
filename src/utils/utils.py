import ast
import json
import pandas as pd
import numpy as np
import requests

from bs4 import BeautifulSoup

from catboost import CatBoostRegressor

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMRegressor

from .maps import POI_MAP, DROP_COLS

from time import time

from xgboost import XGBRegressor



def _distancia_a_metros(s):
    '''
    Convierte una distancia tipo "1,4 Km" o "680 m" a metros (float).
    Si no puede parsear, devuelve NaN.
    '''
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    
    t = str(s).lower().replace(',', '.').replace(' ', '')
    if 'km' in t:
        return float(t.replace('km', '')) * 1000
    if 'm' in t:
        return float(t.replace('m', ''))
    return np.nan


def _safe_eval(x):
    '''
    Convierte strings tipo "{...}" o "[...]" en dict/list de forma segura.
    Si ya es dict/list lo devuelve tal cual.
    En caso de NaN o error, devuelve {}.
    '''
    if isinstance(x, (dict, list)):
        return x
    if pd.isna(x):
        return {}
    s = str(x).strip()
    if s in ('', '{}', '[]', 'None', 'nan'):
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}
    

def extraer_informacion(url: str) -> list:
    '''
    Función para extraer la información relevante de la página de un anuncio.
    '''

    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    estate_component = soup.find('estate-show-v2')
    if not estate_component:
        return []
    
    estate_json_raw = estate_component.get(':estate')
    estate_json_raw = estate_json_raw.replace('&quot;', '"')
    estate_data = json.loads(estate_json_raw)

    data = {
        'dormitorios': estate_data.get('rooms'),
        'superficie_m2': estate_data.get('numeric_surface'),
        'baños': estate_data.get('bathrooms'),
        'url': estate_data.get('detail_url'),
        'features': estate_data.get('features'),
        'descripcion': estate_data.get('description'),
        'precio': estate_data.get('costs'),
        'latitud': estate_data.get('latitude'),
        'longitud': estate_data.get('longitude'),
        'media': estate_data.get('media'),
        'points_of_interest': estate_data.get('points_of_interest'),
        'energy_data': estate_data.get('energy_data'),
    }
    
    return data


def obtener_urls(url: str, X: pd.DataFrame) -> list:
    '''
    Función para extraer las url de los anuncios de la página.
    '''

    df = X.copy()
    
    print(f'Buscando pisos en la página {url} ...')
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    estates_index = soup.find('estates-index')
    estates_raw = estates_index.get(':estates')
    estates_raw = estates_raw.replace('&quot;', '"')
    estates_data = json.loads(estates_raw)

    data = []

    for estate in estates_data:
        url_piso = estate.get('detail_url')

        if url_piso in df['url'].to_list():
            print('Ya lo tengo:', url_piso)
            return data
        data.append(extraer_informacion(url_piso))

    return data


def aplanar_campos_anidados(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Aplana columnas anidadas provenientes del scraping (dict/list en texto):
    - features, media, points_of_interest, energy_data

    Crea columnas planas como:
    - planta, ascensor, calefaccion, categoria, ano_construccion
    - planos, realista, fotografias
    - transporte_publico, escuelas, farmacias, ...
    - clase_energetica, eficiencia_energetica, emisiones_energeticas
    '''

    df = X.copy()

    for c in ['features', 'media', 'points_of_interest', 'energy_data']:
        if c in df.columns:
            df[c] = df[c].apply(_safe_eval)

    if 'features' in df.columns:
        f = df['features']
        df['planta'] = f.apply(lambda d: d.get('floor'))
        df['aire_acondicionado'] = f.apply(lambda d: d.get('air_conditioning'))
        df['ascensor'] = f.apply(lambda d: d.get('elevator'))
        df['calefaccion'] = f.apply(lambda d: d.get('heating'))
        df['categoria'] = f.apply(lambda d: d.get('category'))
        df['ano_construccion'] = f.apply(lambda d: d.get('build_year'))

    if 'media' in df.columns:
        m = df['media']
        df['planos'] = m.apply(lambda d: d.get('floor_plans') is not None)
        df['realista'] = m.apply(lambda d: d.get('has_realistico') is not False)
        df['fotografias'] = m.apply(lambda d: len(d.get('images', [])) if isinstance(d.get('images', []), list) else 0)

    if 'points_of_interest' in df.columns:
        poi = df['points_of_interest']
        df['transporte_publico'] = poi.apply(lambda d: d.get('public_transport'))
        df['escuelas'] = poi.apply(lambda d: d.get('school'))
        df['farmacias'] = poi.apply(lambda d: d.get('pharmacy'))
        df['hospitales'] = poi.apply(lambda d: d.get('hospital'))
        df['supermercados'] = poi.apply(lambda d: d.get('market'))
        df['tiendas'] = poi.apply(lambda d: d.get('shop'))
        df['bares'] = poi.apply(lambda d: d.get('bar'))
        df['restaurantes'] = poi.apply(lambda d: d.get('restaurant'))

    if 'energy_data' in df.columns:
        e = df['energy_data']
        df['clase_energetica'] = e.apply(lambda d: d.get('class_emissions'))
        df['eficiencia_energetica'] = e.apply(lambda d: d.get('efficiency'))
        df['emisiones_energeticas'] = e.apply(lambda d: d.get('emissions'))

    return df


def crear_features_poi(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Genera features numéricas a partir de columnas POI (puntos de interés) que son listas de diccionarios.
    Para cada POI definido en POI_MAP:
    - <pref>_cnt: cantidad de items
    - <pref>_min_dist_m: mínima distancia en metros
    '''
    df = X.copy()

    for col, pref in POI_MAP.items():
        if col not in df.columns:
            continue

        df[f'{pref}_cnt'] = df[col].apply(lambda lst: len(lst) if lst else 0)

        df[f'{pref}_min_dist_m'] = df[col].apply(
            lambda lst: np.nan if not lst else min(_distancia_a_metros(d.get('distance'))
                for d in lst
                if isinstance(d, dict) and d.get('distance'))
        ).fillna(10000)

    return df


def limpiar_y_crear_features(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Limpia y estandariza variables del dataset Tecnocasa y crea features numéricas.

    Devuelve un DataFrame listo para modelado: conversiones a numérico, normalización de booleanos,
    extracción de valores energéticos, recodificación de planta, binarización de aire acondicionado
    y codificación ordinal de categoría y clase energética.
    '''
    df = X.copy()

    df = df.replace({'None': np.nan, 'nan': np.nan, '': np.nan, 'NA': np.nan, 'N/A': np.nan})

    if 'dormitorios' in df.columns:
        df['dormitorios'] = (df['dormitorios'].astype(str).str.replace(r'\s*dorm\.', '', regex=True).str.strip())
        df['dormitorios'] = pd.to_numeric(df['dormitorios'], errors='coerce')

    if 'superficie_m2' in df.columns:
        df['superficie_m2'] = pd.to_numeric(df['superficie_m2'], errors='coerce')

    if 'baños' in df.columns:
        df['baños'] = (df['baños'].astype(str).str.replace(r'\s*baño[s]*', '', regex=True).str.strip())
        df['baños'] = pd.to_numeric(df['baños'], errors='coerce')

    if 'planta' in df.columns:
        df['planta'] = (df['planta'].astype(str).str.replace(r'\s*\(.*\)', '', regex=True).str.strip())

        num = pd.to_numeric(df['planta'], errors='coerce')
        med = num.median(skipna=True)
        q3 = num.quantile(0.75)

        df['planta'] = pd.to_numeric(
            df['planta'].replace({'Planta baja': 0, 'Baja': 0, 'Media': med, 'Alta': q3, 'Ático': q3}), errors='coerce'
        )

    if 'aire_acondicionado' in df.columns:
        df['aire_acondicionado'] = (df['aire_acondicionado'].astype(str).str.replace(r'\s+.+', '', regex=True).str.strip()).fillna('NO')

    if 'calefaccion' in df.columns:
        cal = df['calefaccion'].astype(str)
        df['calefaccion_gas'] = cal.str.contains('gas', case=False, na=False)
        df['calefaccion_electrica'] = cal.str.contains('eléctrica|electrica', case=False, na=False)
        df['calefaccion'] = cal.str.replace(r'\s+.+', '', regex=True).str.strip().fillna('NO')

    if 'eficiencia_energetica' in df.columns:
        s = df['eficiencia_energetica'].astype(str).str.lower().str.replace(',', '.', regex=False)
        df['eficiencia_energetica'] = pd.to_numeric(s.str.extract(r'(\d+(\.\d+)?)')[0], errors='coerce')

    if 'emisiones_energeticas' in df.columns:
        s = df['emisiones_energeticas'].astype(str).str.lower().str.replace(',', '.', regex=False)
        df['emisiones_energeticas'] = pd.to_numeric(s.str.extract(r'(\d+(\.\d+)?)')[0], errors='coerce')

    ''' Groupby de los datos de muestra.
    categoria
    De época    321957.434783
    Media       269423.411932
    Popular     288484.622120
    Señorial    408431.932203
    '''
    if 'categoria' in df.columns:
        map_cat = {'Popular': 0, 'Media': 0, 'De época': 1, 'Señorial': 2}
        df['categoria_ord'] = df['categoria'].map(map_cat)

    if 'clase_energetica' in df.columns:
        map_energy = {'g': 0, 'f': 1, 'e': 2, 'd': 3, 'c': 4, 'b': 5, 'a': 6}
        df['clase_energetica_ord'] = df['clase_energetica'].astype(str).str.lower().map(map_energy)
        df['tiene_certificado'] = ~df['clase_energetica'].isna()

    for c in ['categoria', 'clase_energetica']:
        if c in df.columns:
            df = df.drop(columns=[c])

    for b in ['ascensor', 'planos', 'realistico', 'calefaccion_gas', 'calefaccion_electrica']:
        if b in df.columns:
            df[b] = (df[b]
                     .replace({'True': True, 'False': False, 'true': True, 'false': False,
                               'Sí': True, 'sí': True, 'Si': True, 'si': True,
                               'No': False, 'no': False,
                               '1': True, '0': False})
                     .fillna(False)
                     .astype(bool)
                     .astype(int))

    return df


def make_objective_xgb(X, y, preprocess, cv=5):

    def objective(trial):

        model = XGBRegressor(random_state=42, objective='reg:absoluteerror', n_jobs=-1)

        pipe = Pipeline([('preprocess', preprocess), ('model', model)])

        params = {
            'model__n_estimators': trial.suggest_int('model__n_estimators', 500, 1200),
            'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.15, log=True),
            'model__max_depth': trial.suggest_int('model__max_depth', 4, 8),
            'model__min_child_weight': trial.suggest_int('model__min_child_weight', 1, 5),
            'model__subsample': trial.suggest_float('model__subsample', 0.7, 1.0),
            'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.7, 1.0),
            'model__reg_alpha': trial.suggest_float('model__reg_alpha', 1e-3, 1.0, log=True),
            'model__reg_lambda': trial.suggest_float('model__reg_lambda', 1.0, 5.0),
        }

        pipe.set_params(**params)

        model_log = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

        score = cross_val_score(model_log, X, y, cv=cv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1).mean()

        return score

    return objective


def make_objective_lgb(X, y, preprocess, cv=5):

    def objective(trial):

        model = LGBMRegressor(random_state=42, n_jobs=-1, objective='regression', verbosity=-1)

        pipe = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
        
        params = {
            'model__n_estimators': trial.suggest_int('model__n_estimators', 500, 1200),
            'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.15, log=True),
            'model__max_depth': trial.suggest_int('model__max_depth', 4, 12),
            'model__num_leaves': trial.suggest_int('model__num_leaves', 20, 100),
            'model__min_child_samples': trial.suggest_int('model__min_child_samples', 5, 50),
            'model__subsample': trial.suggest_float('model__subsample', 0.7, 1.0),
            'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.7, 1.0),
            'model__reg_alpha': trial.suggest_float('model__reg_alpha', 1e-3, 1.0, log=True),
            'model__reg_lambda': trial.suggest_float('model__reg_lambda', 1.0, 5.0),
        }

        pipe.set_params(**params)

        model_log = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

        score = cross_val_score(model_log, X, y, cv=cv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1).mean()

        return score

    return objective


def make_objective_cat(X, y, preprocess, cv=5):

    def objective(trial):

        model = CatBoostRegressor(random_state=42, loss_function="MAPE",verbose=False,)

        pipe = Pipeline(steps=[('preprocess', preprocess), ("model", model)])

        params = {
            "model__depth": trial.suggest_int("model__depth", 6, 10),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.15, log=True),
            "model__iterations": trial.suggest_int("model__iterations", 500, 1200),
            "model__l2_leaf_reg": trial.suggest_float("model__l2_leaf_reg", 1.0, 10.0),
            "model__subsample": trial.suggest_float("model__subsample", 0.7, 1.0),
            "model__random_strength": trial.suggest_float("model__random_strength", 0.5, 2.0),
        }

        pipe.set_params(**params)

        model_log = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

        score = cross_val_score(model_log, X, y, cv=cv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1).mean()

        return score

    return objective


def drop_columns(df):
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')


def performar_tiempo(X, y, preprocess, best_params, columna=None):

    if columna:
        X = X.drop(columns=columna)

    lista_col = []
    train_time = []
    score = []

    xgb_base = XGBRegressor(random_state=42, objective='reg:absoluteerror', n_jobs=-1)
    pipe = Pipeline(steps=[('preprocess', preprocess), ('model', xgb_base)])
    pipe.set_params(**best_params)
    pipe_log = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

    for col in X.columns:

        print(f'Ocultando {col}...', end='\t')
        lista_col.append(col)
        X_menos = X.drop(columns=col)

        start = time()
        pipe_log.fit(X_menos, y)
        train_time.append(time() - start)

        score.append(cross_val_score(pipe_log, X_menos, y, cv=5, scoring="neg_mean_absolute_percentage_error", n_jobs=-1).mean())
    
    resultados = pd.DataFrame({'columna': lista_col, 'score': score, 'time': train_time}).set_index('columna').sort_values('score', ascending=False).iloc[0]

    if len(X.columns) > 2:
        print(f'Eliminamos {resultados.name}...')
        resultados = pd.concat([resultados, performar_tiempo(X, y, preprocess, best_params, resultados.name)], axis=1)

    return resultados