import ast
import json
import re
import requests

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup

from maps import POI_MAP


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


def obtener_urls(url: str, df: pd.DataFrame) -> list:
    '''
    Función para extraer las url de los anuncios de la página.
    '''

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


def _safe_eval(x):
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


def parse_nested_tecnocasa(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()

    
    for c in ['features', 'precio', 'media', 'points_of_interest', 'energy_data']:
        if c in df.columns:
            df[c] = df[c].apply(_safe_eval)

    
    if 'features' in df.columns:
        f = df['features']
        df['planta'] = f.apply(lambda d: d.get('floor'))
        df['aire_acondicionado'] = f.apply(lambda d: d.get('air_conditioning'))
        df['ascensor'] = f.apply(lambda d: d.get('elevator'))
        df['calefaccion'] = f.apply(lambda d: d.get('heating'))
        df['categoria'] = f.apply(lambda d: d.get('category'))
        df['año_construccion'] = f.apply(lambda d: d.get('build_year'))


    
    if 'media' in df.columns:
        m = df['media']
        df['planos'] = m.apply(lambda d: d.get('floor_plans') is not None)
        df['realistico'] = m.apply(lambda d: d.get('has_realistico'))
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


def _to_m(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).lower().replace(',', '.').replace(' ', '')
    if 'km' in t:
        return float(t.replace('km', '')) * 1000
    if 'm' in t:
        return float(t.replace('m', ''))
    return np.nan


def poi_features_tecnocasa(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()

    for col, pref in POI_MAP.items():
        if col not in df.columns:
            continue

        df[f'{pref}_cnt'] = df[col].apply(lambda lst: len(lst) if lst else 0)

        df[f'{pref}_min_dist_m'] = df[col].apply(
            lambda lst: np.nan if not lst else min(
                _to_m(d.get('distance'))
                for d in lst
                if isinstance(d, dict) and d.get('distance')
            )
        )

    return df


def final_clean_tecnocasa(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()

    # --- Dormitorios / baños: pasar a numérico ---
    if 'dormitorios' in df.columns:
        df['dormitorios'] = (df['dormitorios']
                             .astype(str)
                             .str.replace(r'\s*dorm\.', '', regex=True)
                             .str.strip())
        df['dormitorios'] = pd.to_numeric(df['dormitorios'], errors='coerce')

    if 'baños' in df.columns:
        df['baños'] = (df['baños']
                       .astype(str)
                       .str.replace(r'\s*baño[s]*', '', regex=True)
                       .str.strip())
        df['baños'] = pd.to_numeric(df['baños'], errors='coerce')

    # --- Planta: sacar paréntesis (ej: '3 (Exterior)' -> '3') ---
    if 'planta' in df.columns:
        df['planta'] = (df['planta']
                        .astype(str)
                        .str.replace(r'\s*\(.*\)', '', regex=True)
                        .str.strip())

    # --- Aire acondicionado: limpieza simple de texto ---
    if 'aire_acondicionado' in df.columns:
        df['aire_acondicionado'] = (df['aire_acondicionado']
                                    .astype(str)
                                    .str.replace(r'\s+.+', '', regex=True)
                                    .str.strip())

    # --- Calefacción: flags y limpieza ---
    if 'calefaccion' in df.columns:
        cal = df['calefaccion'].astype(str)
        df['calefaccion_gas'] = cal.str.contains('gas', case=False, na=False)
        df['calefaccion_electrica'] = cal.str.contains('eléctrica|electrica', case=False, na=False)
        df['calefaccion'] = cal.str.replace(r'\s+.+', '', regex=True).str.strip()

    # --- Energía: limpiar comas / espacios raros ---
    if 'eficiencia_energetica' in df.columns:
        s = df['eficiencia_energetica'].astype(str).str.lower().str.replace(',', '.', regex=False)
        df['eficiencia_energetica'] = pd.to_numeric(s.str.extract(r'(\d+(\.\d+)?)')[0], errors='coerce')

    if 'emisiones_energeticas' in df.columns:
        df['emisiones_energeticas'] = (df['emisiones_energeticas']
                                       .astype(str)
                                       .str.replace(',', '.', regex=False)
                                       .replace({'nan': np.nan, 'None': np.nan, '': np.nan})).astype(float, errors='ignore')

    # --- Booleanos: asegurar 0/1 en ascensor, planos, etc. (si existen) ---
    for b in ['ascensor', 'planos', 'realistico', 'calefaccion_gas', 'calefaccion_electrica']:
        if b in df.columns:
            df[b] = (
                df[b]
                .replace({'True': True, 'False': False, 'true': True, 'false': False, 'Sí': True, 'Si': True, 'No': False, '1': True, '0': False})
                .fillna(False)
                .astype(bool)
                .astype(int)
            )

    df = df.replace({'None': np.nan, 'nan': np.nan, '': np.nan, 'NA': np.nan, 'N/A': np.nan})

    return df