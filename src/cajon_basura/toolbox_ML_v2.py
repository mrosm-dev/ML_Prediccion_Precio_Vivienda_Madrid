import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr, mannwhitneyu, f_oneway

#####################################################################

def describe_df(data:pd.DataFrame, umbral_categorica:int=10, umbral_continua:float=0.1, imprimir:bool=True) -> pd.DataFrame:
    '''
    Genera una descripción esquemática de las variables de un DataFrame, devolviendo un nuevo DataFrame en el que cada columna corresponde
    a una variable del conjunto de datos original y cada fila resume una propiedad básica y una clasificación sugerida de dicha variable.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada sobre el que se calcula la descripción de variables.
    umbral_categorica : int, defecto=10
        Umbral de cardinalidad absoluta para considerar una variable como categórica nominal.
    umbral_continua : float, defecto=0.1
        Umbral de cardinalidad relativa a partir del cual una variable numérica o temporal se considera continua.
    imprimir : bool, defecto=True
        Indica si se imprime por pantalla un resumen de los umbrales utilizados en la clasificación.

    Returns
    -------
    pandas.DataFrame
        DataFrame resumen que contiene, para cada variable del DataFrame original, su tipo de dato, número y porcentaje de valores nulos,
        cardinalidad absoluta y relativa, y una clasificación sugerida del tipo de variable.
    '''

    if not(isinstance(data, pd.DataFrame)):
        raise TypeError('La variable "data" debe ser de un DataFrame.')
    if not(isinstance(umbral_categorica, int)):
        raise TypeError('La variable "umbral_categorica" debe ser un número entero.')
    if not(isinstance(umbral_continua, (int, float))):
        raise TypeError('La variable "umbral_continua" debe ser un número.')
    if not(0 <= umbral_continua <= 1):
        raise ValueError('El valor de "umbral_continua" debe estar comprendido entre 0 y 1.')
    if not(isinstance(imprimir, bool)):
        raise TypeError('La variable "imprimir" debe ser True/False.')
    
    salida = {columna: [] for columna in data.columns}
    for columna in salida.keys():
        tipo = data[columna].dtype
        nulos = data[columna].isna().sum()
        cardinalidad = data[columna].nunique()
        clasificacion = 'Bajo_Interes'

        if cardinalidad == 2:
            clasificacion = 'Categorica_Binaria'

        elif cardinalidad <= umbral_categorica:
            clasificacion = 'Categorica'

        elif pd.api.types.is_numeric_dtype(tipo) or pd.api.types.is_datetime64_any_dtype(tipo):
            if cardinalidad / len(data) <= umbral_continua:
                clasificacion = 'Numerica_Discreta'
            else:
                clasificacion = 'Numerica_Continua'

        salida[columna].append(tipo)
        salida[columna].append(nulos)
        salida[columna].append(round(nulos / len(data) * 100, 1))
        salida[columna].append(cardinalidad)
        salida[columna].append(round(cardinalidad / len(data) * 100, 2))
        salida[columna].append(clasificacion)

    salida['Columnas'] = ['Tipo_Dato', 'Nulos', 'Nulos_%', 'Cardinalidad', 'Cardinalidad_%', 'Clasificacion_sugerida']

    if imprimir:
        print(f'Clasificación sugerida para {len(data)} filas, con un umbral para categórica nominal de {umbral_categorica\
                } sobre la cardinalidad y un umbral para númerica continua de {umbral_continua*100} % sobre la cardinalidad relativa.')
    return pd.DataFrame(salida).set_index('Columnas')


def tipifica_variables(data:pd.DataFrame, umbral_categorica:int=10, umbral_continua:float=0.1) -> dict:
    '''
    Asigna una tipificación sugerida a las variables de un DataFrame en función de su cardinalidad y naturaleza, agrupándolas según su
    posible uso analítico como variables categóricas o numéricas.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada con las variables a analizar.
    umbral_categorica : int, defecto=10
        Umbral de cardinalidad absoluta para considerar una variable como categórica.
    umbral_continua : float, defecto=0.1
        Umbral de cardinalidad relativa a partir del cual una variable numérica se considera continua.

    Returns
    -------
    dict
        Diccionario cuyas claves corresponden a los tipos sugeridos de variable y cuyos valores son listas con los nombres de las
        columnas del DataFrame original asociadas a cada tipo.
    '''

    describe_data = describe_df(data, umbral_categorica, umbral_continua, False)
    diccionario = {'Categorica_Binaria': [], 'Categorica': [], 'Numerica_Discreta': [], 'Numerica_Continua': [], 'Bajo_Interes': []}
    [diccionario[describe_data.loc['Clasificacion_sugerida', columna]].append(columna) for columna in describe_data.columns]

    return diccionario


def get_features_num_regression(data:pd.DataFrame, target_col:str, umbral_corr:float=0.4, pvalue:float=0.05, variables_tipificadas:dict=None) -> pd.DataFrame:
    '''
    Selecciona las variables numéricas de un DataFrame que presentan una relación lineal significativa con una variable objetivo,
    atendiendo a un umbral mínimo de correlación y a un nivel de significación estadística.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada que contiene las variables explicativas y la variable objetivo.
    target_col : str
        Nombre de la columna que se utilizará como variable objetivo del modelo de regresión.
    umbral_corr : float, defecto=0.4
        Valor mínimo de correlación (en valor absoluto) requerido para que una variable sea seleccionada.
    pvalue : float, defecto=0.05
        Nivel de significación estadística máximo permitido en el test de correlación.
    variables_tipificadas : dict, opcional
        Diccionario con la tipificación previa de las variables; si no se proporciona, se calcula internamente.

    Returns
    -------
    pandas.DataFrame
        DataFrame con las variables numéricas seleccionadas, incluyendo su correlación con la variable objetivo y el p-value
        asociado al test estadístico.
    '''

    if not(isinstance(target_col, str)):
        raise TypeError('La variable "target_col" debe ser de un string.')
    if not(isinstance(umbral_corr, (int, float))):
        raise TypeError('La variable "umbral_corr" debe ser un número.')
    if not(0 <= umbral_corr <= 1):
        raise ValueError('El valor de "umbral_corr" debe estar comprendido entre 0 y 1.')
    if not(isinstance(pvalue, (int, float))):
        raise TypeError('La variable "pvalue" debe ser un número.')
    if not(0 <= pvalue <= 1):
        raise ValueError('El valor de "pvalue" debe estar comprendido entre 0 y 1.')
    
    if variables_tipificadas == None:
        variables_tipificadas = tipifica_variables(data)
    elif not(isinstance(variables_tipificadas, dict)):
        raise TypeError('La variable "variables_tipificadas" debe ser de un diccionario.')

    if not target_col in variables_tipificadas['Numerica_Continua']:
        raise ValueError('La columna "target_col" debe ser "Numerica_Continua".')

    data = data.dropna(axis=0)

    columnas_numericas = [columna for numericas in ['Numerica_Continua', 'Numerica_Discreta'] for columna in variables_tipificadas[numericas]]
    salida = {}
    for columna in columnas_numericas:
        if columna != target_col:
            corr_calc, pvalue_calc = pearsonr(data[columna], data[target_col])
            if corr_calc >= umbral_corr and pvalue_calc <= pvalue:
                salida[columna] = [corr_calc, pvalue_calc]

    return pd.DataFrame(salida, index=['Correlacion', 'P_value']).T


def plot_features_num_regression(data:pd.DataFrame, target_col:str, columns:list=[], umbral_corr:float=0.4, pvalue:float=0.05) -> list:
    '''
    Genera gráficos de dispersión entre una variable objetivo y un conjunto de variables numéricas seleccionadas en función
    de su correlación y significación estadística.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada que contiene las variables numéricas y la variable objetivo.
    target_col : str
        Nombre de la columna que se utilizará como variable objetivo.
    columns : str o list de str, defecto=[]
        Columna o lista de columnas candidatas a representar; si está vacía, se consideran todas las variables numéricas del DataFrame.
    umbral_corr : float, defecto=0.4
        Valor mínimo de correlación requerido para que una variable sea representada.
    pvalue : float, defecto=0.05
        Nivel de significación estadística máximo permitido en el test de correlación.

    Returns
    -------
    list
        Lista de nombres de las variables que cumplen los criterios y han sido representadas gráficamente.
    '''

    columns = data.columns.to_list() if columns == [] else columns
    if isinstance(columns, str):
        columns = [columns]
    if not(isinstance(columns, list)):
        raise TypeError('La variable "columns" debe ser de un solo string o una lista.')
    
    data = data.dropna(axis=0)

    columns_calc = get_features_num_regression(data, target_col, umbral_corr, pvalue)

    columnas_pintar = [col for col in columns if col in columns_calc.index]
    
    divisiones = math.ceil(len(columnas_pintar) / 4)
    for div in range(divisiones):
        ncols = min(4, len(columnas_pintar)-div*4)
        _, axs = plt.subplots(ncols=ncols, figsize = (5*ncols, 5))
        axs = axs if isinstance(axs, np.ndarray) else [axs]
        for i, ax in enumerate(axs):
            idx = div * 4 + i
            sns.scatterplot(data, x=columnas_pintar[idx], y=target_col, ax=ax)
            ax.set_xlabel(f'{columnas_pintar[idx]} \n corr = {columns_calc.loc[columnas_pintar[idx]]['Correlacion']:.3g} | p-value = {columns_calc.loc[columnas_pintar[idx]]['P_value']:.3g}')
            if i != 0:
                ax.set_ylabel('')
            else:
                ax.set_ylabel(target_col)

    return columnas_pintar


def get_features_cat_regression(data:pd.DataFrame, target_col:str, pvalue:float=0.05, variables_tipificadas:dict=None) -> pd.DataFrame:
    '''
    Identifica las variables categóricas de un DataFrame cuya relación con una variable objetivo numérica resulta estadísticamente
    significativa según el test de hipótesis adecuado en cada caso.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada que contiene las variables categóricas y la variable objetivo.
    target_col : str
        Nombre de la columna que actúa como variable objetivo numérica continua.
    pvalue : float, defecto=0.05
        Nivel de significación estadística máximo permitido en los tests de relación.
    variables_tipificadas : dict, opcional
        Diccionario con la tipificación previa de las variables; si no se proporciona, se calcula internamente.

    Returns
    -------
    pandas.DataFrame
        DataFrame con una fila que recoge el p-valor asociado a cada variable categórica cuya relación con la variable objetivo
        resulta estadísticamente significativa.
    '''

    if not(isinstance(target_col, str)):
        raise TypeError('La variable "target_col" debe ser de un string.')
    if not(isinstance(pvalue, (int, float))):
        raise TypeError('La variable "pvalue" debe ser un número.')
    if not(0 <= pvalue <= 1):
        raise ValueError('El valor de "pvalue" debe estar comprendido entre 0 y 1.')

    if variables_tipificadas == None:
        variables_tipificadas = tipifica_variables(data)
    elif not(isinstance(variables_tipificadas, dict)):
        raise TypeError('La variable "variables_tipificadas" debe ser de un diccionario.')

    if not target_col in variables_tipificadas['Numerica_Continua']:
        raise ValueError('La columna "target_col" debe ser "Numerica_Continua".')
    
    data = data.dropna(axis=0)

    columnas_categoricas = [columna for categoricas in ['Categorica_Binaria', 'Categorica_Nominal'] for columna in variables_tipificadas[categoricas]]
    salida = {}
    for columna in columnas_categoricas:
        if data[columna].nunique() == 2:
            grupo_a, grupo_b = [data[data[columna] == valor][target_col] for valor in data[columna].unique()]

            _, p_valor = mannwhitneyu(grupo_a, grupo_b)
            if p_valor <= pvalue:
                salida[columna] = p_valor
        else:
            lista = [data[data[columna] == valor][target_col] for valor in data[columna].unique()]
            _, p_valor = f_oneway(*lista)
            if p_valor <= pvalue:
                salida[columna] = p_valor
            
    return pd.DataFrame(salida, index=['P_value']).T


def plot_features_cat_regression(data:pd.DataFrame, target_col:str, columns:list=[], pvalue:float=0.05, with_individual_plot:bool=False) -> None:
    '''
    Visualiza la distribución de una variable objetivo numérica condicionada por variables categóricas cuya relación resulta
    estadísticamente significativa, generando gráficos de densidad para facilitar su comparación.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada que contiene las variables categóricas y la variable objetivo.
    target_col : str
        Nombre de la columna que actúa como variable objetivo numérica.
    columns : str o list de str, opcional
        Lista de variables categóricas candidatas a representar; si está vacía, se consideran todas las categóricas.
    pvalue : float, defecto=0.05
        Nivel de significación estadística máximo permitido en los tests de relación.
    with_individual_plot : bool, defecto=False
        Indica si los gráficos se generan separando individualmente los valores de cada variable categórica.

    Returns
    -------
    list
        Lista de nombres de variables categóricas finalmente representadas.
    '''

    columns = data.columns.to_list() if columns == [] else columns
    if isinstance(columns, str):
        columns = [columns]
    if not(isinstance(columns, list)):
        raise TypeError('La variable "columns" debe ser de un solo string o una lista.') 
    if not isinstance(with_individual_plot, bool):
        raise TypeError('La variable "with_individual_plot" debe ser True/False.')
    
    data = data.dropna(axis=0)

    columns_calc = get_features_cat_regression(data, target_col, pvalue)

    columnas_pintar = [col for col in columns if col in columns_calc.index]

    if with_individual_plot:
        for columna in columnas_pintar:
            valores = data[columna].unique()
            divisiones = math.ceil(len(valores) / 4)
            for div in range(divisiones):
                ncols = min(4, len(valores)-div*4)
                _, axs = plt.subplots(ncols=ncols, figsize = (5*ncols, 5))
                axs = axs if isinstance(axs, np.ndarray) else [axs]
                for i, ax in enumerate(axs):
                    idx = div * 4 + i
                    sns.kdeplot(data[data[columna] == valores[idx]], x=target_col, ax=ax, fill=True, alpha=0.3)
                    ax.set_xlabel(valores[idx])
                    if i != 0:
                        ax.set_ylabel('')
                    else:
                        ax.set_ylabel(f'{columna} \n p-value = {columns_calc.loc[columna]['P_value']:.3g}')

    else:
        divisiones = math.ceil(len(columnas_pintar) / 2)
        for div in range(divisiones):
            ncols = min(2, len(columnas_pintar)-div*2)
            _, axs = plt.subplots(ncols=ncols, figsize = (10*ncols, 10))
            axs = axs if isinstance(axs, np.ndarray) else [axs]
            for i, ax in enumerate(axs):
                idx = div * 2 + i
                sns.kdeplot(data, x=target_col, hue=columnas_pintar[idx], ax=ax, fill=True, alpha=0.3)
                ax.set_xlabel(f'{columnas_pintar[idx]} \n p-value = {columns_calc.loc[columnas_pintar[idx]]['P_value']:.3g}')
                ax.set_ylabel('')
    
    return columnas_pintar