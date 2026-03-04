POI_MAP = {
    "transporte_publico": "tp",
    "escuelas": "esc",
    "farmacias": "fca",
    "hospitales": "hosp",
    "supermercados": "super",
    "tiendas": "tda",
    "bares": "bar",
    "restaurantes": "resto",
}


DROP_COLS = ['url', 'features', 'descripcion', 'precio', 'media', 'points_of_interest',
             'energy_data', 'transporte_publico', 'escuelas', 'farmacias', 'hospitales',
             'supermercados', 'tiendas', 'bares', 'restaurantes']


COLS_OPTIMIZE = ['hosp_min_dist_m', 'super_cnt', 'realista', 'tiene_certificado',
                 'bar_min_dist_m', 'tda_min_dist_m', 'esc_min_dist_m', 'fca_cnt',
                 'calefaccion_electrica', 'eficiencia_energetica', 'resto_min_dist_m',
                 'fotografias', 'tda_cnt', 'emisiones_energeticas', 'tp_cnt',
                 'super_min_dist_m', 'aire_acondicionado', 'resto_cnt', 'tp_min_dist_m',
                 'cluster', 'fca_min_dist_m', 'categoria_ord', 'calefaccion_gas', 'bar_cnt']