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


COLS_OPTIMIZE = ['realista', 'tiene_certificado', 'aire_acondicionado', 'cluster', 'resto_cnt',
                 'tda_min_dist_m', 'fca_cnt', 'super_min_dist_m', 'bar_min_dist_m', 'esc_min_dist_m',
                 'tda_cnt', 'ascensor', 'emisiones_energeticas', 'bar_cnt', 'calefaccion_electrica',
                 'eficiencia_energetica', 'super_cnt', 'planos', 'tp_min_dist_m', 'clase_energetica_ord',
                 'esc_cnt', 'hosp_cnt', 'categoria_ord', 'fotografias', 'resto_min_dist_m', 'calefaccion_gas',
                 'tp_cnt']