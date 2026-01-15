





































coef_records = {}

revenues = []


windows = [1, 2, 4, 8, 16, 32, 64, 128, 256]
for i in range(len(windows)):
    windows[i]*= BACK_WINDOW


# regression stats
    # Residual difference between short and long
    # Residual mean/std

    # Residual z-score all
    # Residual z-score selected
    # Residual sign

    # corr all
    # corr selected

    # coef adjusted mean/std

stats_global = {
    "residual_diff": [], #
    "residual_std": [], #
    "residual_mean": [], #
    "residual_z_all": [],
    "residual_z_selected": [],
    "residual_sign": [],
    "corr_all": [],
    "corr_selected": [],
}


# stats by windows
    # growth diff all
    # growth diff selected
    # growth diff sign

    # Amihud all
    # Amihud selected

    # CMI all
    # CMI selected

    # coin score all
    # coin score selected

    # ATR all
    # ATR selected

    # Bollinger Bands all
    # Bollinger Bands selected

    # Volatility all
    # Volatility selected

stats_by_window = {
    "growth_diff_all": {w: [] for w in windows},
    "growth_diff_selected": {w: [] for w in windows},
    "growth_diff_sign": {w: [] for w in windows},

    "amihud_all": {w: [] for w in windows},
    "amihud_selected": {w: [] for w in windows},

    "cmi_all": {w: [] for w in windows},
    "cmi_selected": {w: [] for w in windows},

    "coin_all": {w: [] for w in windows},
    "coin_selected": {w: [] for w in windows},

    "atr_all": {w: [] for w in windows},
    "atr_selected": {w: [] for w in windows},

    "bb_all": {w: [] for w in windows},
    "bb_selected": {w: [] for w in windows},

    "vol_all": {w: [] for w in windows},
    "vol_selected": {w: [] for w in windows},
}

