import os
import sys
import main
import geopandas as gpd
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def stats_multiple_times(radius, var, time, temp):

    temp = temp.sub(temp.mean(axis=0), axis=1)
    temp = temp.div(temp.std(axis=0), axis=1)

    vars = gpd.read_parquet(f'/Users/lisawink/Documents/paper1/data/processed_data/processed_station_params_{radius}.parquet')
    vars.index = vars['station_id']
    to_remove = ['station_id','station_no','station_name','station_long_name','station_type','station_lat','station_lon','mounting_structure','sky_view_factor','dominant_land_use','local_climate_zone','urban_atlas_class','urban_atlas_code','geometry','SVF_3D']
    vars = vars.drop(to_remove, axis=1)
    vars.index = vars.index.str[2:]
    vars = vars.merge(temp, left_on='station_id', right_on='station_id',how='inner')

    vars["BuAdj"] = -vars["BuAdj"]  # Invert BuAdj values

    if var == 'BuIBD':
        vars = vars.drop(['TIEN'], axis=0)  # Remove BuAdj if var is BuIBD

    #scaler = StandardScaler()
    #vars_scaled = scaler.fit_transform(vars)
    #vars = pd.DataFrame(vars_scaled, columns=vars.columns, index=vars.index)

    data = vars[[var] + list(time)].copy().reset_index()
    data = data.melt(id_vars=[var,'station_id'], value_vars=time, var_name='time', value_name='temperature')
    data = data.dropna()

    if len(data) <= 2:
        spearman_corr = np.nan
        p_value = np.nan
        pearson_corr = np.nan
        r_squared = np.nan
        rmse = np.nan
        cooks_d = np.nan
        mi = [np.nan]
        y_pred = np.nan
        mean = np.nan
        std = np.nan
    
    else:
        mean = data[var].mean()
        std = data[var].std()
        
        # Compute Spearman correlation
        spearman_corr, p_value = spearmanr(data[var], data['temperature'])

        #Pearson and r squared
        pearson_corr, _ = pearsonr(data[var], data['temperature'])
        X = sm.add_constant(data[var])  # Add constant for regression
        model = sm.OLS(data['temperature'], X).fit()
        r_squared = model.rsquared

        # Get the predicted values (fitted values)
        y_pred = model.fittedvalues

        # Calculate the residuals (errors)
        residuals = data['temperature'] - y_pred

        # Calculate the least squares error (RSS)
        rss = np.sum(residuals ** 2)
        # Calculate the Mean Squared Error (MSE)
        mse = rss / len(data[var])
        # Calculate the Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        # Compute Cook's distance
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0].max()  # Max Cook's distance

        mi = mutual_info_regression(data[[var]], data['temperature'].values)

    return data, mean, std, spearman_corr, p_value, pearson_corr, r_squared, rmse, cooks_d, mi[0], y_pred

def define_lcz_colors():
    # Data as a list of dictionaries
    data = [
        {"LCZ": "LCZ 1", "Description": "Compact highrise", "Color": "#910613"},
        {"LCZ": "LCZ 2", "Description": "Compact midrise", "Color": "#D9081C"},
        {"LCZ": "LCZ 3", "Description": "Compact lowrise", "Color": "#FF0A22"},
        {"LCZ": "LCZ 4", "Description": "Open highrise", "Color": "#C54F1E"},
        {"LCZ": "LCZ 5", "Description": "Open midrise", "Color": "#FF6628"},
        {"LCZ": "LCZ 6", "Description": "Open lowrise", "Color": "#FF985E"},
        {"LCZ": "LCZ 7", "Description": "Lightweight low-rise", "Color": "#FDED3F"},
        {"LCZ": "LCZ 8", "Description": "Large lowrise", "Color": "#BBBBBB"},
        {"LCZ": "LCZ 9", "Description": "Sparsely built", "Color": "#FFCBAB"},
        {"LCZ": "LCZ 10", "Description": "Heavy Industry", "Color": "#565656"},
        {"LCZ": "LCZ 11 (A)", "Description": "Dense trees", "Color": "#006A18"},
        {"LCZ": "LCZ 12 (B)", "Description": "Scattered trees", "Color": "#00A926"},
        {"LCZ": "LCZ 13 (C)", "Description": "Bush, scrub", "Color": "#628432"},
        {"LCZ": "LCZ 14 (D)", "Description": "Low plants", "Color": "#B5DA7F"},
        {"LCZ": "LCZ 15 (E)", "Description": "Bare rock or paved", "Color": "#000000"},
        {"LCZ": "LCZ 16 (F)", "Description": "Bare soil or sand", "Color": "#FCF7B1"},
        {"LCZ": "LCZ 17 (G)", "Description": "Water", "Color": "#656BFA"}
        ]
    # Create DataFrame
    df = pd.DataFrame(data)
    df['LCZ_number'] = df['LCZ'].str.extract('(\d+)').astype(int)

    lcz = pd.read_csv("/Users/lisawink/Documents/paper1/data/Freiburg-Street-Level-Weather-Station-Network-MetaData-V1-0.csv")
    lcz = lcz[['station_id','local_climate_zone']]
    # extract description in brackets
    lcz['LCZ_description'] = lcz['local_climate_zone'].str.extract(r'\((.*?)\)')
    # merge with df on LCZ_description to get color and number
    lcz = lcz.merge(df, left_on='LCZ_description', right_on='Description', how='inner')

    # make a dictionary out of station_id and color
    lcz_colors_dict = dict(zip(lcz['station_id'], lcz['Color']))
    return lcz_colors_dict

def custom_lcz_legend():

        lcz_colors = {
             
        "LCZ 1: Compact highrise": "#910613",
        "LCZ 2: Compact midrise": "#D9081C",
        "LCZ 3: Compact lowrise": "#FF0A22",
        "LCZ 4: Open highrise": "#C54F1E",
        "LCZ 5: Open midrise": "#FF6628",
        "LCZ 6: Open lowrise": "#FF985E",
        "LCZ 7: Lightweight low-rise": "#FDED3F",
        "LCZ 8: Large lowrise": "#BBBBBB",
        "LCZ 9: Sparsely built": "#FFCBAB",
        "LCZ 10: Heavy Industry": "#565656",
        "LCZ 11 (A): Dense trees": "#006A18",
        "LCZ 12 (B): Scattered trees": "#00A926",
        "LCZ 13 (C): Bush, scrub": "#628432",
        "LCZ 14 (D): Low plants": "#B5DA7F",
        "LCZ 15 (E): Bare rock or paved": "#000000",
        "LCZ 16 (F): Bare soil or sand": "#FCF7B1",
        "LCZ 17 (G): Water": "#656BFA"
        }

        lcz_colors = {
             
        "LCZ 2: Compact midrise": "#D9081C",
        "LCZ 4: Open highrise": "#C54F1E",
        "LCZ 5: Open midrise": "#FF6628",
        "LCZ 6: Open lowrise": "#FF985E",
        "LCZ 8: Large lowrise": "#BBBBBB",
        "LCZ 9: Sparsely built": "#FFCBAB",
        "LCZ 11 (A): Dense trees": "#006A18",
        "LCZ 12 (B): Scattered trees": "#00A926",
        "LCZ 14 (D): Low plants": "#B5DA7F",
        "LCZ 17 (G): Water": "#656BFA"}

        lcz_labels = list(lcz_colors.keys())
        lcz_colors = list(lcz_colors.values())
        
        # Create custom legend
        custom_lines = [plt.Line2D([0], [0], marker='o', color='w', label=lcz_labels[i],
                                        markerfacecolor=lcz_colors[i], markersize=10) for i in range(len(lcz_labels))]
        return custom_lines

def simple_plot(ax, radius, var, time):

    data, spearman_corr, p_value, pearson_corr, r_squared, rmse, cooks_d, mi, y_pred = stats_multiple_times(radius, var, time)

    print(f"Pearson ρ: {pearson_corr:.2f}\n$r^2$: {r_squared:.3f}\nRMSE: {rmse:.3f}\nSpearman ρ: {spearman_corr:.2f}\nSpearman p-val: {p_value:.2f}\nMutual Info: {mi:.3f}\nMax Cook's d: {cooks_d:.3f}")

    # Add textbox with correlation and Cook’s distance
    textstr = f"$r^2$: {r_squared:.3f}\nRMSE: {rmse:.3f}\nSpearman ρ: {spearman_corr:.2f}\nSpearman p-val: {p_value:.2f}\nMax. Cook's D: {cooks_d:.3f}\nMutual Info: {mi:.3f}"
    ax.text(0.55, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='grey', facecolor='none'))

    lcz_colors = define_lcz_colors()
    colors = [lcz_colors[station] for station in data['station_id']]  # Assign colors to each station
    ax.scatter(data[var], data['temperature'], marker ='x', c=colors, alpha =0.5, label = data['station_id'])
    ax.plot(data[var], y_pred, color='black', linewidth=1)  # Plot regression
    ax.set_xlabel(var,fontsize=16)
    ax.set_ylabel('Normalised Temperature (K)',fontsize=16)
    #ax.set_title(var+' vs Temperature'+' for '+str(radius)+'m radius')

    for i, txt in enumerate(data['station_id'].unique()):
        ax.annotate(txt, (data[data['station_id'] == txt][var].iloc[0], data[data['station_id'] == txt]['temperature'].iloc[0]), color=lcz_colors[txt])

def simple_plot_reduced(ax, radius, var, time, temp):

    data, mean, std, spearman_corr, p_value, pearson_corr, r_squared, rmse, cooks_d, mi, y_pred = stats_multiple_times(radius, var, time, temp)

    print(f"Spearman ρ: {spearman_corr:.2f}\nMutual Info.: {mi:.2f}")

    # Add textbox with correlation and Cook’s distance
    textstr = (
        fr'$\rho_{{fix,300,\langle UHI\rangle}} = {spearman_corr:.2f}$' '\n'
        fr'$\mathrm{{MI}}_{{fix,300,\langle UHI\rangle}} = {mi:.2f}$'
    )
    ax.text(0.6, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='grey', facecolor='none'))

    lcz_colors = define_lcz_colors()
    colors = [lcz_colors[station] for station in data['station_id']]  # Assign colors to each station
    ax.scatter(data[var], data['temperature'], marker ='x', c=colors, alpha =0.5, label = data['station_id'])
    #ax.plot(data[var], y_pred, color='black', linewidth=1)  # Plot regression

    var_name_mapping = {
        'BuAre_sum': '$\it{A_B}$ (m$^2$)',
        'BuVol_3D_sum': '$\it{V_B}$ (m$^3$)',
        'BuEWA_3D_sum': '$\it{A_F}$ (m$^2$)',
        'BuIBD': '$\it{IBD}$ (m)', 
        'BuAdj': '$\it{Adj}$',
        'BuSWR_3D_median': '$\it{SWR}$',
        'BuHt_wmean': '$\it{H_B^A}$ (m)',
        'StrHW_median': '$\it{H/W}$',
        'SVF_3D_mean': '$\it{SVF}$',
        'BuERI_mode': '$\it{ERI}$',
        'StrClo400_median': '$\it{C}$'}

    ax.set_xlabel(var_name_mapping[var],fontsize=16)
    ax.set_ylabel('Standardised Temperature',fontsize=16)
    #ax.set_title(var+' vs Temperature'+' for '+str(radius)+'m radius')

    for i, txt in enumerate(data['station_id'].unique()):
        ax.annotate(txt, (data[data['station_id'] == txt][var].iloc[0], data[data['station_id'] == txt]['temperature'].iloc[0]), color=lcz_colors[txt])


def plot_seasons(ax, radius, var, hiwn, hispn, hisn, hian):

    data_hiwn, spearman_corr_hiwn, p_value_hiwn, pearson_corr_hiwn, r_squared_hiwn, rmse_hiwn, cooks_d_hiwn, mi_hiwn, y_pred_hiwn = stats_multiple_times(radius, var, hiwn)
    data_hispn, spearman_corr_hispn, p_value_hispn, pearson_corr_hispn, r_squared_hispn, rmse_hispn, cooks_d_hispn, mi_hispn, y_pred_hispn = stats_multiple_times(radius, var, hispn)
    data_hisn, spearman_corr_hisn, p_value_hisn, pearson_corr_hisn, r_squared_hisn, rmse_hisn, cooks_d_hisn, mi_hisn, y_pred_hisn = stats_multiple_times(radius, var, hisn)
    data_hian, spearman_corr_hian, p_value_hian, pearson_corr_hian, r_squared_hian, rmse_hian, cooks_d_hian, mi_hian, y_pred_hian = stats_multiple_times(radius, var, hian)

    # Add textbox with correlation and Cook’s distance
    textstr = f"Winter\n$r^2$: {r_squared_hiwn:.3f}\nRMSE: {rmse_hiwn:.3f}\nSpearman ρ: {spearman_corr_hiwn:.2f}\nMax Cook's D: {cooks_d_hiwn:.3f}\nMutual Info: {mi_hiwn:.3f}"
    ax.text(0.55, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='none'))
    
    textstr = f"Summer\n$r^2$: {r_squared_hisn:.3f}\nRMSE: {rmse_hisn:.3f}\nSpearman ρ: {spearman_corr_hisn:.2f}\nMax Cook's D: {cooks_d_hisn:.3f}\nMutual Info: {mi_hispn:.3f}"
    ax.text(0.05, 0.75, textstr, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='orange', facecolor='none'))
    
    textstr = f"Spring\n$r^2$: {r_squared_hispn:.3f}\nRMSE: {rmse_hispn:.3f}\nSpearman ρ: {spearman_corr_hispn:.2f}\nMax Cook's D: {cooks_d_hispn:.3f}\nMutual Info: {mi_hisn:.3f}"
    ax.text(0.55, 0.75, textstr, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='none'))
    
    textstr = f"Autumn\n$r^2$: {r_squared_hispn:.3f}\nRMSE: {rmse_hispn:.3f}\nSpearman ρ: {spearman_corr_hispn:.2f}\nMax Cook's D: {cooks_d_hispn:.3f}\nMutual Info: {mi_hian:.3f}"
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='brown', facecolor='none'))

    lcz_colors = define_lcz_colors()
    colors = [lcz_colors[station] for station in data_hisn['station_id']]  # Assign colors to each station
    scatter = ax.scatter(data_hisn[var], data_hisn['temperature'], marker ='x', c=colors, alpha =0.5, label = data_hisn['station_id'])

    ax.plot(data_hiwn[var], y_pred_hiwn, color='black', linewidth=1)  # Plot regression
    ax.plot(data_hisn[var], y_pred_hisn, color='orange', linewidth=1)  # Plot regression
    ax.plot(data_hispn[var], y_pred_hispn, color='green', linewidth=1)  # Plot regression
    ax.plot(data_hian[var], y_pred_hian, color='brown', linewidth=1)  # Plot regression

    ax.set_xlabel(var,fontsize=16)
    ax.set_ylabel('Normalised Temperature (K)',fontsize=16)
    #ax.set_title(var+' vs Temperature'+' for '+str(radius)+'m radius')

    for i, txt in enumerate(data_hiwn['station_id'][:30]):
        ax.annotate(txt, (data_hiwn[var].iloc[i], data_hiwn['temperature'].iloc[i]), color=lcz_colors[txt])
