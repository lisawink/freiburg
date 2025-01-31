import pandas as pd
import geopandas as gpd
import numpy as np
import momepy
import libpysal
import geoplanar
from itertools import combinations
from shapely.geometry import Point
from scipy.stats import iqr
from scipy.stats import median_abs_deviation
from scipy.stats import skew
import rasterio
from rasterstats import zonal_stats
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

def buffer_stations(stations, radius=100, input_crs='EPSG:4326', output_crs='EPSG:31468'):
    
    geometry = [Point(xy) for xy in zip(stations['station_lon'], stations['station_lat'])]
    stn_gdf = gpd.GeoDataFrame(stations, crs=input_crs, geometry=geometry)
    stn_gdf = stn_gdf.to_crs(output_crs)
    stn_gdf['geometry'] = stn_gdf.buffer(radius)
    return stn_gdf

def random_buffers(buildings, number=50, radius=100):
    """
    Generates buffers of set radius around random buildings

    """
    random_building_buffers = buildings.sample(n=number)
    random_building_buffers['geometry'] = random_building_buffers.centroid.buffer(radius)
    random_building_buffers.set_geometry('geometry', inplace=True)
    return random_building_buffers

def random_buffers(buildings, number=50, radius=100):
    """
    Generates buffers of set radius around random buildings

    """
    random_building_buffers = buildings.sample(n=number)
    random_building_buffers['geometry'] = random_building_buffers.centroid.buffer(radius)
    random_building_buffers.set_geometry('geometry', inplace=True)
    return random_building_buffers


def block_params(buildings,height,streets):

    """
    Extracts the following parameters for each building and street segment in the city:
    - dimension
    - courtyards
    - shape
    - proximity
    - streets
    - intensity
    - connectivity
    - COINS

    Parameters
    ----------
    buildings : GeoDataFrame
        GeoDataFrame containing building footprints
    height : float
        Series containing building heights
    streets : GeoDataFrame
        GeoDataFrame containing street segments

    Returns
    -------
    bldgs : GeoDataFrame
        GeoDataFrame containing building parameters
    streets : GeoDataFrame
        GeoDataFrame containing street parameters
    nodes : GeoDataFrame
        GeoDataFrame containing nodes
    edges : GeoDataFrame
        GeoDataFrame containing edges

    """

    bldgs = buildings.copy()

    # dimension
    bldgs['BuAre'] = bldgs.geometry.area
    bldgs['BuHt'] = height
    bldgs['BuPer'] = bldgs.geometry.length
    bldgs['BuLAL'] = momepy.longest_axis_length(bldgs)
    bldgs[['BuCCD_mean','BuCCD_std']] = momepy.centroid_corner_distance(bldgs)
    bldgs['BuCor'] = momepy.corners(bldgs)

    # courtyards
    bldgs['CyAre'] = momepy.courtyard_area(bldgs)
    bldgs['CyInd'] = momepy.courtyard_index(bldgs)

    # shape
    bldgs['BuCCo'] = momepy.circular_compactness(bldgs)
    bldgs['BuCWA'] = momepy.compactness_weighted_axis(bldgs, bldgs['BuLAL'])
    bldgs['BuCon'] = momepy.convexity(bldgs)
    bldgs['BuElo'] = momepy.elongation(bldgs)
    bldgs['BuERI'] = momepy.equivalent_rectangular_index(bldgs)
    bldgs['BuFR'] = momepy.facade_ratio(bldgs)
    bldgs['BuFF'] = momepy.form_factor(bldgs, height)
    bldgs['BuFD'] = momepy.fractal_dimension(bldgs)
    bldgs['BuRec'] = momepy.rectangularity(bldgs)
    bldgs['BuShI'] = momepy.shape_index(bldgs, bldgs['BuLAL'])
    bldgs['BuSqC'] = momepy.square_compactness(bldgs)
    bldgs['BuCorDev'] = momepy.squareness(bldgs)

    # proximity
    bldgs['BuSW'] = momepy.shared_walls(bldgs)
    bldgs['BuSWR'] = momepy.shared_walls(bldgs)/bldgs['BuPer']
    bldgs['BuOri'] = momepy.orientation(bldgs)

    ## building adjacency

    delaunay = libpysal.graph.Graph.build_triangulation(geoplanar.trim_overlaps(bldgs).centroid).assign_self_weight()
    bldgs['BuAli'] = momepy.alignment(bldgs['BuOri'], delaunay)

    # streets
    bldgs["street_index"] = momepy.get_nearest_street(bldgs, streets)
    streets['StrLen'] = streets.geometry.length

    # street alignment
    str_orient = momepy.orientation(streets)
    bldgs['StrAli'] = momepy.street_alignment(momepy.orientation(bldgs), str_orient, bldgs["street_index"])

    # street profile
    streets[['StrW','StrOpe','StrWD','StrH','StrHD','StrHW']] = momepy.street_profile(streets, bldgs, height=height)

    # intensity
    building_count = momepy.describe_agg(bldgs['BuAre'], bldgs["street_index"], statistics=["count"])
    streets = streets.merge(building_count, left_on=streets.index, right_on='street_index', how='left')
    streets['BpM'] = streets['count'] / streets.length

    #shape
    streets['StrLin'] = momepy.linearity(streets)

    #connectivity
 
    graph = momepy.gdf_to_nx(streets)
    graph = momepy.closeness_centrality(graph, radius=400, name="StrClo400", distance="mm_len", weight="mm_len")
    #graph = momepy.closeness_centrality(graph, radius=1200, name="StrClo1200", distance="mm_len", weight="mm_len")
    graph = momepy.betweenness_centrality(graph, radius=400, name="StrBet400", distance="mm_len", weight="mm_len")
    #graph = momepy.betweenness_centrality(graph, radius=1200, name="StrBet1200", distance="mm_len", weight="mm_len")
    graph = momepy.meshedness(graph, radius=400, distance="mm_len", name="StrMes400")
    #graph = momepy.meshedness(graph, radius=1200, distance="mm_len", name="StrMes1200")
    graph = momepy.gamma(graph, radius=400, distance="mm_len", name="StrGam400")
    #graph = momepy.gamma(graph, radius=1200, distance="mm_len", name="StrGam1200")
    graph = momepy.cyclomatic(graph, radius=400, distance="mm_len", name="StrCyc400")
    #graph = momepy.cyclomatic(graph, radius=1200, distance="mm_len", name="StrCyc1200")
    graph = momepy.edge_node_ratio(graph, radius=400, distance="mm_len", name="StrENR400")
    #graph = momepy.edge_node_ratio(graph, radius=1200, distance="mm_len", name="StrENR1200")
    graph = momepy.node_degree(graph, name='StrDeg')
    graph = momepy.clustering(graph, name='StrSCl')
    #graph = momepy.betweenness_centrality(graph, name="StrBetGlo", mode="nodes", weight="mm_len") # will take ages
    nodes, edges = momepy.nx_to_gdf(graph)

    #COINS
    coins = momepy.COINS(streets)
    stroke_gdf = coins.stroke_gdf()
    stroke_attr = coins.stroke_attribute()
    streets['COINS_index'] = stroke_attr
    streets = streets.merge(stroke_gdf, left_on='COINS_index', right_on='stroke_group')
    streets['StrCNS']=streets['geometry_y'].length

    return bldgs, streets, nodes

def neighbourhood_graph_params(buildings, stations):
    """
    
    Extracts the following parameters for each station:
    - Building adjacency
    - Interbuilding distance
        
    Parameters
    ----------
    buildings : GeoDataFrame
        GeoDataFrame containing building footprints
        stations : GeoDataFrame
        GeoDataFrame containing station buffers

    Returns
    -------
    stations : GeoDataFrame
        GeoDataFrame containing station parameters
    
    """
    
    buildings = geoplanar.trim_overlaps(buildings)
    overlapping = buildings.sjoin(stations,predicate='within',how='inner')
    
    # create libpysal graph of the buildings in overlapping for each station id
    libpysal_graphs = {}
    
    for stn_id in overlapping['station_id'].unique():

        ol_buildings = overlapping[overlapping['station_id'] == stn_id]
        
        # Generate all unique pairs of indices as adjacency list
        adjacency_list = [(i, j) for i, j in combinations(ol_buildings.index, 2)]

        # Add symmetric pairs to make it undirected (i.e., (i, j) and (j, i))
        adjacency_list += [(j, i) for i, j in adjacency_list]

        # Create a DataFrame from the adjacency list
        adjacency_df = pd.DataFrame(adjacency_list, columns=['focal', 'neighbor'])
        adjacency_df['weight'] = 1  # Assign a default weight of 1

        ref_area_graph = libpysal.graph.Graph.from_adjacency(adjacency_df)

        libpysal_graphs[stn_id] = ref_area_graph

        #calculate bua
        contig = libpysal.graph.Graph.build_contiguity(ol_buildings)
        bua = momepy.building_adjacency(contig, ref_area_graph)
        ol_buildings['BuAdj'] = bua

        #calculate ibd
        if len(ol_buildings) <= 2:
            ol_buildings['BuIBD'] = None
        else:
            delaunay = libpysal.graph.Graph.build_triangulation(ol_buildings.centroid).assign_self_weight()
            ibd = momepy.mean_interbuilding_distance(ol_buildings, delaunay, ref_area_graph)
            ol_buildings['BuIBD'] = ibd

        overlapping.loc[ol_buildings.index, ['BuAdj', 'BuIBD']] = ol_buildings[['BuAdj', 'BuIBD']]

    if 'BuAdj' not in overlapping.columns:
        overlapping['BuAdj'] = None
    if 'BuIBD' not in overlapping.columns:
        overlapping['BuIBD'] = None

    bua = overlapping.groupby('station_id')['BuAdj'].mean()
    ibd = overlapping.groupby('station_id')['BuIBD'].mean()
    stations = stations.merge(bua, left_on='station_id', right_on=bua.index)
    stations = stations.merge(ibd, left_on='station_id', right_on=ibd.index)
    
    return stations

def select_objects(buildings, streets, nodes, stations):
    """

    Selects the buildings, streets and nodes for each station


    Parameters
    ----------
    buildings : GeoDataFrame
        GeoDataFrame containing building parameters
    streets : GeoDataFrame
        GeoDataFrame containing street parameters
    nodes : GeoDataFrame
        GeoDataFrame containing node parameters
    stations : GeoDataFrame
        GeoDataFrame containing station buffers

    Returns
    -------
    df : DataFrame
        DataFrame containing aggregated parameters for each station

    """

    # select buildings whose area is at least 50% within the station buffer

    # Calculate the area of each building
    buildings['area'] = buildings.geometry.area
    streets['length'] = streets.geometry.length

    # Perform a spatial join to find buildings that intersect with station buffers
    joined_buildings = gpd.sjoin(buildings, stations, how='inner', predicate='intersects')
    joined_streets = gpd.sjoin(streets, stations, how='inner', predicate='intersects')
    joined_nodes = gpd.sjoin(nodes, stations, how='inner', predicate='intersects')

    # Ensure geodataframes have a column named geometry
    joined_buildings['geometry'] = joined_buildings.geometry
    joined_streets['geometry'] = joined_streets.geometry
    joined_nodes['geometry'] = joined_nodes.geometry

    # Calculate the intersection area for each building-station pair
    intersection_area = joined_buildings.apply(
        lambda row: row.geometry.intersection(stations.loc[row['index_right']].geometry).area, axis=1
    )
    if intersection_area.empty: 
        joined_buildings['intersection_area'] = None
    else:
        joined_buildings['intersection_area'] = intersection_area

    # Calculate the intersection area for each building-station pair
    intersection_length = joined_streets.apply(
        lambda row: row.geometry.intersection(stations.loc[row['index_right']].geometry).length, axis=1
    )
    if intersection_length.empty:
        joined_streets['intersection_length'] = None
    else:
        joined_streets['intersection_length'] = intersection_length

    # Calculate the percentage of each building's area that is within each station buffer
    joined_buildings['percentage_within_buffer'] = (joined_buildings['intersection_area'] / joined_buildings.geometry.area) * 100
    joined_streets['percentage_within_buffer'] = (joined_streets['intersection_length'] / joined_streets.geometry.length) * 100

    # Select buildings where this percentage is at least 50%
    selected_buildings = joined_buildings[joined_buildings['percentage_within_buffer'] >= 50]
    selected_streets = joined_streets[joined_streets['percentage_within_buffer'] >= 50]
    selected_nodes = joined_nodes

    # Output the selected buildings
    selected_buildings = selected_buildings.drop(columns=['index_right'])
    selected_streets = selected_streets.drop(columns=['index_right'])
    
    return selected_buildings, selected_streets, selected_nodes

def weighted_stats(group, i, weight):

    weighted_mean = np.sum(group[i] * group[weight]) / np.sum(group[weight])
    weighted_std = np.sqrt(np.sum(group[weight] * (group[i] - weighted_mean) ** 2) / np.sum(group[weight]))

    # weighted median
    sorted_group = group.sort_values(i)
    cumulative_weight = sorted_group[weight].cumsum()
    cutoff = sorted_group[weight].sum() / 2
    weighted_median = sorted_group.loc[cumulative_weight >= cutoff, i].iloc[0]

    # Weighted minimum and maximum
    weighted_min = sorted_group.loc[sorted_group[weight].idxmin(), i]
    weighted_max = sorted_group.loc[sorted_group[weight].idxmax(), i]

    # Weighted sum
    weighted_sum = np.sum(group[i] * group[weight])

    # Weighted mode (most frequently occurring value by weight)
    mode_idx = group.groupby(i)[weight].sum().idxmax()
    weighted_mode = mode_idx

    # Weighted 25th and 75th percentiles
    q25_cutoff = sorted_group[weight].sum() * 0.25
    q75_cutoff = sorted_group[weight].sum() * 0.75

    weighted_q25 = sorted_group.loc[cumulative_weight >= q25_cutoff, i].iloc[0]
    weighted_q75 = sorted_group.loc[cumulative_weight >= q75_cutoff, i].iloc[0]

    return pd.Series({
        'weighted_mean': weighted_mean,
        'weighted_std': weighted_std,
        'weighted_median': weighted_median,
        'weighted_min': weighted_min,
        'weighted_max': weighted_max,
        'weighted_sum': weighted_sum,
        'weighted_mode': weighted_mode,
        'weighted_25th_percentile': weighted_q25,
        'weighted_75th_percentile': weighted_q75,
    })


def aggregate_params(selected_buildings, selected_streets, selected_nodes, stations, weight='BuAre'):
    
    #station_df = pd.DataFrame()
    df = pd.DataFrame()
    for i in ['BuAre','BuHt','BuPer','BuLAL','BuCCD_mean','BuCCD_std','BuCor','CyAre','CyInd','BuCCo','BuCWA','BuCon','BuElo','BuERI','BuFR','BuFF','BuFD','BuRec','BuShI','BuSqC','BuCorDev','BuSWR','BuOri','BuAli','StrAli',
              'BuCir', 'BuHem_3D', 'BuCon_3D', 'BuFra', 'BuFra_3D', 'BuCubo_3D', 'BuSqu', 'BuCube_3D', 'BumVE_3D', 'BuMVE_3D', 'BuFF_3D', 'BuEPI_3D', 'BuProx', 'BuProx_3D', 'BuEx', 'BuEx_3D', 'BuSpi', 'BuSpi_3D', 'BuPerC', 
              'BuCf_3D', 'BuDep', 'BuDep_3D', 'BuGir', 'BuGir_3D', 'BuDisp', 'BuDisp_3D', 'BuRan', 'BuRan_3D', 'BuRough', 'BuRough_3D', 'BuSWA_3D', 'BuSurf_3D', 'BuVol_3D', 'BuSA_3D', 'BuSWR_3D']:
        df[[i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum',i+'_mode']] = momepy.describe_agg(selected_buildings[i], selected_buildings["station_id"], statistics=["mean", "median", "std", "min", "max", "sum", "mode"])
        if i != 'BuAre':    
            df[[i+'_wmean',i+'_wstd',i+'_wmedian',i+'_wmin',i+'_wmax',i+'_wsum',i+'_wmode',i+'_wper25',i+'_wper75']] = selected_buildings.groupby('station_id')[[i,weight]].apply(weighted_stats, i, weight)
        df[[i+'_IQR',i+'_MAD',i+'_skew']] = selected_buildings.groupby('station_id')[i].agg([iqr,median_abs_deviation,skew])
        df[[i+'_per25',i+'_per75']] = selected_buildings.groupby('station_id')[i].quantile([0.25,0.75]).unstack()
        df['BuNum'] = len(selected_buildings.groupby('station_id'))

    for i in ['StrLen', 'StrW', 'StrOpe', 'StrWD', 'StrH', 'StrHD', 'StrHW', 'BpM', 'StrLin', 'StrCNS']:
        df[[i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum' ,i+'_mode']] = momepy.describe_agg(selected_streets[i], selected_streets["station_id"], statistics=["mean", "median", "std", "min", "max", "sum", "mode"])
        df[[i+'_IQR',i+'_MAD',i+'_skew']] = selected_streets.groupby('station_id')[i].agg([iqr,median_abs_deviation,skew])
        df[[i+'_per25',i+'_per75']] = selected_streets.groupby('station_id')[i].quantile([0.25,0.75]).unstack()

    for i in ['StrClo400', 'StrBet400', 'StrMes400', 'StrGam400', 'StrCyc400', 'StrENR400', 'StrDeg', 'StrSCl']:
        df[[i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum' ,i+'_mode']] = momepy.describe_agg(selected_nodes[i], selected_nodes["station_id"], statistics=["mean", "median", "std", "min", "max", "sum", "mode"])
        df[[i+'_IQR',i+'_MAD',i+'_skew']] = selected_nodes.groupby('station_id')[i].agg([iqr,median_abs_deviation,skew])
        df[[i+'_per25',i+'_per75']] = selected_nodes.groupby('station_id')[i].quantile([0.25,0.75]).unstack()

    stations = stations.merge(df, left_on='station_id', right_on=df.index, how='inner')
    stations['BuCAR'] = stations['BuAre_sum']/stations.geometry.area

    return stations

def agg_raster(raster_path, stations, parameter_name):
    """
    Extracts the following parameters for each station:
    - Raster mean
    - Raster median
    - Raster standard deviation
    - Raster IQR

    Parameters
    ----------
    raster_path : ndarray
        Raster path
    stations : GeoDataFrame
        GeoDataFrame containing station buffers
    parameter_name : str
        Name of the raster parameter
        
    Returns
    -------
    stations : GeoDataFrame
        GeoDataFrame containing raster parameters

    """
    with rasterio.open(raster_path) as src:
        crs = src.crs
    stations = stations.to_crs(crs.to_epsg())

    if stations.empty:
        stations[parameter_name+'_mean'] = None
        stations[parameter_name+'_std'] = None
        stations[parameter_name+'_median'] = None
        stations[parameter_name+'_IQR'] = None

    else:
        stats = zonal_stats(stations, raster_path, stats=['mean', 'max', 'min', 'count', 'std', 'median', 'sum', 'range','percentile_25','percentile_75'])

        stations[parameter_name] = stats

        stations[parameter_name+'_mean'] = stations[parameter_name].apply(lambda x: x['mean'])
        stations[parameter_name+'_std'] = stations[parameter_name].apply(lambda x: x['std'])
        stations[parameter_name+'_median'] = stations[parameter_name].apply(lambda x: x['median'])
        stations[parameter_name+'_IQR'] = stations[parameter_name].apply(lambda x: x['percentile_75']) - stations[parameter_name].apply(lambda x: x['percentile_25'])

    return stations


# Function to calculate correlations and mutual information
def calculate_statistics(data, target_column, bootstrap = False):
    results = []
    
    # Ensure the target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Loop through each column except the target column
    for col in data.columns:
        if col == target_column:
            continue

        #print(f"Calculating statistics for '{col}'...")

        # Drop NA values for pairwise comparison
        valid_data = data[[col, target_column]].dropna()

        if len(valid_data) <= 2:
            # Append results
            results.append({
                'Parameter': col,
                'Pearson Correlation': None,
                'Pearson p-value': None,
                'Spearman Correlation': None,
                'Spearman p-value': None,
                'Mutual Information': None
            })
        else:

            x = valid_data[col]
            y = valid_data[target_column]

            # Calculate Pearson correlation
            pearson_corr, pearson_pval = pearsonr(x, y)

            # Calculate Spearman's rank correlation
            spearman_corr, spearman_pval = spearmanr(x, y)
            if bootstrap:
                bs = bootstrap(spearmanr, x, y)
            else:
                bs = {}

            # Calculate mutual information
            if len(x) < 4:
                mi = None
            else:
                mi = mutual_info_regression(x.values.reshape(-1, 1), y)[0]

            # Append results
            results.append({
                'Parameter': col,
                'Pearson Correlation': pearson_corr,
                'Pearson p-value': pearson_pval,
                'Spearman Correlation': spearman_corr,
                'Spearman p-value': spearman_pval,
                'Mutual Information': mi
            }.update(bs))

    return pd.DataFrame(results)

def _ci_index(data):
    lower = np.percentile(data, 2.5)
    upper = np.percentile(data, 97.5)

    index = (upper - lower) / (upper + lower)/2
    return index

def bootstrap(func, X, Y, n_bootstrap=1000):

    # Step 1: Calculate observed  correlation
    rho_obs, _ = func(X, Y)

    # Step 2: Bootstrap  correlation
    bootstrap_corr = []

    for _ in range(n_bootstrap):
        # Resample data with replacement
        indices = np.random.choice(len(X), len(X), replace=True)
        X_bootstrap = X[indices]
        Y_bootstrap = Y[indices]
        rho_boot, _ = spearmanr(X_bootstrap, Y_bootstrap)
        bootstrap_corr.append(rho_boot)

    # Step 3: Null distribution by permuting Y
    n_null = n_bootstrap
    null_corr = []

    for _ in range(n_null):
        Y_permuted = np.random.permutation(Y)  # Shuffle Y
        rho_null, _ = spearmanr(X, Y_permuted)
        null_corr.append(rho_null)

    # Step 4: Calculate p-values
    bootstrap_corr = np.array(bootstrap_corr)
    null_corr = np.array(null_corr)

    # Bootstrap p-value
    p_bootstrap = np.mean(bootstrap_corr >= rho_obs)

    # Null p-value
    p_null = np.mean(null_corr >= rho_obs)

    # Output results
    results = {
        "Observed correlation": rho_obs,
        "Mean correlation (bootstrap)": np.mean(bootstrap_corr),
        "Standard deviation (bootstrap)": np.std(bootstrap_corr),
        "95% confidence interval (bootstrap)": np.percentile(bootstrap_corr, [2.5, 97.5]),
        "95% confidence interval index (bootstrap)": _ci_index(bootstrap_corr),
        "P-value (bootstrap)": p_bootstrap,
        "P-value (null)": p_null
    }

    return results

def plot(radius, param, temp, time):
    vars = gpd.read_parquet(f'/Users/lisawink/Documents/paper1/data/processed_data/processed_station_params_{radius}.parquet')
    vars.index = vars['station_id']
    vars = vars.merge(temp[time], left_on='station_id', right_on='station_id',how='inner')

    var = param

    plt.scatter(vars[var], vars[time])
    plt.xlabel(var)
    plt.ylabel('Temperature')
    plt.title(var+' vs Temperature')

    for i, txt in enumerate(vars.index):
        plt.annotate(txt, (vars[var][i], vars[time][i]))
    plt.show()