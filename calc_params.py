import pandas as pd
import geopandas as gpd
import momepy
import libpysal
import geoplanar


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
    bldgs['BuSqu'] = momepy.squareness(bldgs)

    # proximity
    bldgs['SWR'] = momepy.shared_walls(bldgs)/bldgs['BuPer']
    bldgs['BuOri'] = momepy.orientation(bldgs)

    ## building adjacency

    delaunay = libpysal.graph.Graph.build_triangulation(geoplanar.trim_overlaps(bldgs).centroid).assign_self_weight()
    bldgs['AliOri'] = momepy.alignment(bldgs['BuOri'], delaunay)

    # streets
    bldgs["street_index"] = momepy.get_nearest_street(bldgs, streets)
    streets['StrLen'] = streets.geometry.length

    # street alignment
    str_orient = momepy.orientation(streets)
    bldgs['StrAli'] = momepy.street_alignment(momepy.orientation(bldgs), str_orient, bldgs["street_index"])

    # street profile
    streets[['StrW','StrOp','StrWD','StrH','StrHD','StrHW']] = momepy.street_profile(streets, bldgs, height=height)

    # intensity
    building_count = momepy.describe_agg(bldgs['BuAre'], bldgs["street_index"], statistics=["count"])
    streets = streets.merge(building_count, left_on=streets.index, right_on='street_index', how='left')
    streets['BpM'] = streets['count'] / streets.length

    #shape
    streets['StrLin'] = momepy.linearity(streets)

    #connectivity
 
    graph = momepy.gdf_to_nx(streets)
    graph = momepy.closeness_centrality(graph, radius=400, name="StrClo400", distance="mm_len", weight="mm_len")
    graph = momepy.closeness_centrality(graph, radius=1200, name="StrClo1200", distance="mm_len", weight="mm_len")
    graph = momepy.betweenness_centrality(graph, radius=400, name="StrBet400", distance="mm_len", weight="mm_len")
    graph = momepy.betweenness_centrality(graph, radius=1200, name="Strbet1200", distance="mm_len", weight="mm_len")
    graph = momepy.meshedness(graph, radius=400, distance="mm_len", name="StrMes400")
    graph = momepy.meshedness(graph, radius=1200, distance="mm_len", name="StrMes1200")
    graph = momepy.gamma(graph, radius=400, distance="mm_len", name="StrGam400")
    graph = momepy.gamma(graph, radius=1200, distance="mm_len", name="StrGam1200")
    graph = momepy.cyclomatic(graph, radius=400, distance="mm_len", name="StrCyc400")
    graph = momepy.cyclomatic(graph, radius=1200, distance="mm_len", name="StrCyc1200")
    graph = momepy.edge_node_ratio(graph, radius=400, distance="mm_len", name="StrENR400")
    graph = momepy.edge_node_ratio(graph, radius=1200, distance="mm_len", name="StrENR1200")
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
    streets['Str_CNS']=streets['geometry_y'].length

    return bldgs, streets, nodes

def aggregate_block_params(buildings, streets, nodes, stations, radius=100):

    # select buildings whose area is at least 50% within the station buffer

    # Calculate the area of each building
    buildings['area'] = buildings.geometry.area
    streets['length'] = streets.geometry.length

    # Perform a spatial join to find buildings that intersect with station buffers
    joined_buildings = gpd.sjoin(buildings, stations, how='inner', predicate='intersects')
    joined_streets = gpd.sjoin(streets, stations, how='inner', predicate='intersects')
    joined_nodes = gpd.sjoin(nodes, stations, how='inner', predicate='intersects')

    print(joined_buildings.geometry)

    # Ensure geodataframes have a column named geometry
    joined_buildings['geometry'] = joined_buildings.geometry
    joined_streets['geometry'] = joined_streets.geometry
    joined_nodes['geometry'] = joined_nodes.geometry

    # Calculate the intersection area for each building-station pair
    joined_buildings['intersection_area'] = joined_buildings.apply(
        lambda row: row.geometry.intersection(stations.loc[row['index_right']].geometry).area, axis=1
    )
    # Calculate the intersection area for each building-station pair
    joined_streets['intersection_length'] = joined_streets.apply(
        lambda row: row.geometry.intersection(stations.loc[row['index_right']].geometry).length, axis=1
    )

    # Calculate the intersection area for each building-station pair
    joined_nodes['intersection_length'] = joined_nodes.apply(
        lambda row: row.geometry.intersection(stations.loc[row['index_right']].geometry).length, axis=1
    )

    # Calculate the percentage of each building's area that is within each station buffer
    joined_buildings['percentage_within_buffer'] = (joined_buildings['intersection_area'] / joined_buildings['area']) * 100
    joined_streets['percentage_within_buffer'] = (joined_streets['intersection_length'] / joined_streets['length']) * 100

    # Select buildings where this percentage is at least 50%
    selected_buildings = joined_buildings[joined_buildings['percentage_within_buffer'] >= 50]
    selected_streets = joined_streets[joined_streets['percentage_within_buffer'] >= 50]
    selected_nodes = joined_nodes

    # Output the selected buildings
    selected_buildings = selected_buildings.drop(columns=['index_right'])
    selected_streets = selected_streets.drop(columns=['index_right'])
    
    #station_df = pd.DataFrame()
    df = pd.DataFrame()
    for i in ['BuAre','BuPer','BuLAL','BuCCD_mean','BuCCD_std','BuCor','CyAre','CyInd','BuCCo','BuCWA','BuCon','BuElo','BuERI','BuFR','BuFF','BuFD','BuRec','BuShI','BuSqC','BuSqu','SWR','BuOri','AliOri','StrAli']:
        buildings[i] = buildings[i].astype(float)
        df[[i+'_count',i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum' ,i+'_nunique',i+'_mode']] = momepy.describe_agg(selected_buildings[i], selected_buildings["station_id"])
        #station_df = pd.concat([station_df, df], axis=1)

    for i in ['StrLen', 'StrW', 'StrOp', 'StrWD', 'StrH', 'StrHD', 'StrHW', 'BpM', 'StrLin', 'Str_CNS']:
        df[[i+'_count',i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum' ,i+'_nunique',i+'_mode']] = momepy.describe_agg(selected_streets[i], selected_streets["station_id"])
        #station_df = pd.concat([station_df, df], axis=1)

    for i in ['StrClo400', 'StrClo1200', 'StrBet400', 'Strbet1200', 'StrMes400', 'StrMes1200', 'StrGam400', 'StrGam1200', 'StrCyc400', 'StrCyc1200', 'StrENR400', 'StrENR1200', 'StrDeg', 'StrSCl']:
        df[[i+'_count',i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum' ,i+'_nunique',i+'_mode']] = momepy.describe_agg(selected_nodes[i], selected_nodes["station_id"])
        #station_df = pd.concat([station_df, df], axis=1)

    return df