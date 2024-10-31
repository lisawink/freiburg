import pandas as pd
import geopandas as gpd
import momepy
from libpysal import graph


def block_params(buildings,height,streets):

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

    delaunay = graph.Graph.build_triangulation(bldgs.centroid).assign_self_weight()
    bldgs['AliOri'] = momepy.alignment(bldgs['BuOri'], delaunay)

    # streets
    bldgs["street_index"] = momepy.get_nearest_street(bldgs, streets)

    # street alignment
    str_orient = momepy.orientation(streets)
    bldgs['StrAli'] = momepy.street_alignment(momepy.orientation(bldgs), str_orient, bldgs["street_index"])

    # street profile
    streets[['StrW','StrOp','StrWD','StrH','StrHD','StrHW']] = momepy.street_profile(streets, bldgs, height=height)

    # intensity
    ######### NEED TO FIX THIS
    #streets['BpM'] = momepy.describe_agg(bldgs['BuAre'], bldgs["street_index"], statistics=["count"]) / streets.length

    #shape

    streets['StrLin'] = momepy.linearity(streets)

    return bldgs, streets

def aggregate_block_params(buildings, streets, stations, radius=100):

    # select buildings whose area is at least 50% within the station buffer

    # Calculate the area of each building
    buildings['area'] = buildings.geometry.area
    streets['length'] = streets.geometry.length

    # Perform a spatial join to find buildings that intersect with station buffers
    joined_buildings = gpd.sjoin(buildings, stations, how='inner', predicate='intersects')
    joined_streets = gpd.sjoin(streets, stations, how='inner', predicate='intersects')

    # Calculate the intersection area for each building-station pair
    joined_buildings['intersection_area'] = joined_buildings.apply(
        lambda row: row.geometry.intersection(stations.loc[row['index_right']].geometry).area, axis=1
    )
    # Calculate the intersection area for each building-station pair
    joined_streets['intersection_length'] = joined_streets.apply(
        lambda row: row.geometry.intersection(stations.loc[row['index_right']].geometry).length, axis=1
    )

    # Calculate the percentage of each building's area that is within each station buffer
    joined_buildings['percentage_within_buffer'] = (joined_buildings['intersection_area'] / joined_buildings['area']) * 100
    joined_streets['percentage_within_buffer'] = (joined_streets['intersection_length'] / joined_streets['length']) * 100

    # Select buildings where this percentage is at least 50%
    selected_buildings = joined_buildings[joined_buildings['percentage_within_buffer'] >= 50]
    selected_streets = joined_streets[joined_streets['percentage_within_buffer'] >= 50]

    # Output the selected buildings
    selected_buildings = selected_buildings.drop(columns=['index_right'])
    selected_streets = selected_streets.drop(columns=['index_right'])
    
    station_df = pd.DataFrame()
    df = pd.DataFrame()
    for i in ['BuAre','BuPer','BuLAL','BuCCD_mean','BuCCD_std','BuCor','CyAre','CyInd','BuCCo','BuCWA','BuCon','BuElo','BuERI','BuFR','BuFF','BuFD','BuRec','BuShI','BuSqC','BuSqu','SWR','BuOri','AliOri','StrAli']:
        buildings[i] = buildings[i].astype(float)
        df[[i+'_count',i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum' ,i+'_nunique',i+'_mode']] = momepy.describe_agg(selected_buildings[i], selected_buildings["station_id"])
        station_df = pd.concat([station_df, df], axis=1)

    for i in ['StrW','StrOp','StrWD','StrH','StrHD','StrHW','StrLin']:
        df[[i+'_count',i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum' ,i+'_nunique',i+'_mode']] = momepy.describe_agg(selected_streets[i], selected_streets["station_id"])
        station_df = pd.concat([station_df, df], axis=1)

    return station_df

def aggregate_neighborhood_params():
    return None