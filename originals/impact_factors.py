# Nature Impact Factors
# 08/22/2024
# Running models
# updating the code to be able to swap out any crop


import logging
import numpy as np
import os
import pygeoprocessing as pygeo
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from osgeo import gdalconst
from osgeo import gdalnumeric
from osgeo import ogr
from matplotlib import pyplot as plt
import multiprocessing as mp
import hazelbean as hb
import ineqpy
import rasterio
from rasterio.mask import mask
import re
import requests
import sys
import time
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from shapely.geometry import Polygon, LineString, box
from shapely.ops import split
from shapely import LineString
import statsmodels.api as sm
from itertools import combinations
import natcap.invest.carbon
import natcap.invest.utils

# from carbon_new import *
# from biodiversity import *

os.chdir("C:/Users/kibby/OneDrive/Desktop/Research/Nature Impact Factors/")

# I just want to replace the LULC pixel where the ' + crop + '_all_tech production is >0?
country_list = [
    "China",
    "India",
    "United States",
]  # I really only want the contiguous U.S.

crop = "cott"  # starting with cotton

# This is not being passed to the function below, right?
raster_paths = [
    "data/LULC/lulc_esa_2020.tif",
    "data/scenario_construction/potential_vegetation/potential_vegetation.tif",
    "data/Carbon/carbon_zones_rasterized.tif",
    "data/biodiversity/GlobalLayers/Amphibians.tif",  # Isn't this for biodiversity?
    "data/Crop Production/harvested_ha/" + crop + "_all_tech.tif",
]


country_boundaries = gpd.read_file("data/Polygon_vectors/ee_r264_correspondence.gpkg")


# Shouldn't this function return something?
def clip_raster_to_country(c, raster_paths):
    if c == "United States":
        country_boundary = gpd.read_file(
            "data/Polygon_vectors/United States.gpkg"
        )  # Clipped to the contiguous U.S. in QGIS
    else:
        country_boundary = country_boundaries[country_boundaries["ee_r264_name"] == c]
        country_boundary.to_file("data/Polygon_vectors/" + c + ".gpkg")
    for ras in raster_paths:  # We extract the country pixels from the global rasters?
        raster_dataset = gdal.Open(ras)
        gdal.Warp(
            re.sub(r"\.(.*?)$", r"_clipped_{}.\1".format(c), ras),
            raster_dataset,
            cutlineDSName="data/Polygon_vectors/" + c + ".gpkg",
            cropToCutline=True,
        )


# for c in country_list:
#     clip_raster_to_country(c)

# clip_raster_to_country('United States', raster_paths)


### REPLACING THE LULC WITH THE POTENTIAL VEG IN THE AREAS WHERE THERE IS CROP PRODUCTION ###

# def modify_classification(clipped_lulc, clipped_potential_veg, crop_raster):
#     # Replace values in clipped_lulc with values from clipped_potential_veg
#     # where crop_raster > 0, otherwise keep clipped_lulc values.
#     result = np.where(crop_raster > 0, clipped_potential_veg, clipped_lulc)
#     return result

# # Define the operation in the raster calculator
# for c in country_list:
#     clipped_lulc_path = 'data/LULC/lulc_esa_2020_clipped_' + c + '.tif'
#     clipped_potential_veg = 'data/scenario_construction/potential_vegetation/potential_vegetation_clipped_' + c + '.tif'
#     clipped_lulc_without_ag = 'data/LULC/lulc_without_ag_esa_2020_clipped_' + c + '.tif'
#     crop_raster_path = 'data/Crop Production/harvested_ha/' + crop + '_all_tech_clipped_' + c + '.tif'
#     # Define paths for the input rasters
#     base_raster_path_list = [
#         clipped_lulc_path,
#         clipped_potential_veg,
#         crop_raster_path
#     ]

#     # Define paths for the output aligned/resampled rasters
#     target_raster_path_list = [
#         'data/LULC/aligned_lulc_' + c + '.tif',
#         'data/scenario_construction/potential_vegetation/aligned_veg_' + c + '.tif',
#         'data/Crop Production/harvested_ha/aligned_cotton_' + c + '.tif'
#     ]

#     # Define the resampling method for each raster ('near' for categorical, 'bilinear' for continuous)
#     resample_method_list = ['near', 'near', 'bilinear']

#     # Use the pixel size of the highest resolution raster (clipped_lulc) as the target pixel size
#     # Get the pixel size of the LULC raster
#     lulc_info = pygeo.get_raster_info(clipped_lulc_path)
#     target_pixel_size = lulc_info['pixel_size']  # Example: (x_size, y_size)

#     # Align and resize rasters
#     pygeo.align_and_resize_raster_stack(
#         base_raster_path_list,
#         target_raster_path_list,
#         resample_method_list,
#         target_pixel_size,
#         bounding_box_mode='intersection'  # You can use 'union' if needed
#     )

#     pygeo.raster_calculator(
#         [(target_raster_path_list[0], 1), (target_raster_path_list[1], 1), (target_raster_path_list[2], 1)],  # Add cotton raster
#         modify_classification,  # Local operation
#         clipped_lulc_without_ag,  # Output raster path
#         gdal.GDT_Byte,  # Output data type
#         None  # Nodata value (use None if not specified)
#     )

#     pygeo.reclassify_raster((clipped_lulc_without_ag, 1),
#                             {0:0, 10:10, 11:11, 12:12, 13:13, 20:20,
#                             30:30, 40:40, 50:50, 60:60, 61:61, 62:62, 70:70, 71:71, 72:72,
#                              80:80,  81:81, 90:90, 100:100, 110:110, 120:120, 121:121, 122:122,
#                              130:130, 140:140, 150:150, 151:151, 152:152, 153:153, 160:160,
#                              170:170, 180:180, 190:190, 200:200, 201:201, 202:202, 210:210, 220:220,
#                              255:0},
#                             'data/LULC/lulc_without_ag_esa_2020_clipped_' + c + '_v02.tif',
#                             gdal.GDT_Byte,
#                             0, values_required=True)

#     # clip_raster_to_country(c, ['Results/carbon_new/' + c + '/lulc_without_ag_esa_2020_clipped_' + c + '_v02_carbon.tif'])

for c in country_list:
    # path = 'Results/carbon_new/' + c + '/lulc_without_ag_esa_2020_clipped_' + c + '_v02_carbon.tif'
    # Is this raster generated by the carbon_new.py script?
    path = "Results/carbon_new/" + c + "/aligned_lulc_" + c + "_al_v02_carbon.tif"

    if os.path.exists(path):
        clip_raster_to_country(c, [path])  # Is anything returned?
        os.remove(path)

    if (
        os.path.exists(re.sub(".tif", "_compress.tif", path)) == 0
    ):  # What does this check?
        # Open the input raster
        input_raster_path = re.sub(
            r"\.(.*?)$", r"_clipped_{}.\1".format(c), path
        )  # What does the regex do?
        # 'Results/carbon_new/' + c + '/lulc_without_ag_esa_2020_clipped_' + c + '_v02_carbon_clipped_' + c + '.tif'
        output_raster_path = re.sub(".tif", "_compress.tif", path)

        # Open the dataset
        src_ds = gdal.Open(input_raster_path)

        # Define the output options, including compression
        creation_options = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]

        # Use gdal.Translate to apply the compression
        gdal.Translate(output_raster_path, src_ds, creationOptions=creation_options)

        # Close the dataset
        src_ds = None

        os.remove(input_raster_path)
        print(f"Compressed raster saved to {output_raster_path}")


#### SUMMARIZING THE CARBON RESULTS ####


def sub_rasters(lulc_without_ag, lulc):
    result = np.where(
        (lulc_without_ag != 255) & (lulc != 255), lulc_without_ag - lulc, 0
    )
    # result = lulc_without_ag - lulc
    return result


# for c in country_list:
#     base_raster_list = [('Results/carbon_new/' + c + '/lulc_without_ag_esa_2020_clipped_' + c + '_v02_carbon_compress.tif', 1),
#                         ('Results/carbon_new/' + c + '/aligned_lulc_' + c + '_al_v02_carbon_compress.tif', 1)]
#     pygeo.raster_calculator(base_raster_list,
#                             local_op=sub_rasters,
#                             target_raster_path = 'Results/carbon_new/' + c + '/lulc_diff_2020_' + c + '.tif',
#                             datatype_target=gdal.GDT_Float32,
#                             nodata_target=None

# )


#### SUMMARIZING BY COUNTRY ####
# columns = ['Country', 'Real carbon stored', 'Carbon stored in natural vegetation scenario', 'Difference between natural veg. and real']
# df = pd.DataFrame(columns=columns)
# for c in country_list:
#     carbon_path = 'Results/carbon_new/' + c + '/aligned_lulc_' + c + '_al_v02_carbon_compress.tif'
#     country_boundary = "data/Polygon_vectors/" + c + ".gpkg"

#     carbon_stats = pygeo.zonal_statistics(base_raster_path_band = (carbon_path, 1),
#                                                         aggregate_vector_path = country_boundary,
#                                                         ignore_nodata = True)
#     tot_carbon = carbon_stats[1]['sum']

#     veg_path = 'Results/carbon_new/' + c + '/lulc_without_ag_esa_2020_clipped_' + c + '_v02_carbon_compress.tif'
#     veg_stats = pygeo.zonal_statistics(base_raster_path_band = (veg_path, 1),
#                                                         aggregate_vector_path = country_boundary,
#                                                         ignore_nodata = True)
#     veg_carbon = veg_stats[1]['sum']


#     diff_path = 'Results/carbon_new/' + c + '/lulc_diff_2020_' + c + '.tif'
#     diff_stats = pygeo.zonal_statistics(base_raster_path_band = (diff_path, 1),
#                                                         aggregate_vector_path = country_boundary,
#                                                         ignore_nodata = True)
#     diff_carbon = diff_stats[1]['sum']

#     row = [c, tot_carbon, veg_carbon, diff_carbon]

#     row_df = pd.DataFrame([row], columns=columns)
#     df = pd.concat([df, row_df])

# df.to_csv('Results/carbon_new/country_results.csv')

#### SUMMARIZING RESULTS BY STATE ####

# for c in country_list:
#     tot_carbon = {}
#     veg_carbon = {}
#     diff_carbon = {}
#     if c == 'United States':
#         state_boundaries = "data/Polygon_vectors/tl_2019_us_state/tl_2019_us_state.shp"
#         states_gdf = gpd.read_file(state_boundaries)
#         states = states_gdf['NAME'].to_list()
#     else:
#         abbr = {'China': 'CHN',
#                 'India': 'IND'}
#         state_boundaries = "data/Polygon_vectors/States/geoBoundaries-" + abbr[c] + "-ADM1_simplified.geojson"
#         states_gdf = gpd.read_file(state_boundaries)
#         states = states_gdf['shapeName'].to_list()

#     carbon_path = 'Results/carbon_new/' + c + '/aligned_lulc_' + c + '_al_v02_carbon_compress.tif'
#     carbon_stats = pygeo.zonal_statistics(base_raster_path_band = (carbon_path, 1),
#                                                         aggregate_vector_path = state_boundaries,
#                                                         ignore_nodata = True)

#     veg_path = 'Results/carbon_new/' + c + '/lulc_without_ag_esa_2020_clipped_' + c + '_v02_carbon_compress.tif'
#     veg_stats = pygeo.zonal_statistics(base_raster_path_band = (veg_path, 1),
#                                                         aggregate_vector_path = state_boundaries,
#                                                         ignore_nodata = True)

#     diff_path = 'Results/carbon_new/' + c + '/lulc_diff_2020_' + c + '.tif'
#     diff_stats = pygeo.zonal_statistics(base_raster_path_band = (diff_path, 1),
#                                                         aggregate_vector_path = state_boundaries,
#                                                         ignore_nodata = True)

#     for i, state in enumerate(states):
#         tot_carbon[state] = carbon_stats[i]['sum']
#         veg_carbon[state] = veg_stats[i]['sum']
#         diff_carbon[state] = diff_stats[i]['sum']

#     tot_carbon_df = pd.DataFrame([tot_carbon])
#     tot_carbon_df = tot_carbon_df.transpose().reset_index().rename(columns={'index':'State'})
#     tot_carbon_df.columns = ['State', 'Real carbon stored']

#     veg_carbon_df = pd.DataFrame([veg_carbon])
#     veg_carbon_df = veg_carbon_df.transpose().reset_index().rename(columns={'index':'State'})
#     veg_carbon_df.columns = ['State', 'Carbon stored in natural vegetation scenario']

#     diff_carbon_df = pd.DataFrame([diff_carbon])
#     diff_carbon_df = diff_carbon_df.transpose().reset_index().rename(columns={'index':'State'})
#     diff_carbon_df.columns = ['State', 'Difference between natural veg. and real']

#     df = pd.merge(tot_carbon_df, veg_carbon_df,
#                   how = 'inner', on = 'State')

#     df = pd.merge(df, diff_carbon_df,
#                   how = 'inner', on = 'State')

#     df.sort_values(by=['State'], inplace=True)

#     df.to_csv('Results/carbon_new/' + c + '_results_by_state.csv')


#### SCC ###
# r = 0.025 # discount rate = 2.5%
# scc = 120
# rental_rate = scc*(1 - np.exp(-r)) # from Parisa et al., 2022

# # for c in country_list:
# #     results_df = pd.read_csv('Results/carbon_new/' + c + '_results_by_state.csv')
# #     carbon_es_lost = results_df['Difference between natural veg. and real']

# #     results_df['Total value of lost carbon storage'] = carbon_es_lost*scc
# #     results_df['Rental value of lost carbon storage'] = carbon_es_lost*rental_rate
# #     results_df.to_csv('Results/carbon_new/' + c + '_results_by_state_valuation.csv')

# countries_results = pd.read_csv('Results/carbon_new/country_results.csv')
# carbon_es_lost = countries_results['Difference between natural veg. and real']
# countries_results['Total value of lost carbon storage'] = carbon_es_lost*scc
# countries_results['Rental value of lost carbon storage'] = carbon_es_lost*rental_rate
# countries_results.to_csv('Results/carbon_new/country_results_valuation.csv')
