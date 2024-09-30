import os

import geopandas as gpd
import numpy as np
from osgeo import gdal
import pandas as pd
import pygeoprocessing.geoprocessing as pygeo
from osgeo import osr


def execute(args):
    """
    Reimplementation of Justin's cython carbon model to use pygeoprocessing.
    The original model read in the full rasters, which was causing memory use
    issues when multiprocessing. This version uses the raster calculator which
    is memory efficient.

    This version also checks whether the carbon value for a non-natural class
    is greater than the carbon value for the potential vegetation, and if so,
    reduces the non-natural carbon

    args should contain:
    + lu_raster
    + pixel_area
    + potential_vegetation
    + target_folder
    + carbon_zone_file
    + carbon_table_file
    """

    output_folder = args["target_folder"]
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    carbon_table_path = args["carbon_table_file"]
    carbon_zones_path = args["carbon_zone_file"]

    # set up lookup tables
    lookup_table_df = pd.read_csv(carbon_table_path, index_col=0)
    table_shape = (len(lookup_table_df.index), len(lookup_table_df.columns))
    lookup_table = np.float32(lookup_table_df.values)
    # row_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.index)}
    # col_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.columns)}
    rows = [int(r) for r in lookup_table_df.index]
    cols = [int(r) for r in lookup_table_df.columns]

    row_idx = np.zeros(max(rows) + 1, dtype="int")
    col_idx = np.zeros(max(cols) + 1, dtype="int")

    for i, r in enumerate(rows):
        row_idx[r] = i
    for i, c in enumerate(cols):
        col_idx[c] = i

    def local_op(lu, pv, cz, pa):
        lu_carb = lookup_table[row_idx[cz], col_idx[lu]] * 100 * pa

        pv_carb = lookup_table[row_idx[cz], col_idx[pv]] * 100 * pa
        # 100 scaling is for ha -> km2 (pixel area in ha, carbon in tons/km2)

        result = np.zeros_like(pa)
        result = np.minimum(lu_carb, pv_carb)
        bmp_mask = np.any([lu == 16, lu == 26], axis=0)
        result[bmp_mask] = 0.9 * lu_carb[bmp_mask] + 0.1 * pv_carb[bmp_mask]
        return result

    raster_band_list = [
        (args["lu_raster"], 1),
        (args["potential_vegetation"], 1),
        (args["carbon_zone_file"], 1),
        (args["pixel_area"], 1),
    ]
    sname = os.path.splitext(os.path.basename(args["lu_raster"]))[0]
    target_file = os.path.join(output_folder, f"{sname}_carbon.tif")

    pygeo.raster_calculator(
        raster_band_list, local_op, target_file, gdal.GDT_Float32, -9999.0
    )


########### RUNNING THE MODEL
os.chdir("C:/Users/kibby/OneDrive/Desktop/Research/Nature Impact Factors/")

country_list = [
    "China",
    "India",
    "United States",
]  # I really only want the contiguous U.S.

lulc_info = pygeo.get_raster_info("data/LULC/lulc_esa_2020.tif")
pixel_size = lulc_info["pixel_size"]  # Example: (x_size, y_size)
# these are pixel sizes in degrees, not meters


# Function to calculate pixel area based on latitude
def calculate_pixel_area(lat, pixel_width, pixel_height, earth_radius):
    lat_rad = np.radians(lat)
    area = (earth_radius**2) * np.abs(
        np.radians(pixel_width) * np.radians(pixel_height) * np.cos(lat_rad)
    )
    return area


# Define parameters
pixel_width = pixel_size[0]
pixel_height = pixel_size[1]
earth_radius = 6371.0  # Earth radius in km

for c in country_list:
    # Load the shapefile using GeoPandas
    shapefile_path = "data/Polygon_vectors/" + c + ".gpkg"
    gdf = gpd.read_file(shapefile_path)

    # Get the bounding box of the shapefile
    bounds = gdf.total_bounds  # (min_lon, min_lat, max_lon, max_lat)
    min_lon, min_lat, max_lon, max_lat = bounds

    # Calculate the number of rows and columns based on the bounding box and pixel size
    cols = int((max_lon - min_lon) / pixel_width)
    rows = int((max_lat - min_lat) / abs(pixel_height))

    # Create an empty array for the raster data
    pixel_area_km2 = np.zeros((rows, cols))

    # Fill the raster with pixel area values, calculate pixel size per latitude
    for i in range(rows):
        lat = max_lat + i * pixel_height
        pixel_area_km2[i, :] = calculate_pixel_area(
            lat, pixel_width, pixel_height, earth_radius
        )

    # Create the raster file
    driver = gdal.GetDriverByName("GTiff")
    out_raster = driver.Create(
        "data/Carbon/pixel_area_clipped_" + c + ".tif", cols, rows, 1, gdal.GDT_Float32
    )

    # Set geotransform (top left x, w-e pixel resolution, 0, top left y, 0, n-s pixel resolution)
    geotransform = (min_lon, pixel_width, 0, max_lat, 0, pixel_height)
    out_raster.SetGeoTransform(geotransform)

    # Set projection to WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    out_raster.SetProjection(srs.ExportToWkt())

    # Write the data to the raster band
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(pixel_area_km2)

    # Flush the dataset to disk
    out_raster.FlushCache()
    out_band = None
    out_raster = None

# Regular run
# for c in country_list:
# base_raster_path_list = ['data/LULC/aligned_lulc_' + c + '.tif',
#                          'data/Carbon/pixel_area_clipped_' + c + '.tif',
#                          'data/scenario_construction/potential_vegetation/potential_vegetation_clipped_' + c + '.tif',
#                          'data/Carbon/carbon_zones_rasterized_clipped_' + c + '.tif']

# target_raster_path_list = ['data/LULC/aligned_lulc_' + c + '_al.tif',
#                          'data/Carbon/pixel_area_clipped_' + c + '_al.tif',
#                          'data/scenario_construction/potential_vegetation/potential_vegetation_clipped_' + c + '_al.tif',
#                          'data/Carbon/carbon_zones_rasterized_clipped_' + c + '_al.tif']

# resample_method_list = ['near', 'bilinear', 'near', 'near'] # Define the resampling method for each raster ('near' for categorical, 'bilinear' for continuous)

# lulc_info = pygeo.get_raster_info(base_raster_path_list[0])
# target_pixel_size = lulc_info['pixel_size']  # Example: (x_size, y_size)

# pygeo.align_and_resize_raster_stack(
#     base_raster_path_list,
#     target_raster_path_list,
#     resample_method_list,
#     target_pixel_size,
#     bounding_box_mode='intersection'
# )

# pygeo.reclassify_raster((target_raster_path_list[0], 1),
#                     {0:0, 10:10, 11:11, 12:12, 13:13, 20:20,
#                     30:30, 40:40, 50:50, 60:60, 61:61, 62:62, 70:70, 71:71, 72:72,
#                         80:80,  81:81, 90:90, 100:100, 110:110, 120:120, 121:121, 122:122,
#                         130:130, 140:140, 150:150, 151:151, 152:152, 153:153, 160:160,
#                         170:170, 180:180, 190:190, 200:200, 201:201, 202:202, 210:210, 220:220,
#                         255:0},
#                     'data/LULC/aligned_lulc_' + c + '_al_v02.tif',
#                     gdal.GDT_Byte,
#                     0, values_required=True)


for c in ["China"]:
    args = {
        "lu_raster": "data/LULC/aligned_lulc_" + c + "_al_v02.tif",
        "pixel_area": "data/Carbon/pixel_area_clipped_" + c + "_al.tif",
        "potential_vegetation": "data/scenario_construction/potential_vegetation/potential_vegetation_clipped_"
        + c
        + "_al.tif",
        "target_folder": "Results/carbon_new/" + c,
        "carbon_zone_file": "data/Carbon/carbon_zones_rasterized_clipped_"
        + c
        + "_al.tif",
        "carbon_table_file": "data/Carbon/exhaustive_carbon_table_just_esa.csv",
    }

    execute(args)

# Run with natural areas instead of cotton cropland
# for c in country_list:
# args_no_ag = {'lu_raster': 'data/LULC/lulc_without_ag_esa_2020_clipped_' + c + '_v02.tif',
#         'pixel_area': 'data/Carbon/pixel_area_clipped_' + c + '_al.tif',
#         'potential_vegetation': 'data/scenario_construction/potential_vegetation/potential_vegetation_clipped_' + c + '_al.tif',
#         'target_folder': 'Results/carbon_new/' + c,
#         'carbon_zone_file': 'data/Carbon/carbon_zones_rasterized_clipped_' + c + '_al.tif',
#         'carbon_table_file': 'data/Carbon/exhaustive_carbon_table_just_esa.csv'}
# execute(args_no_ag)
