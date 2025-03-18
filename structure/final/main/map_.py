import json
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, Polygon, MultiPolygon

def map_(geojson_file, df):
    """
    This function takes a GeoJSON file and a DataFrame, matches the counties from the DataFrame
    with the counties in the GeoJSON, and plots the geometries on a map with a basemap.

    geojson_file: Path to the GeoJSON file containing county geometries.
    df (pandas.DataFrame): DataFrame containing the 'County of Injury' column.
    """
    # Load json File with Coordinates and County Names
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)

    # Extract said information from the file
    county_mapping = {}
    for feature in geojson_data["features"]:
        county_name = feature["properties"]["NAME"].upper()  # Convert to uppercase for matching
        county_mapping[county_name] = feature["geometry"]  # Store geometry for later extraction

    # Now match the Counties in the file to Counties in our Data
    injury_counties = df['County of Injury'].dropna().unique()

    non_matched = {}
    matched_counties = {}
    matched_coordinates = {}

    for county in injury_counties:
        # Upper case
        county_upper = county.upper() 
        if county_upper in county_mapping:
            matched_counties[county_upper] = county_mapping[county_upper]
            # Extract the coordinates for the matched county
            geometry = county_mapping[county_upper]
            if geometry["type"] == "Point":
                # For point geometries, extract the single coordinate
                matched_coordinates[county_upper] = geometry["coordinates"]
            elif geometry["type"] == "Polygon" or geometry["type"] == "MultiPolygon":
                # For polygons and multipolygons, extract the coordinates
                matched_coordinates[county_upper] = geometry["coordinates"]
        else:
            non_matched[county_upper] = "No match found"

    # Check if there are any matched coordinates to plot
    #if not matched_coordinates:
    #    print("No matched coordinates to plot!")
    #    return

    # Create an empty list to store the geometries
    geometries = []

    # Loop through matched coordinates
    for county, coords in matched_coordinates.items():
        # Try-Except for easier Error-handling
        try:
            # Handle Point 
            if isinstance(coords, list) and len(coords) == 1 and isinstance(coords[0], float):  
                geometries.append(Point(coords[0]))

            # Handle Polygon or MultiPolygon geometries
            elif isinstance(coords, list): 
                if isinstance(coords[0], list): 
                    # Polygon
                    if len(coords) == 1: 
                        geometries.append(Polygon(coords[0]))  
                        
                    # MultiPolygon
                    else:  # MultiPolygon
                        multipolygon = MultiPolygon([Polygon(p[0]) for p in coords])
                        geometries.append(multipolygon)
                # Neither
                else: 
                    raise ValueError(f"Invalid format for {county} with coordinates: {coords}")

            else:
                raise ValueError(f"Unexpected coordinate format for {county}: {coords}")

        except Exception as e:
            print(f"Error processing {county}: {e}")

    # Check for empty geometries
    if not geometries:
        print("No geometries were successfully processed!")
        return

    # Create a GeoDataFrame from the geometries
    gdf = gpd.GeoDataFrame(geometry=geometries)

    # Set CRS (Coordinate Reference System) for the GeoDataFrame to match the basemap
    gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)

    # Plotting with background map
    ax = gdf.plot(figsize=(10, 10), alpha=0.7, edgecolor='k', color='orange')

    # Add basemap using contextily (Web Map Tile Service)
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

    plt.show()