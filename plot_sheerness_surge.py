"""
plot_sheerness_surge.py

A python script to plot surge residuals near Sheerness.
jelt 14May2026

To run:
    conda activate ntslf_py39
    python plot_sheerness_surge.py

Know issues:
    None

"""
import xarray as xr
import matplotlib.pyplot as plt

def main():
    filename_ssh = "/Users/jelt/Downloads/20260422T0000Z-surge_ukcff_det-surge_points.nc"
    
    # Load the NetCDF dataset
    ds = xr.open_dataset(filename_ssh)
    
    # Convert station_name to string for easier matching
    station_names = ds['station_name'].values
    station_strs = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in station_names]
    
    target_station = "EA-Sheerness"
    if target_station not in station_strs:
        print(f"Station '{target_station}' not found in the dataset.")
        # Find similar stations
        similar = [s for s in station_strs if "Sheerness" in s]
        if similar:
            print(f"Did you mean one of these? {similar}")
        return

    target_idx = station_strs.index(target_station)
    
    # Get target coordinates
    target_lat = ds['latitude'].values[target_idx]
    target_lon = ds['longitude'].values[target_idx]
    
    # Find all stations within 1 degree
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    
    nearby_indices = []
    for i in range(len(station_strs)):
        if abs(lats[i] - target_lat) <= 0.1 and abs(lons[i] - target_lon) <= 0.1:
            nearby_indices.append(i)
            
    print(f"Found {len(nearby_indices)} stations within 0.1 degree of {target_station}.")
    
    # Plotting
    plt.figure(figsize=(12, 8))
    forecast_period = ds['forecast_period']
    time = ds['time']

    for idx in nearby_indices:
        station_name = station_strs[idx]
        station_data = ds.isel(station=idx)
        zos_residual = station_data['zos_residual']
        

        # Output as text (time, zos_residual)
        print(f"\nStation: {station_name}")
        print("time, zos_residual")
        for t, z in zip(time.values, zos_residual.values):
            print(f"{t}, {z}")

        plt.plot(time, zos_residual, marker='.', linestyle='-', label=station_name)
    
    plt.title(f"Surge Residual vs Forecast Period\nStations within 0.1 degree of {target_station}")
    plt.xlabel("Forecast Period")
    plt.ylabel("Surge Residual (zos_residual)")
    plt.grid(True)
    # Move legend outside the plot so it doesn't overlap the data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Export data to CSV
    hourly_csv(ds, nearby_indices, station_strs)

def hourly_csv(ds, nearby_indices, station_strs, filename="hourly_surge.csv"):
    import pandas as pd
    
    time_vals = ds['time'].values
    data = {'time': time_vals}
    
    for idx in nearby_indices:
        station_name = station_strs[idx]
        zos_residual = ds.isel(station=idx)['zos_residual'].values
        data[station_name] = zos_residual
        
    df = pd.DataFrame(data)
    df.to_csv(filename, sep="\t", index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    main()
