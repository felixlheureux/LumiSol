import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

# The file you downloaded
FILEPATH = "sample.tif"

def pick_coordinates():
    if not os.path.exists(FILEPATH):
        print(f"❌ Error: Could not find '{FILEPATH}' in the backend folder.")
        print("   Please download a MNS GeoTIFF from Données Québec and rename it to sample.tif")
        return

    # 1. Load Data
    with rasterio.open(FILEPATH) as src:
        print(f"✅ Loaded {FILEPATH}")
        print(f"   Size: {src.width} pixels wide x {src.height} pixels tall")
        
        # Read the elevation data
        data = src.read(1)
        
        # Handle "No Data" values (The cause of the white blob)
        # If the file has a defined nodata value, use it to mask
        nodata = src.nodata
        if nodata is not None:
            mns = np.ma.masked_equal(data, nodata)
        else:
            mns = np.ma.masked_invalid(data)

        # Extra safety: Mask out absurdly low values (e.g. -9999) often found in GIS data
        # Assuming Montreal is not below -50m
        mns = np.ma.masked_less(mns, -50)

    # 2. Setup Interactive Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate robust contrast limits (2nd to 98th percentile)
    # This ignores outliers and focuses color on the actual data range
    if mns.count() > 0:
        vmin = np.percentile(mns.compressed(), 2)
        vmax = np.percentile(mns.compressed(), 98)
    else:
        vmin, vmax = None, None

    # Use the robust limits for plotting
    img = ax.imshow(mns, cmap='terrain', vmin=vmin, vmax=vmax)
    ax.set_title("CLICK ON A ROOF to get coordinates!\n(Close window to exit)")
    fig.colorbar(img, ax=ax, label='Elevation (m)')

    # 3. The Click Handler Function
    def on_click(event):
        # Check if the click was inside the plot axis
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            
            # Check bounds
            if 0 <= x < mns.shape[1] and 0 <= y < mns.shape[0]:
                val = mns[y, x]
                if np.ma.is_masked(val):
                    val_str = "No Data / Masked"
                else:
                    val_str = f"{val:.2f} meters"

                print(f"\n🎯 FOUND TARGET:")
                print(f"   click_x = {x}")
                print(f"   click_y = {y}")
                print(f"   Height at this point: {val_str}")
                print("-" * 30)
                
                # Draw a red 'X' where you clicked
                ax.plot(x, y, 'rx', markersize=12, markeredgewidth=2)
                fig.canvas.draw()

    # Connect the click event to the plot
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("Window opened. Please click on the roof you want to test...")
    plt.show()

if __name__ == "__main__":
    pick_coordinates()