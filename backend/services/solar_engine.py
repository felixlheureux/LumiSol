import numpy as np
import pandas as pd
import pvlib
from rvt import vis
from services.weather_service import WeatherService

class SolarEngine:
    def __init__(self, config):
        self.config = config
        self.weather = WeatherService()
        self.system_efficiency = 0.20 * 0.85  # 20% Panel * 85% System Perf Ratio

    def calculate_snow_loss_factor(self, snow_depth_m, temp_c, slope_deg):
        """
        Returns a factor (0.0 to 1.0) for every hour of the year.
        1.0 = Clean Panels, 0.0 = Covered in Snow.
        """
        # Threshold: 2cm of snow blocks light
        is_covered = snow_depth_m > 0.02
        
        # Recovery: Snow melts if Temp > 0°C OR slides if Roof is steep (>60°)
        is_melting = temp_c > 0
        is_steep = slope_deg > 60
        
        # Loss occurs if Covered AND (Not Melting AND Not Steep)
        is_blocked = is_covered & ~(is_melting | is_steep)
        
        # Return 0.0 if blocked, 1.0 if clear
        return np.where(is_blocked, 0.0, 1.0)

    def simulate_year(self, lat, lon, slope, aspect, area_m2):
        """
        Simulates 365 days of energy production.
        """
        # 1. Get Hourly Weather (8760 rows)
        weather = self.weather.get_last_year_weather(lat, lon)
        
        # 2. Calculate Solar Position for every hour
        solpos = pvlib.solarposition.get_solarposition(weather.index, lat, lon)
        
        # 3. Transposition (Calculate Plane-of-Array Irradiance)
        # This converts horizontal sun (GHI) to titled sun on the roof (POA)
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=slope,
            surface_azimuth=aspect,
            dni=weather['dni'],
            ghi=weather['ghi'],
            dhi=weather['dhi'],
            solar_zenith=solpos['apparent_zenith'],
            solar_azimuth=solpos['azimuth']
        )
        
        # 4. Apply Physics Losses
        # A. Snow Loss
        snow_factor = self.calculate_snow_loss_factor(
            weather['snow_depth_m'].values, 
            weather['temp_c'].values, 
            slope
        )
        
        # 5. Calculate Energy (kWh)
        # Formula: Irradiance (W/m2) * Area (m2) * Eff * SnowFactor / 1000
        hourly_kwh = (
            poa['poa_global'] * area_m2 * self.system_efficiency * snow_factor
        ) / 1000.0
        
        # 6. Aggregate to Daily Totals (Sum)
        # This creates the 365 data points
        daily_kwh = hourly_kwh.resample('D').sum()
        
        # 7. Format for Frontend (JSON Array)
        # Format: [{"date": "2024-01-01", "kwh": 12.5}, ...]
        graph_data = [
            {"date": date.strftime('%Y-%m-%d'), "kwh": round(val, 1)}
            for date, val in daily_kwh.items()
            if not pd.isna(val) # Safety check
        ]
        
        total_annual_kwh = int(hourly_kwh.sum())
        
        return graph_data, total_annual_kwh

    def analyze_patch_potential(self, dsm, lat, lon, pixel_res=0.2):
        """
        Generates an Annual kWh map using Last Year's Real Weather.
        """
        # 1. Geometry (Slope/Aspect)
        dy, dx = np.gradient(dsm, pixel_res)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        aspect_rad = np.arctan2(-dx, dy)
        aspect_deg = (np.degrees(aspect_rad) + 360) % 360

        # 2. Get Historical Weather (DataFrame: 8760 rows)
        # This gives us the actual GHI/DNI for every hour of the past year
        weather_df = self.weather.get_last_year_weather(lat, lon)
        
        # 3. Horizon Shading (Static SVF for Diffuse)
        svf_dict = vis.sky_view_factor(
            dem=dsm.astype(np.float32), 
            resolution=pixel_res, 
            compute_svf=True, 
            nr_directions=16, 
            max_radius=100
        )
        svf_map = svf_dict['svf'] # (H, W)

        # 4. Vectorized Irradiance Accumulation (The "Engine")
        # We process the year in chunks or simplified sums to keep it fast.
        
        # A. Calculate "Effective POA" Factor per pixel
        # Ideal Factor = cos(Incident Angle)
        # We approximate this by summing the components weighted by geometry
        
        total_potential_map = np.zeros_like(dsm, dtype=np.float32)
        
        print("   ☀️ Calculating annual potential (Vectorized)...")
        
        # Optimization: Instead of looping 8760 times, we group by "Sun Position Bins"
        # or we just sum the components.
        # For maximum accuracy vs speed, we calculate the SUM of GHI
        # and apply a "Mean Efficiency Factor" based on the geometry.
        
        total_ghi = weather_df['ghi'].sum() / 1000.0 # Total annual GHI (kWh/m2)
        
        # B. Geometry Efficiency Map
        # South (180) @ Lat-Tilt (~45) is 100%. 
        # North (0) is ~60%. Walls are 0%.
        
        # Simple transposition proxy:
        # Eff = 1.0 - (Deviation from Optimal)
        optimal_slope = lat
        optimal_azimuth = 180
        
        slope_penalty = np.abs(slope_deg - optimal_slope) / 180.0
        aspect_penalty = np.abs(aspect_deg - optimal_azimuth) / 180.0
        
        # Flat roofs (slope < 10) don't care about aspect
        aspect_penalty[slope_deg < 10] = 0 
        
        geo_efficiency = 1.0 - (0.5 * slope_penalty) - (0.3 * aspect_penalty)
        geo_efficiency = np.clip(geo_efficiency, 0.1, 1.1) # Boost optimal slightly
        
        # C. Snow Loss Map (The "Research" Part)
        # We calculate the % of the year the roof is covered based on the weather file
        # Filter weather for snow events
        snowy_hours = weather_df[
            (weather_df['snow_depth_m'] > 0.02) & (weather_df['temp_c'] < 0)
        ]
        fraction_snow_covered = len(snowy_hours) / 8760.0
        
        # Apply logic: Steeper roofs shed snow faster
        # If slope > 45, snow loss is 0. If slope < 10, snow loss is full duration.
        shedding_factor = np.clip((slope_deg - 10) / 35.0, 0.0, 1.0) 
        actual_snow_loss = fraction_snow_covered * (1.0 - shedding_factor)
        
        snow_efficiency = 1.0 - actual_snow_loss

        # D. Combine
        # Total = Base_Resource * Geometry * Shading * Snow * System_Perf
        total_potential_map = (
            total_ghi * geo_efficiency * svf_map * snow_efficiency * self.system_efficiency
        )
        
        # Mask out walls (Vertical surfaces > 60deg are rarely used for solar)
        total_potential_map[slope_deg > 60] = 0
        
        return total_potential_map
