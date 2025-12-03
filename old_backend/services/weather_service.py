import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

class WeatherService:
    def __init__(self):
        # Setup caching (stores data in a local SQLite file for 1 day)
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600*24)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)
        self.url = "https://archive-api.open-meteo.com/v1/archive"

    def get_last_year_weather(self, lat, lon):
        """
        Fetches hourly solar & snow data for the full previous year.
        """
        # Calculate dates for "Last Year"
        end_date = pd.Timestamp.now().floor('d') - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=365)
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m", 
                "snow_depth", 
                "shortwave_radiation", # GHI
                "direct_normal_irradiance", # DNI
                "diffuse_radiation" # DHI
            ],
            "timezone": "auto"
        }

        # Fetch data
        responses = self.client.weather_api(self.url, params=params)
        response = responses[0]

        # Parse into Pandas DataFrame
        hourly = response.Hourly()
        
        # Helper to extract numpy arrays
        def get_series(index):
            return hourly.Variables(index).ValuesAsNumpy()

        data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temp_c": get_series(0),
            "snow_depth_m": get_series(1),
            "ghi": get_series(2),
            "dni": get_series(3),
            "dhi": get_series(4)
        }
        
        df = pd.DataFrame(data=data)
        return df
