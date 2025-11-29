import rasterio
from affine import Affine

# Mock transform from the logs
# Transform: | 1.00, 0.00, 299363.38|
# | 0.00,-1.00, 5052882.86|
# | 0.00, 0.00, 1.00|
transform = Affine(1.0, 0.0, 299363.38, 0.0, -1.0, 5052882.86)

# Target Point
x = 301318.39
y = 5045690.28

# Apply Inverse Transform
res = ~transform * (x, y)
print(f"Result: {res}")

# Expected: col ~ 1955, row ~ 7192
