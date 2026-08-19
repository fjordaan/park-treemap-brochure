import numpy as np
from PIL import Image
from rembg import remove, new_session
import io

input_path  = 'park-treemap-bg1.jpg'
output_path = 'park-treemap-bg1-nobg3.png'

print("Loading model...")
session = new_session("u2net_human_seg")
print("Running inference...")
with open(input_path, 'rb') as f:
    result = remove(f.read(), session=session, decontaminate=True)

img = Image.open(io.BytesIO(result)).convert("RGBA")
r, g, b, a = img.split()
arr = np.array(a)
arr = np.where(arr > 76, 255, np.where(arr < 25, 0, arr))
img.putalpha(Image.fromarray(arr.astype('uint8')))
img.save(output_path)
print("Done:", output_path)
