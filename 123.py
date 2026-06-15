from PIL import Image

img = Image.open("Script2.png")
img = img.convert("RGB")
img.save("clean2.png")