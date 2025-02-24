from PIL import Image

# Open the image file (ensure the path is correct)
image_path = "figure-3.png"
img = Image.open(image_path)
img = img.convert("RGB")

# Save the image as an EPS file
img.save("figure-3.eps", format="EPS")
