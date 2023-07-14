from PIL import Image
#Given a sprite sheet of m x n images, assuming that all sprites have the same
#dimensions this script extracts each image and saves it in a separate file. 
# Load the image
def imageloaderFromMatix(image_path,idx):
    input_path = ".\\upzised\\"
    output_path = ".\\upzised\\out\\"
    img = Image.open(input_path+image_path)

# Calculate the size of the individual images
    width, height = img.size
    single_width = width // 3
    single_height = height // 4

# Loop through the matrix
    for i in range(4):
        for j in range(3):
        # Calculate the box to extract
            left = j * single_width
            top = i * single_height
            right = (j+1) * single_width
            bottom = (i+1) * single_height
            box = (left, top, right, bottom)
        
        # Extract and save the image
            extracted_img = img.crop(box)
            extracted_img.save(f"{output_path}image_{i}_{j}_{idx}.png")

for i in range(15):
    imageloaderFromMatix(f"flowersMatrix-upsized{i+1}.png",i)
