from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np





# Define path to the directory that contains the images
image_path_dir = '.\\upzised\\out\\'
image_path_dir_pre = '.\\label\\'

# Load MobileNet
image.LOAD_TRUNCATED_IMAGES = True 
model = MobileNet(weights='imagenet', include_top=False)

# Define function to preprocess images
def model_predict(image_path):
    img = image.load_img(image_path, target_size=(309, 232))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    mobilenet_feature = model.predict(img_data)
    
    return mobilenet_feature

# List all the files in the directory
filelist = glob.glob(os.path.join(image_path_dir, '*.png'))

# Extract features from the images
mobilenet_feature_list = [model_predict(fname) for fname in filelist]

# Prepare the features for clustering
mobilenet_feature_list_np = np.array(mobilenet_feature_list)
img_features = mobilenet_feature_list_np.reshape(mobilenet_feature_list_np.shape[0],-1)

# Apply KMeans
max_clusters= 8
kmeans = KMeans(n_clusters=max_clusters, random_state=0).fit(img_features)

# Print the labels
print(kmeans.labels_)

# Rename files based on the labels
for i, label in enumerate(kmeans.labels_):
    # Define the new name for the file with cluster label
    new_name = f"cluster_{label}_{os.path.basename(filelist[i])}"
    # Define the output path
    output_path = os.path.join(image_path_dir_pre, new_name)

    # Copy and rename the file
    shutil.copy(filelist[i], output_path)


# Calculate the centroids of the clusters
centroids = kmeans.cluster_centers_

# Calculate the similarity matrix between clusters
similarity_matrix = np.zeros((len(centroids), len(centroids)))

for i in range(len(centroids)):
    for j in range(i+1, len(centroids)):
        distance = euclidean(centroids[i], centroids[j])
        similarity = 1 / (1 + distance)  # Convert distance to similarity
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

print(similarity_matrix)

#---------------------------
# Generate random positions for each cluster
positions = np.random.rand(len(similarity_matrix), 2)

# Create a scatter plot
plt.scatter(positions[:, 0], positions[:, 1], c='blue')

# Add labels for each cluster
for i, pos in enumerate(positions):
    plt.text(pos[0], pos[1], str(i), ha='center', va='center', fontsize=10)

# Add connections between similar clusters
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i, j] > 0:
            plt.plot([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]], 'gray', alpha=similarity_matrix[i, j])

# Set axis labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Similarity Matrix Scatter Plot')

# Show the plot
plt.show()

#----------------------------------

# Grid configuration
max_images_per_cluster = 30  # Adjust as per your requirements
zoom = 0.7 # Adjust as per your requirements. Higher values = larger images

# Group the images by their cluster labels
clustered_images = {label: np.array(filelist)[kmeans.labels_==label] for label in set(kmeans.labels_)}

# save each cluster separataley
for col, (label, images) in enumerate(clustered_images.items()):
    # Calculate figure size based on number of images in the cluster
    fig_height = min(len(images), max_images_per_cluster) * 5 * zoom

    # Create the figure for the current cluster
    fig, axs = plt.subplots(min(len(images), max_images_per_cluster), 1, figsize=(5 * zoom, fig_height))

    for row in range(min(len(images), max_images_per_cluster)):
        ax = axs[row]
        img = Image.open(images[row])
        ax.imshow(img)
        ax.axis('off')
        # Add a title for the cluster to the first subplot
        if row == 0:
            ax.set_title(f"Cluster {label}", fontsize=10)

    # Adjust layout for the title
    plt.tight_layout(pad=1.0)

    # Save the figure
    fig.savefig(f"cluster_{label}.png")
    
    plt.close(fig)  # Close the figure to free up memory

#------------------------------------

# Calculate figure size based on number of clusters
fig_width = len(clustered_images) * 1 * zoom  # 5 inches per cluster, adjust as per your needs
fig_height = max_images_per_cluster * 5 * zoom  # 5 inches per image, adjust as per your needs

# Create the figure and the subplots
fig, axs = plt.subplots(max_images_per_cluster, len(clustered_images), figsize=(fig_width, fig_height))

for col, (label, images) in enumerate(clustered_images.items()):
    for row in range(max_images_per_cluster):
        ax = axs[row, col]
        if row < len(images):  # If there is an image for this row
            img = Image.open(images[row])
            ax.imshow(img)
            ax.axis('off')
        else:  # If there are no more images in the cluster, remove the axis
            fig.delaxes(ax)
    
    # Add a title for the cluster
    axs[0, col].set_title(f"Cl_{label}", fontsize=10)

# Adjust layout for the titles
plt.tight_layout(pad=3.0)  # Use a padding of 3.0 inches on each side
plt.show()
    # Save the figure
fig.savefig(f"fullmosaic.png")
    
plt.close(fig)  # Close the figure to free up memory