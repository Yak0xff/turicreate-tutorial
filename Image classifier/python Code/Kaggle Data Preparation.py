# import Turi Create
import turicreate as tc

# define data directory
data_dir = '../kagglecatsanddogs/PetImages/'

# Load images (Note: you can ignore 'Not a JPEG file' errors)
data = tc.image_analysis.load_images(data_dir, with_path=True)

# From the path-name, create a label name
data['label'] = data['path'].apply(lambda path: 'dog' if '/Dog' in path else 'cat')

# Save the data for future use
data.save('cat-dogs.sframe')

# Explore interactively
data.explore()
