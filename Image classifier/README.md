# Image Classifier


Given an image, the goal of an image classifier is to assign it to one of a pre-determined number of labels. Deep learning methods have recently been shown to give incredible results on this challenging problem. Yet this comes at the cost of extreme sensitivity to model hyper-parameters and long training time. This means that one can spend months testing different model configurations, much too long to be worth the effort. However, the image classifier in Turi Create is designed to minimize these pains, and making it possible to easily create a high quality image classifier model.



#### Loading Data

The [Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765) provides labeled cat and dog images.[<sup>1</sup>](../datasets.md) After downloading and decompressing the dataset, navigate to the main **kagglecatsanddogs** folder, which contains a **PetImages** subfolder.

```python
import turicreate as tc

# Load images (Note: you can ignore 'Not a JPEG file' errors)
data = tc.image_analysis.load_images('PetImages', with_path=True)

# From the path-name, create a label column
data['label'] = data['path'].apply(lambda path: 'dog' if '/Dog' in path else 'cat')

# Save the data for future use
data.save('cats-dogs.sframe')

# Explore interactively
data.explore()
```

#### Introductory Example

The task is to **predict if a picture is a cat or a dog**.  Let’s
explore the use of the image classifier on the Cats vs. Dogs dataset.
