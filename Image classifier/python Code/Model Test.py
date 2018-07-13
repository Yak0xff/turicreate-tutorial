import turicreate as tc

# load saved model
imageClassifier = tc.load_model('imageClassifier.model')

# load sessions from preprocessed data
data = tc.SFrame('cat-dogs.sframe')

# filter the cat image
new_cats_dogs = data[(data['label'] == 'cat')][100:120]

print(new_cats_dogs)

# do predict
predicts = imageClassifier.predict(new_cats_dogs)
print(predicts)
