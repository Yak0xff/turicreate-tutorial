import  turicreate as tc

# load sessions from preprocessed data
data = tc.SFrame('hapt_data.sframe')

# train/test split by recording sessions
train, test = tc.activity_classifier.util.random_split_by_session(data, session_id='exp_id', fraction=0.8)

# define features
features = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# create an activity classifier
model = tc.activity_classifier.create(train, session_id='exp_id', target='activity', features=features, prediction_window=50, max_iterations=20)

# evaluate the model and save result into dictionary
metrics = model.evaluate(test)
print (metrics['accuracy'])



# Save the model for later use in Turi Create
model.save('mymodel.model')

# Export for use in Core ML
model.export_coreml('MyActivityClassifier.mlmodel')