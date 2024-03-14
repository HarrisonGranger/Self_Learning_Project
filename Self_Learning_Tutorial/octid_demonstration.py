# Harrison Granger
# Dr. Ying Ding
# University of Texas at Austin
# AI in Healthcare 

# Import OCTID class, pandas, and matplotlib.
import octid_class
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Define training, validation, and unlabelled datasets.
training_dateset = "your training dataset path"
validation_dataset = "your validation dataset path"
unlabelled_dataset = "your unlabelled dataset path" 

# Define models and create empty dictionary for analysis.
models = ['alexnet', 'vgg11', 'reset18', 'densenet121' 'inception_v3', 'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v2', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'mnasnet1_0']
results = {'Model Name':[], 'Accuracy':[]}
# Classify results with each model.
for i, model_name in enumerate(models):
    try:
        # For each model, call OCTID classification.
        print('Attempting use of %s model for analysis.' % model_name)
        classify_model = octid_class.octid(model = model_name, customised_model = False, feature_dimension = 3, outlier_fraction_of_SVM = 0.03, training_dataset = training_dateset, validation_dataset = validation_dataset, unlabeled_dataset=unlabelled_dataset)
        accuracy = classify_model()
        # Append results to results dictionary.
        results['Model Name'].append(model_name)
        results['Accuracy'].append(accuracy)
    except:
        print('Failed on model: %s' %model_name)
# Create pandas dataframe of results.
accuracy_results = pd.DataFrame(results)
# Create graph of results with each model name as x axis.
ax = accuracy_results.plot(kind='bar', x='Model Name', y='Accuracy', ylim=(.85, 1), title='Accuracy of Trained Models with Introduced Data')
# Format y-axis as percent.
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set_xlabel('Model Name')
ax.set_ylabel('Percent of Samples')
results_figure = ax.get_figure()
# Rotate x-axis labels so they are legible.
results_figure.autofmt_xdate(rotation=45)
# Save figure image.
results_figure.savefig('Accuracy Analysis Between Trained Models and Introduced Data.png')
