# Harrison Granger
# Self Learning Tutorial
# March 8, 2024

import octid_class

# Your dataset paths:
td = "Path to your training data"
vd = "Path to your validation data"
ud = "Path to your unlabelled data"

# Call the octid class with default options.
classify_model = octid_class.octid(model = 'googlenet', 
                             customised_model = False, 
                             feature_dimension = 3, 
                             outlier_fraction_of_SVM = 0.03, 
                             training_dataset=td, 
                             validation_dataset=vd, 
                             unlabeled_dataset=ud
                             )
# Cass the classify_model function to begin classification.
classify_model()
