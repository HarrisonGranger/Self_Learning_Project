# Harrison Granger
# Self Learning Tutorial 

import octid_class

# Your dataset paths:
td = "C:/Users/htgra/Desktop/Self_Learning_Tutorial/OCTID Source/OCTID-main/small_samples/training_dataset"
vd = "C:/Users/htgra/Desktop/Self_Learning_Tutorial/OCTID Source/OCTID-main/small_samples/validation_dataset"
ud = "C:/Users/htgra/Desktop/Self_Learning_Tutorial/OCTID Source/OCTID-main/small_samples/unlabelled_dataset"

# Lets try an example with the default values.
classify_model = octid_class.octid(model = 'googlenet', 
                             customised_model = False, 
                             feature_dimension = 3, 
                             outlier_fraction_of_SVM = 0.03, 
                             training_dataset=td, 
                             validation_dataset=vd, 
                             unlabeled_dataset=ud
                             )
classify_model()
