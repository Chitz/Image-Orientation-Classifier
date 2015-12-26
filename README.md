# Image Orientation Classifier

A classifier that decides the correct orientation of a given image

Dataset- Images from the Flickr photo sharing website, 10,000 Training set and 1,000 Test set of Images.

Each raw image is converted into numerical feature vectors., that is, n × m × 3 color image (the third dimension is because color images are stored as three separate planes ? the red, green, and blue), and append all of the rows together to produce a single vector of size 1 × 3mn.

Each training image is rotated 4 times to give four times as much training data. The data is of the following format -
photo_id correct_orientation r11 g11 b11 r12 g12 b12 ...

where:? photo id is a photo ID for the image.? correct orientation is 0, 90, 180, or 270. Note that some small percentage of these labels may be wrong because of noise in the data; this is just a fact of life when dealing with data from real-world sources.? r11 refers to the red pixel value at row 1 column 1, r12 refers to red pixel at row 1 column 2, etc.

Implemented the following algorithms -

1. K-Nearest Neighbor -

Command line: 

python orient.py train_file.txt test_file.txt knn k

2. Neural Network

Command line:

python orient.py train_file.txt test_file.txt nnet hidden_count

3. Best mode:

python orient.py train_file.txt test_file.txt best model_file

