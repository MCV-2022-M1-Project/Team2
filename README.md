# Content Based Image Retrieval
This is a project of the M1 of the Computer Vision Master where we aim is to search images in a large image database based on visual contents.


## Requirements
Before running the project, make sure you have installed the following packages:
- cv2
- numpy
- imutils
- argparse
- collections
- pickle

## Run
Start the project running the following:
```
python ./search.py -d (folder of the database) -i (path to store indexs) -q (folder of the first query) -b (folder of the second query) 
        -t (Path to the test query)  -m (path to store the masks)
```
### Parameters
-d: Path to the folder of the database

-i: Path to where we stored our index

-q: Path to the folder of the query 1

-b: Path to the folder of the query 2

-t: Path to the folder of the test query 1

-m: Path to the folder where you want to save the masks