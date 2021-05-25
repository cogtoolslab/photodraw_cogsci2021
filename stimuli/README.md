Directory to contain stimulus preprocessing code for this project.

### `/kiddraw_generate_stims`
Contains jupyter notebooks and python scripts needed for generating kiddraw experiment stimuli & metadata. The notebook `upload_stims_to_s3_and_mongo.ipynb` requires the folder `./photodraw2`. The data for this folder can be found [here](https://osf.io/49ypj/). Use this notebook to generate `photodraw2_meta.js`.

To upload images to s3, run `upload_images_to_s3.py`.
