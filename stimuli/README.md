# photodraw stimuli

Directory to contain stimulus preprocessing code for this project. Running `download_data.py` creates the stimuli folders and metadata files that these files would have created.

### `/kiddraw_generate_stims`
Contains jupyter notebooks and python scripts needed for generating `kiddraw` experiment stimuli & metadata. The notebook `upload_stims_to_s3_and_mongo.ipynb` requires the folder `./photodraw2`. The data for this folder can be found [here](https://osf.io/49ypj/). Use this notebook to generate `photodraw2_meta.js`.

To upload images to s3, run `upload_images_to_s3.py`.

### `/photodraw2x2_generate_stims`
Contains jupyter notebooks and two small csv files needed for generating both `photodraw2x2` and `photodraw_norming` experiment stimuli & metadata.
- `generate_photocue_metadata.ipynb` puts 96 indices into mongoDB for each of instancedraw_photo and categorydraw_photo. These indices get popped of mongoDB each time a participant loads the experiment. The indices ensure that all images get even coverage.
- `select_sketchy_classes_for_photodraw32.ipynb` reads in manually annoted categories from the `Sketchy` dataset (Sangkloy et. al, 2016) and selects 32 to be used in `photodraw2x2`. Then, it constructs metadata for the `photodraw_norming` and `photodraw2x2` experiments.
- `upload_stims_to_s3.ipynb` uploads the 1024 photo-cue stimuli used in `photodraw2x2` into the `photodraw32` bucket of s3.

### `/photodraw2x2_human_recog_stims`
Contains jupyter notebooks needed for generating metadata and stimuli for `recogdraw_category` and `recogdraw_instance` experiments.
- `generate_recog_mongo_data.ipynb` puts the into mongoDB the indices used in `recogdraw_category` and `recogdraw_instance`. These indices get popped of mongoDB each time a participant loads the experiment. The indices ensure that all sketches get even coverage. 
- `instancedraw_generate_stims.ipynb` generates metadata used in `recogdraw_instance`. In particular, for each sketch it finds the 8 images which are most similar to that sketch using k-nearest neighbors in VGG-19 fc6 feature space. These are the images used in the 8-AFC exemplar recognition task.
- `photodraw_recog_task_setup.ipynb` accomplishes two tasks: it determines experiment structure (e.g., trial length, pricing, ratings per sketch) and it turns our experiment metadata into a .js format. 
- `upload_sketches_to_s3.ipynb` uploads the 12,288 sketches from the `photodraw2x2` experiment and uploads them to amazon s3. 

## To appear after running `download_data.py`:

### `/photodraw32_stims`
Contains all photo-cue stimuli for the 32 categories used in the `photodraw2x2` experiment. All 32 stimuli per category can be viewed in a single file in `photodraw_cogsci2021/gallery/photodraw2x2_stims_gallery`.

### `/photodraw_stims`
Contains all photo-cue stimuli for the `kiddraw` experiment.
