# photodraw results

Contains main results for this project. Running `download_data.py` loads tidy dataframes into the `results/csv` directory, while the various analyses notebooks contribute to the `results/plots` directory.

- `/results/csv`: dataframes
  - `photodraw_sketch_data.csv` contains sketch-level information for the `kiddraw` experiment
  - `photodraw_stroke_data.csv` contains stroke-level information for the `kiddraw` experiment
  - `photodraw_survey_data.csv` contains survey information for the `kiddraw` experiment
  - `photodraw2x2_sketch_data.csv` contains sketch-level information for the `photodraw2x2` experiment
  - `photodraw2x2_stroke_data.csv` contains stroke-level information for the `photodraw2x2` experiment
  - `photodraw2x2_survey_data.csv` contains survey information for the `photodraw2x2` experiment
  - `photodraw2x2_category_by_experiment_variances.csv` contains variance within category-2x2 condition for sketches in the `photodraw2x2` experiment (e.g., variation of cat drawings in the instancedraw-photo conditon)
  - `photodraw2x2_category_recog_ratings.csv` contains information for each rating in the category-level recognition validation task.
  - `photodraw2x2_category_recog_survey.csv` contains survey data in the category-level recognition validation task.
  - `photodraw2x2_instance_recog_ratings.csv` contains information for each rating in the instance-level recognition validation task.
  - `photodraw2x2_instance_recog_survey.csv`contains survey data in the instance-level recognition validation task.
  - `photodraw_metadata_instance.csv` is a helper csv for instance-level classification of the `kiddraw` experiment.
- `/results/plots`: plots
  - various plots will be loaded in through the notebooks in `/analysis`
