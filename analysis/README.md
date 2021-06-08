# photodraw analysis

Directory to contain analysis notebooks/scripts for this project.

_______________________________________________________________

## `kiddraw`

- `kiddraw_setup.ipynb` turns the data stored in MongoDB from the `kiddraw` experiment into three tidy dataframes, stored in the `/results` directory: 
  - `photodraw_sketch_data.csv`
  - `photodraw_stroke_data.csv`
  - `photodraw_survey_data.csv`
- `kiddraw_analysis_1.ipynb` contains exploratory analyses for the `kiddraw` experiment. It primarily looks at the basic distributions of both low-level (number of strokes, sketch time, total ink, and stroke length) and high-level (pixel-level recognizability, fc6-level recognizability) variables, factored by condition (photo- vs. text-cue) and category. It also looks for potential effects of fatigue (trial number) or input device (e.g., tablet) on these variables. Finally, it explores how well low-level and high-level features are able to classify various labels, such as category and condition.  
- `kiddraw_analysis_2.ipynb` contains exploratory analyses for the `kiddraw` experiment. It explores the within-category variation in low-level and high-level features across conditions. 
- `kiddraw_analysis_R.Rmd` conducts inferential statistics, fitting the linear mixed-effects models found in _Study 1: How do drawings cued by prototypical exemplars differ from drawings cued by category labels?_. It also constructs the barplots used in `Figure 3`.
- `classdata.py` houses the `Data` class, which allows for cleaner code in `kiddraw_analysis_2.ipynb`.

_______________________________________________________________

## `photodraw2x2`

- `photodraw_2x2_setup.ipynb` turns the data stored in MongoDB from the `photodraw2x2` experiment into three tidy dataframes, stored in the `/results` directory: 
  - `photodraw2x2_sketch_data.csv`
  - `photodraw2x2_stroke_data.csv`
  - `photodraw2x2_survey_data.csv`
- `photodraw_2x2_analysis.ipynb` contains exploratory analyses for the `photodraw2x2` experiment, such as barplots for number of strokes, sketch time, total ink, fc6-level recognizability factored on cue-type and representational goals. We also expore the relationship between classification accuracy and the above low-level variables. We then explore the ratio between category-level variable and image variability in the photo-cued drawings, and constructed representational dissimilarity matrices (RDMs) examining the correlations between our different factors. Constructs figure 5(E) in section "How does photo-cue typicality relate to sketch recognizability?"
- `photodraw_2x2_analysis_jefan.ipynb` constructs the dataframe `photodraw2x2_category_by_experiment_variances.csv`, which is passed into R for constructing figure 5(D). It examines the within-category variance for each 2x2 factor. 
- `photodraw_2x2_analysis_R.Rmd` conducts inferential statistics, fitting the linear mixed-effects models found in _Study 2: Disentangling the contributions of
sensory information, goals, and typicality_. It also constructs the plots used in `Figure 5`.

_______________________________________________________________

## Miscellaneous

#### `upload_data.ipynb`

Uploads data into public amazon s3 bucket, so that data can be loaded in using `download_daata.py`.

#### `utils,py`

Contains helper functions used throughout the photodraw analysis notebooks. 

#### `embeddings.py` 

Embeds raw pixels features in VGG-19 feature space. Can specify which layer to embed into.

#### `extract_features.py`

Command line script which can be used to convert raw images (sketches) to VGG-19 features.
