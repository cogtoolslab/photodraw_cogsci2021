# Visual communication of object concepts at different levels of abstraction

Project investigating our ability to produce drawings of specific entities (e.g., "Garfield") as well as general categories (e.g., "cat"). 

To get started, first download the repository and run `download_data.py` in the home directory `/photodraw_cogsci2021`. Doing so will download tidy `*.csv` files, all participant sketches, experiment metadata, and other gallery files and plots for browsing. 

<p align="center" style="font-size: smaller">
  <img width="85%" src="https://github.com/cogtoolslab/photodraw_cogsci2021/blob/master/experiments/instancedraw_photo/stimuli/instance_photo_screencap.gif"></img><br/>
</p>

## Experiments


We ran a series of experiments examining how sensory information and representational goals jointly constrain the content and form of our drawings. In our first experiment (`experiments/kiddraw`) participants produced a total of 12 drawings, corresponding to 12 familiar basic-level categories, where 6 drawings were cued using a category label and the other 6 were cued using a typical exemplar from that category. 

In the second experiment (`photodraw2x2`) we independently manipulated sensory information (photo/text cue type) and representaional goals (to draw an exemplar vs. a category) to form a 2x2 factorial design. This time, participants drew 32 drawings corresponding, to 32 basic-level categories.

The individual subdirectories within `/experiments` provide greater detail on the implementation of the experiments. 

<p align="center" style="font-size: smaller">
  <img width="75%" src="https://github.com/cogtoolslab/photodraw_cogsci2021/blob/master/results/plots/photodraw2x2_gallery.png"></img><br/>
  Example cues and drawings in photodraw2x2 task
</p>

## Analysis


The `/analysis` directory can be largely categorized into a few main components. `*.Rmd` files are primarily used for inferential statistics and making plots to be used in papers, while `*.ipynb` files are primarily used for data preprocessing (`*_setup.ipynb`) and for exploratory data analysis `*_analysis.ipynb`.


<p align="center" style="font-size: smaller">
  <img width="95%" src="https://github.com/cogtoolslab/photodraw_cogsci2021/blob/master/results/plots/photodraw2x2_results.png"></img><br/>
  Main results of photodraw2x2
</p>


## Stimuli


The `/stimuli` directory contains stimulus preprocessing code for the project. Running `download_data.py` downloads the stimuli folders and files that would have been created by running the files in `/stimuli`.

<p align="center" style="font-size: smaller">
  <img width="100%" src="https://github.com/cogtoolslab/photodraw_cogsci2021/blob/master/results/plots/photodraw2x2_cats.png"></img><br/>
  All photo-cue cat stimuli in photodraw2x2 experiment, sorted by participant-rated typicality
</p>
