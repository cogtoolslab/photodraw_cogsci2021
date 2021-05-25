This directory contains the code necessary to run the `kiddraw` experiment. 

![](https://github.com/cogtoolslab/photodraw_cogsci2021/blob/master/experiments/kiddraw/stimuli/photo_cue_demo.png =250x)

### How to run the experiment
- SSH into user@cogtoolslab.org 
- run `node app.js --gameport XXXX` in the kiddraw directory
- run `node store.js` in the kiddraw directory
- navigate to https://cogtoolslab.org:XXXX/index.html to demo the experiment

Note: you need to run `npm install` to get `node-modules` in the experiment directory.
