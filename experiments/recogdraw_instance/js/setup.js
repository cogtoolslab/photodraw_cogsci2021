// Define experiment metadata object
function Experiment () {
  this.type = 'image-button-response';
  this.dbname = 'photodraw';
  this.colname = 'recogdraw-instance' // I really don't like this name
  this.iterationName = 'run0';   //pilot0 remember to change to correct iteration name when running study   livetest1
  this.devMode = false; // Change this to TRUE if testing in dev mode (short trial videos) or FALSE for real experiment
  this.force_wait = 1000;
}

function setupGame() {
  var socket = io.connect();
  socket.on("onConnected", function(d) { 
    console.log('connection data:', d)

    // get experiment ID information from URL
    var queryString = window.location.search;
    var urlParams = new URLSearchParams(queryString);
    var prolificID = urlParams.get('PROLIFIC_PID')   // ID unique to the participant
    var studyID = urlParams.get('STUDY_ID')          // ID unique to the study
    var sessionID = urlParams.get('SESSION_ID')      // ID unique to the particular submission
    
    var main_on_finish = function(data) {
      socket.emit('currentData', data);
      console.log('emitting trial data', data)
    }

    var additionalInfo = {
      prolificID: prolificID,
      studyID: studyID,
      sessionID: sessionID,  // this is the same as gameID
      gameID: d.gameid,   // we can replace with sessionID if we want to
      on_finish: main_on_finish
    }  

    // Create raw trials list
    function createTrialsList(callback) {                          // need to eventually change batch_num to pull from mongo
      rawTrials = _.map(_.shuffle(metadata32.filter(o => o.batch_num == d.batchNum)), function(n, i) {
        return trial = _.extend({}, new Experiment, {
            batch_num: d.batchNum,
            sketcher_gameID: n.gameID,
            sketcher_goal: n.goal, 
            sketcher_condition: n.condition,
            sketcher_category: n.category,
            sketcher_image_id: n.image_id,
            sketch_id: n.sketch_filename,
            nearest_photo_filenames: n.nearest_photo_filenames,
            true_photo_filename: n.true_photo_filename,
            true_photo_s3_url: n.true_photo_s3_url,          


            prompt: "<p id = promptid>Which image does this drawing depict?</p>",
            stimulus: n.sketch_s3_url,
            choices: _.shuffle(JSON.parse(n.nearest_photo_s3_urls.replace(/'/g, '"'))), // convert string representation into array of URLs
               
            catch_trial: false,
            prep_trial: false,
            post_trial_gap: 500,
          }
        );
      })
      callback(rawTrials) // add catch trials
    };

    function createCatchTrials(callback) {
      // manually create a catch trial metadata object
      catch_paths = [{'category': 'camel', 'stim_path': 'stimuli/catch_trials/catch_0/camel_catch.jpg', 
                      'choices': Array.from(Array(8).keys()).map(x => 'stimuli/catch_trials/catch_0/camel_choice_' + x + '.jpg')}, 
                      {'category': 'giraffe', 'stim_path': 'stimuli/catch_trials/catch_1/giraffe_catch.jpg', 
                      'choices': Array.from(Array(8).keys()).map(x => 'stimuli/catch_trials/catch_1/giraffe_choice_' + x + '.jpg')}, 
                      {'category': 'helicopter', 'stim_path': 'stimuli/catch_trials/catch_2/helicopter_catch.jpg', 
                      'choices': Array.from(Array(8).keys()).map(x => 'stimuli/catch_trials/catch_2/helicopter_choice_' + x + '.jpg')}, 
                      {'category': 'rocket', 'stim_path': 'stimuli/catch_trials/catch_3/rocket_catch.jpg', 
                      'choices': Array.from(Array(8).keys()).map(x => 'stimuli/catch_trials/catch_3/rocket_choice_' + x + '.jpg')}
                    ]

      prep_paths = [{'category': 'camel', 'stim_path': 'stimuli/prep_trials/chicken/chicken_instanceprep_sketch.jpg', 
                      'choices': Array.from(Array(8).keys()).map(x => 'stimuli/prep_trials/chicken/chicken_instanceprep_' + x + '.jpg')}, 
                     {'category': 'giraffe', 'stim_path': 'stimuli/prep_trials/guitar/guitar_instanceprep_sketch.jpg', 
                      'choices': Array.from(Array(8).keys()).map(x => 'stimuli/prep_trials/guitar/guitar_instanceprep_' + x + '.jpg')}, 
                     {'category': 'helicopter', 'stim_path': 'stimuli/prep_trials/hermitcrab/hermitcrab_instanceprep_sketch.jpg', 
                      'choices': Array.from(Array(8).keys()).map(x => 'stimuli/prep_trials/hermitcrab/hermitcrab_instanceprep_' + x + '.jpg')}, 
                     {'category': 'rocket', 'stim_path': 'stimuli/prep_trials/sword/sword_instanceprep_sketch.jpg', 
                      'choices': Array.from(Array(8).keys()).map(x => 'stimuli/prep_trials/sword/sword_instanceprep_' + x + '.jpg')}
                  ]

      // make list of catch trials in same format as the other trials
      catchtrials = _.map(catch_paths, function(n,i) {
        return trial = _.extend({}, new Experiment, {
          batch_num: d.batchNum,
          category: n.category,
          prompt: "<p id = promptid>Which image does this drawing depict?</p>",
          stimulus: n.stim_path,
          choices: _.shuffle(n.choices),

          catch_trial: true,
          prep_trial: false,
          post_trial_gap: 500
        });
      })

      preptrials = _.map(prep_paths, function(n,i) {
        return trial = _.extend({}, new Experiment, {
          batch_num: d.batchNum,
          category: n.category,
          prompt: "<p id = promptid>Which image does this drawing depict?</p>",
          stimulus: n.stim_path,
          choices: _.shuffle(n.choices),

          catch_trial: false,
          prep_trial: true,
          post_trial_gap: 500
        });
      })
  
      // add catch trials to trial list, evenly distributed
      before_length = rawTrials.length
      catchtrials.forEach(function(n, i) {
        rawTrials.splice((i+1)*(before_length / catchtrials.length), 0, n)   // trials.length / catchtrials.length
      })

      for (var i = 0; i < preptrials.length; i++) {
        rawTrials.unshift(preptrials[i]);
      };

      // add trialNum to trial list with catch trials included now
      rawTrials = rawTrials.map((n,i) => {
                var o = Object.assign({}, n);
                o.trialNum = i
                return o
      })

      var trials = _.flatten(_.map(rawTrials, function(trialData, i) {
        var trial = _.extend({}, additionalInfo, trialData, {trialNum: i}); 
          return trial;
        })); 	  
      callback(trials);
    };

    // Define consent form language             
    consentHTML = {    
      'str1' : '<p> Hello! In this study, you will be asked to match various sketches to their source image! </p><p> We expect the average game to last about 25 minutes, including the time it takes to read these instructions. For your participation in this study, you will be paid $5.00.</p><i><p> Note: We recommend using Chrome. We have not tested this study in other browsers.</p></i>',
      'str2' : ["<u><p id='legal'>Consenting to Participate:</p></u>",
    "<p id='legal'>By completing this study, you are participating in a study being performed by cognitive scientists at the University of California, San Diego, USA. If you have questions about this research, please contact the <b>Cognitive Tools Lab</b> at <b><a href='mailto://cogtoolslab.requester@gmail.com'>cogtoolslab.requester@gmail.com</a></b>. You must be at least 18 years old to participate. There are neither specific benefits nor anticipated risks associated with participation in this study. Your participation in this research is voluntary. You may decline to answer any or all of the following questions. You may decline further participation, at any time, without adverse consequences. Your anonymity is assured; the researchers who have requested your participation will not reveal any personal information about you.</p>"].join(' ')
    }
    // Define instructions language
    instructionsHTML = {  
      'str1' : "<p id = 'tightinstruction'> We are interested in your ability to recognize a drawing --- specifically, <i> how quickly and accurately you can match a drawing to its source image</i>.</p> <p> In total, you will be asked to match 136 sketches.</p>",
      'str2' : '<p id = "exampleprompt"> On each trial you will be shown a drawing and eight images. Your job will be to select the image that best represents the drawing as quickly as you can. Here is an example of a trial: <br> <img id="exampleimg" height = "500" src = "stimuli/recogdraw_instance_exampletrial.png"></p>',
      'str3' : "<p> Please adjust your screen (by zooming in/out) such that the drawings and images are not blocked in any way.</p> <p>In total, this study should take around 25 minutes. Once you are finished, the study will be automatically submitted for approval. If you encounter a problem or error, please send us an email <a href='mailto://cogtoolslab.requester@gmail.com'>(cogtoolslab.requester@gmail.com)</a> and we will make sure you're compensated for your time. Thank you again for contributing to our research! Let's begin! </p>"
    }  

  
    // Create consent + instructions instructions trial
    var welcome = {
      type: 'instructions',
      pages: [
        consentHTML.str1,
        consentHTML.str2,
        instructionsHTML.str1,
        instructionsHTML.str2,
        instructionsHTML.str3,
      ],
      force_wait: 2000, 
      show_clickable_nav: true,
      allow_keys: false,
      allow_backward: false
    };

    // exit survey trials
    var surveyChoiceInfo = _.omit(_.extend({}, additionalInfo, new Experiment),['type','dev_mode']);  
    var exitSurveyChoice = _.extend( {}, surveyChoiceInfo, {
      type: 'survey-multi-choice',
      preamble: "<strong><u>Exit Survey</u></strong>",
      questions: [
        {prompt: "What is your sex?",
         name: "participantSex",
         horizontal: false,
         options: ["Male", "Female", "Neither/Other/Do Not Wish To Say"],
         required: true
        },
        {prompt: "Which of the following did you use to make these labels?",
         name: "inputDevice",
         horizontal: false,
         options: ["Mouse", "Trackpad", "Touch Screen", "Stylus", "Other"],
         required: true
        }
      ],
      on_finish: main_on_finish
    });

    // Add survey page after trials are done
    var surveyTextInfo = _.omit(_.extend({}, additionalInfo, new Experiment),['type','dev_mode']);
    var exitSurveyText =  _.extend({}, surveyTextInfo, {
      type: 'survey-text',
      preamble: "<strong><u>Exit Survey</u></strong>",
      questions: [
      { name: 'participantAge', 
        prompt: "Please enter your age", 
        placeholder: "e.g. 32",
        required: true
      },        
      { name: 'participantComments', 
        prompt: "Thank you for participating in our study! Do you have any other comments or feedback to share with us about your experience?", 
        placeholder: "I had a lot of fun!",
        rows: 5, 
        columns: 50,
        required: false
      }
      ],
      on_finish: main_on_finish
    });    

    // Create goodbye trial (this doesn't close the browser yet)
    var goodbye = {
      type: 'instructions',
      pages: [
        'Thanks for participating in our experiment! You are all done now. Please click the button to be redirected to the prolific app (this will record your completion of the study).'
             ],
      show_clickable_nav: true,
      allow_backward: false,
      button_label_next: '< submit',    
      on_finish: () => { 
        window.open("https://app.prolific.co/submissions/complete?cc=1D15EA38","_self")
      }
    }


    function addBookends(trials) {
      // // add welcome trial to start of survey
      trials.unshift(welcome);
      
      // // append exit surveys
      trials.push(exitSurveyChoice);
      trials.push(exitSurveyText);

      // append goodbye trial
      trials.push(goodbye);


      jsPsych.init({
          timeline: trials,
          default_iti: 1000,
          show_progress_bar: true
      });    
    }


    // create trials list and add instrutions and exit survey
    createTrialsList(function (rawTrials) {
      createCatchTrials(function (trials) {
        addBookends(trials);
      })
    })
  });
}


