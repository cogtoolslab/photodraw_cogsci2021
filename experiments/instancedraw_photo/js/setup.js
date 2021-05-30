// Define experiment metadata object
function Experiment () {
  this.type = 'jspsych-cued-drawing';
  this.dbname = 'photodraw';
  this.colname = 'instancedraw-photo'
  this.iterationName = 'closed';
  this.devMode = true; // Change this to TRUE if testing in dev mode or FALSE for real experiment
}

// Define session metadata object 
function Session (index) {
  // Create raw trials list
  this.trials = _.map(_.shuffle(metadata32.filter(o => o.index == index)), function (n,i) {
    return trial = _.extend({}, new Experiment, { 
        batch_num: index,
        category: n.category,
        trialNum: i,
        numTrials: metadata32.filter(o => o.index == index).length,
        condition: 'photo', 
        imageURL: n.s3_url  
        }
      )
  }.bind(this))
}

// main function for running the experiment
function setupGame() {
  socket.on('onConnected', function(d) {
    console.log('data recieved:', d)
    
    // At end of each trial save score locally and send data to server
    var main_on_finish = function(data) {
      socket.emit('currentData', data);
      console.log('emitting trial data', data);
    }

    // get experiment ID information from URL
    var queryString = window.location.search;
    var urlParams = new URLSearchParams(queryString);
    var prolificID = urlParams.get('PROLIFIC_PID')   // ID unique to the participant
    var studyID = urlParams.get('STUDY_ID')          // ID unique to the study
    var sessionID = urlParams.get('SESSION_ID')      // ID unique to the particular submission

    //var SONA_ID = urlParams.get('id') // SONA-assigned anonymous ID. Note: needs study to be entered in the form https://cogtoolslab.org:XXXX/index.html?id=%SURVEY_CODE%

            
    // Add additional boilerplate info to each trial object
    var additionalInfo = {
      gameID: d.gameid,
      prolificID : prolificID,
      studyID : studyID,
      sessionID : sessionID,
      //SONA_ID: SONA_ID,
      on_finish: main_on_finish
    }    

    // Create trial list
    var session = new Session(d.indexNum); 
    var trials = _.flatten(_.map(session.trials, function(trialData, i) {
      var trial = _.extend({}, additionalInfo, trialData, {trialNum: i});
      return trial;
    }));
    var practiceTrial = _.extend({}, new Experiment, {
      category: 'face',//'Make a sketch! &#8594',
      condition: 'photo',
      practiceTrial: true  
    });
    
    // Define consent form language
    consentHTML = {
      'str1' : '<p> Hello! In this study, you will make some drawings of objects! </p><p> We expect the average game to last about 30 minutes, including the time \
                it takes to read these instructions. For your participation in this study, you will be paid $6.00.</p><i><p> Note: We recommend using Chrome. We have not tested \
                this experiment in other browsers.</p></i>',
      'str2' : ["<u><p id='legal'>Consenting to Participate:</p></u>",
    "<p id='legal'>By completing this session, you are participating in a study being performed by cognitive scientists at UC San Diego. If you have questions about this research, please\
     contact the <b>Cognitive Tools Lab</b> at <b><a href='mailto://cogtoolslab.requester@gmail.com'>cogtoolslab.requester@gmail.com</a></b>. You must be at least 18 years old to participate.\
      There are neither specific benefits nor anticipated risks associated with participation in this study. Your participation in this research is voluntary. You may decline to answer any \
      or all of the following questions. You may decline further participation, at any time, without adverse consequences. Your anonymity is assured; the researchers who have requested your\
       participation will not reveal any personal information about you.</p>"].join(' ')
    }
    // Define instructions language
    instructionsHTML = {
      'str1' : '<p>In this study, you will be making drawings of various objects, given an image. Your goal is to make drawings that look like the <b>specific</b> object that you were shown.</p>\
                <p>For example, suppose we asked you to draw a face. Instead of drawing a generic smiley face, you will be shown a <b>specific person’s face</b> to make a drawing of. Importantly, someone should be able to guess which person’s face you were shown, a lineup of different faces. \
                 You do not, however, need to be concerned about making them look pretty.</p>\
                <img height = "300" src = "stimuli/instances_only.png">',
      'str2': '<p>Also, when making your drawing, please do not shade or add any words, arrows, numbers, or surrounding context around your object drawing. For example, if you are drawing a horse, please do not draw grass around it.<p/>\
                <img height = "300" src = "stimuli/not_allowed_added_shading.png">',
      'str3': '<p>On every trial, you will be shown an image (e.g., of a face) for 8 seconds. After the 8 seconds is up, you will produce a drawing of that <b>specific</b> image you were just shown:</p>\
                <img height = "300" src = "stimuli/instance_photo_screencap.gif">\
                <p>Although you will have as much time as you need to make your drawing, you won’t be able to erase or ‘undo’ any part of your drawing while you are making it. \
                So please do your best to think about how you want your drawing to look before you begin each one. When you are satisfied with the drawing, please click SUBMIT.</p>',
      'str4': '<p>Finally, please adjust your screen such that the drawing space is not blocked in any way. <br> Let\'s begin!</p>'
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
        instructionsHTML.str4
      ],
      force_wait: 1500, 
      show_clickable_nav: true,
      allow_keys: false,
      allow_backward: true
    }

    // Create comprehension check survey
    var comprehensionSurvey = {
      type: 'survey-multi-choice',
      preamble: "<strong>Comprehension Check</strong>",
      questions: [
        {
          prompt: "What should your goal be when making each drawing?",
          name: 'goalOfDrawing',  // "To make a drawing that looks like a generic version of each object, but not any specific example."
          options: ["To make a drawing that is recognizable, but not one that could be matched to the image I was shown.", "To make a drawing that looks pretty!", "To make a drawing that can be matched to the image I was shown"],
          required: true
                  },
          {
        prompt: "Should you shade or add words, arrows, or any surrounding context to your drawing?", 
        name: 'bannedDrawings',
        options: ["Yes", "No"],
        required: true
          },
        //   {
        // prompt: "You will be shown an image representing the category we want you to draw. Should someone be able to associate your drawing with that image, from a lineup of similar images?",
        // name: 'instanceLevelPhoto',
        // options: ["Yes. I should make my drawing identifiable at the level of a specific image.", "No. I should make my drawing identifiable as a member of a target category, but not look like the specific image I was shown."],
        // required: true
        //   },          
                {
        prompt: "Can you undo or erase things you already drew?",
        name: 'canUndo',
        options: ["Yes, I am able to undo or erase things I already drew.", "No, I won't be able to erase or undo my work once I begin."],
        required: true
                }
      ]
    }

    // Check whether comprehension check is answered correctly
    var loopNode = {
      timeline: [comprehensionSurvey],
      loop_function: function(data) {
          resp = JSON.parse(data.values()[0]['responses']);
          if ((resp["bannedDrawings"] == 'No' 
            && resp["goalOfDrawing"] == "To make a drawing that can be matched to the image I was shown" 
            // && resp['instanceLevelPhoto'] == "Yes. I should make my drawing identifiable at the level of a specific image."
            && resp['canUndo'] == "No, I won't be able to erase or undo my work once I begin.")) { 
              return false;
          } else {
              alert('Try again! One or more of your responses was incorrect.');
              return true;
        }
      }
    }

    // Create goodbye trial (this doesn't close the browser yet)
    var goodbye = {
      type: 'instructions',
      pages: [
        'Thanks for participating in our experiment! You are all done. Please click the button to submit this study.'
      ],
      show_clickable_nav: true,
      allow_backward: false,
      on_finish: () => {
        // do something to give credit to participants 
        //completion_url = "https://ucsd.sona-systems.com/webstudy_credit.aspx?experiment_id=1989&credit_token=c41f43c84bd44f4f8412479930659e9c&survey_code=" + SONA_ID 
        window.open("https://app.prolific.co/submissions/complete?cc=1BD370CB","_self")
      }
    }

    
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
        {prompt: "Which of the following did you use to make your drawings?",
         name: "inputDevice",
         horizontal: false,
         options: ["Mouse", "Trackpad", "Touch Screen", "Stylus", "Other"],
         required: true
        },
        {prompt: "How skilled do you consider yourself to be at drawing? (1: highly unskilled; 7: highly skilled)",
         name: "subjectiveSkill",
         horizontal: false,
         options: ["1","2","3","4","5","6","7"],
         required: true
        },
      ],
      on_finish: main_on_finish
    });

    
    // Add survey page after trials are done
    var surveyTextInfo = _.omit(_.extend({}, additionalInfo, new Experiment),['type','dev_mode']);
    var exitSurveyText =  _.extend({}, surveyTextInfo, {
      type: 'survey-text',
      preamble: "<strong><u>Exit Survey</u></strong>",
      questions: [
      {name: "TechnicalDifficultiesFreeResp",
        prompt: "If you encountered any technical difficulties, please briefly describe the issue.",
        placeholder: "I did not encounter any technical difficulities.",
        rows: 5, 
        columns: 50, 
        required: false
      },
      { name: 'participantAge', 
        prompt: "What is your year of birth?", 
        placeholder: "e.g. 1766", 
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

    // insert comprehension check 
    trials.unshift(loopNode);
    // insert welcome trials before check
    trials.unshift(welcome);
    // insert practice trial
    // trials.unshift(practiceTrial);   

    // insert exit surveys
    trials.push(exitSurveyChoice);
    trials.push(exitSurveyText);
    
    // append goodbye trial
    trials.push(goodbye);

    // create jspsych timeline object
    jsPsych.init({
      timeline: trials,
      default_iti: 1000,
      show_progress_bar: true
    });
      
  });


}
