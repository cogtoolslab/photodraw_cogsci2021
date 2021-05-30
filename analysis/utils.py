import os
import base64
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from io import BytesIO
from itertools import product
from matplotlib import pyplot, gridspec
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import RepeatedKFold, cross_validate

plt = pyplot
sns.set_context('talk')
sns.set_style('white')


# for fun
def progressBar(current, total, barLength=20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


def render_images(D, 
                 metadata = ['condition', 'category'],
                 out_dir = './sketches',
                 delimiter = '_',
                 overwrite = True,
                 clear = True):
    '''
    input: dataframe D containing png data (see data keyword argument)
           and list of metadata attributes (see metadata keyword argument)
           out_dir = which directory to save the pngs to
           delimiter = when constructing each filename, what character to stick in between each attribute
    output: return list of PIL Images;
            saves images out as PNG files to out_path 
            where each filename is constructed from concatenating metadata attributes
    '''
    for i, d in D.iterrows():
        # convert pngData string into a PIL Image object
        im = Image.open(BytesIO(base64.b64decode(d['pngData']))).resize((224, 224))

        # construct the filename by concatenating attributes
        attributes = [str(d[attr]) for attr in metadata]
        fname = delimiter.join(attributes)        

        # create the out_dir if it does not already exist
        if not os.path.exists(out_dir): 
            os.makedirs(out_dir)

        # now save the image out to that directory
        if (overwrite or not os.path.exists(os.path.join(out_dir, fname+'.png'))):
            progressBar(i+1, D.shape[0])
            im.save(os.path.join(out_dir, fname+'.png'), 'PNG')
        else:
            print('Skipping {} | {} of {}'.format(d['category'], i + 1, D.shape[0])) 
        if clear:
            clear_output(wait=True) 
    print('')
    print('Done rendering {} images to {}.'.format(D.shape[0],out_dir))


# preprocesses images by flagging some data as invalid
def preprocess_sketches(K):
    '''
    input: sketch data K
    output: transformed data K
    '''
    invalidGameIDs = ['4434-0c539391-2e3b-41f1-bfae-dc3210e5d889',
                      '2711-2e1c7608-7333-42c4-b127-2e5a9697a420',
                      '5805-a0808dca-f32e-4fbd-9889-b0ccdc4eb867',
                      '9396-ba434979-1e51-4621-8db7-c96fb08e9d46']
    K_exclusions = ['numStrokes', 'activeSketchTime','totalInk']
    K_means = [K[label].describe()[1] for label in K_exclusions]
    K_sds = [K[label].describe()[2] for label in K_exclusions]

    outliers = [None] * len(K)
    invalid = [None] * len(K)
    for index, row in K.iterrows():
        # mark sketches as outliers if values are 4 standard deviations over (right skewed data)
        for ind, label in enumerate(K_exclusions):
            if row[label] > K_means[ind] + 4 * K_sds[ind]:
                outliers[index] = True
                break
            else:
                outliers[index] = False

        # mark sketches as invalid if values are from the list of invalid game IDs
        for ID in invalidGameIDs:
            if K['gameID'][index] == ID:
                invalid[index] = True
                break
            else:
                invalid[index] = False
    K = K.assign(isOutlier=outliers)
    K = K.assign(isInvalid=invalid)

    return K


def feature_heatmap(data, abstraction, metric='euclidean'):
    ordered = data[3::4].append(data[~data['cue_id'].str.contains('text')]).reset_index(drop=True)
    if metric != 'correlation':
        frame = pd.DataFrame(squareform(pdist(np.stack(ordered['mean_feature'].values), metric=metric)), columns=ordered['cue_id'].values)
    else:
        frame = pd.DataFrame(data=np.corrcoef(np.stack(ordered['mean_feature'].values)), columns=ordered['cue_id'].values)
    frame.index = ordered['cue_id'].values

    plt.figure(figsize=(18, 25))
    sns.heatmap(frame, cbar_kws={'orientation': 'horizontal'})
    plt.xlabel('cue ids'), plt.ylabel('cue ids')
    if metric != 'correlation':
        plt.title(f'Pairwise {metric} distance of mean feature vectors of each cue id ({abstraction})', fontsize=26)
    else:
        plt.title(f'Pairwise correlation coefficients of mean feature vectors of each cue id ({abstraction})', fontsize=26)


def compute_f_stat(mean_features):
    mean_photo_id_features = mean_features[mean_features.photoid != 'text']
    mean_category_features = pd.DataFrame(mean_photo_id_features.groupby(['category'])['mean_feature'].apply(np.mean))
    overall_photo_cue_features = mean_category_features.mean_feature.mean(axis=0)
    between_group = 3 * sum((np.linalg.norm(features - overall_photo_cue_features))**2 for features in mean_category_features.mean_feature.values) / (12-1)
    within_group = []
    for category, category_features in mean_category_features.iterrows():
        for j in ['1', '2', '3']:
            Y_i_j = mean_photo_id_features[(mean_photo_id_features.category == category) & (mean_photo_id_features.photoid == j)].mean_feature.values[0]
            Y_i = category_features.mean_feature
            within_group.append((np.linalg.norm(Y_i_j-Y_i))**2 / (36 - 12))
    within_group = np.sum(within_group, axis=0)

    F_stat = (between_group / within_group)
    return F_stat


def between_condition_RDM(features, abstraction, metric='euclidean', get_upper_triangulars=False):
    mean_photo_id_features = features[features.condition == 'photo']
    mean_category_features = pd.DataFrame(mean_photo_id_features.groupby(['category'])['mean_feature'].apply(np.mean))
    mean_category_features['category'] = mean_category_features.index
    mean_category_features['condition'] = 'photo'
    mean_category_features = mean_category_features.reset_index(drop=True)
    rdm_matrix_values = pd.concat([features[features.condition == 'text'], mean_category_features]).reset_index(drop=True)
    rdm_matrix_values = rdm_matrix_values.assign(cat_cond=rdm_matrix_values.category + '_' + rdm_matrix_values.condition)
    if metric != 'correlation':
        rdm_cat_cond_dists = pd.DataFrame(squareform(pdist(np.stack(rdm_matrix_values['mean_feature'].values), metric=metric)), columns=rdm_matrix_values['cat_cond'].values)
    else:
        rdm_cat_cond_dists = pd.DataFrame(np.corrcoef(np.stack(rdm_matrix_values['mean_feature'].values)), columns=rdm_matrix_values['cat_cond'].values)

    if get_upper_triangulars == True:
        return np.triu(rdm_cat_cond_dists.to_numpy()[0:12, 0:12])[np.triu_indices(12,1)], np.triu(rdm_cat_cond_dists.to_numpy()[12:24, 12:24])[np.triu_indices(12, 1)]

    rdm_cat_cond_dists.index = rdm_matrix_values['cat_cond'].values

    plt.figure(figsize=(18, 25))
    sns.heatmap(rdm_cat_cond_dists, cbar_kws={'orientation':'horizontal'})
    plt.xlabel('category-condition pairs'), plt.ylabel('category-condition pairs')
    if metric != 'correlation':
        plt.title(f'Pairwise {metric} distance of mean feature vectors of each category-condition pair ({abstraction})', fontsize=26)
    else:
        plt.title(f'Pairwise correlation coefficients of mean feature vectors of each category-condition pair ({abstraction})', fontsize=26)
    plt.hlines(12, xmin=0, xmax=24)
    plt.vlines(12, ymin=0, ymax=24)


def photoid_sd_barplots(data, variable, prettyname):
    plt.figure(figsize=(14, 6))
    result = data.groupby(['category','photoid'])[variable].aggregate(np.std).reset_index().sort_values(['category', 'photoid']).reset_index(drop=True)
    sns.barplot(data=result, x='category', y=variable, hue='photoid')
    plt.ylabel(f'{prettyname} (standard deviation)'),plt.title(f'Standard deviation of {prettyname} within each photo-id')


def photoid_sd_distplots(data, variable, prettyname):
    result = data.groupby(['category','photoid'])[variable].aggregate(np.std).reset_index().sort_values(['category','photoid']).reset_index(drop=True)
    sns.distplot(result[result['photoid'] != 'text'][variable], label='photo')
    sns.distplot(result[result['photoid'] == 'text'][variable], label='text')
    plt.legend()
    plt.xlabel(f'{prettyname} (std) within each photo-id')
    plt.ylabel('Density')
    plt.title(f'Distibution of standard deviation of {prettyname} per sketch')


def generate_acc_probs(features, metadata, num_splits=5, num_repeats=1, alt_labels=None):
    # initialize lists to be appended to
    prob_arrs, prediction, truths, class_probs = [], [], [], []
    if alt_labels is None:
        classes = ['category',
                   'category',
                   ['airplane', 'bike','bird','car', 'cat', 'chair', 'cup', 'hat', 'house', 'rabbit', 'tree', 'watch']]
    else:
        classes = alt_labels

    rkf = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats)
    logit = LogisticRegression(max_iter=1000)

    # perform repeated cross validation to extract class probabilities and prediction scores
    for train_index, test_index in rkf.split(features):
        model = logit.fit(features[train_index], metadata[classes[0]][train_index])
        probs = model.predict_proba(features[test_index])
        preds = model.predict(features[test_index])
        actual = metadata[classes[0]][test_index].values
        prob_arrs.extend(list(zip(actual, probs)))
        prediction.extend(preds)
        truths.extend(actual)

    # turn raw lists into a more manageable dataframe to make heatmaps from
    frame = pd.DataFrame(prob_arrs, columns=[classes[1], 'class_probs'])

    if alt_labels == None:
        for i in range(12):
            class_probs.append(frame[frame.category == classes[2][i]]['class_probs'].values.mean(axis=0))
    else: 
        for i in sorted(frame[classes[1]].unique()):
            class_probs.append(frame[frame[classes[1]] == i]['class_probs'].values.mean(axis=0))
    class_probs = np.array(class_probs)
    class_probs = pd.DataFrame(class_probs, columns=classes[2], index=classes[2])
    acc_scores = pd.DataFrame(confusion_matrix(truths, prediction), columns=classes[2], index=classes[2])

    return class_probs, acc_scores


def generate_acc_probs_2x2(features, metadata, num_splits=5, num_repeats=1):
    # divide data into their correct categories
    photo_features = features[metadata[metadata['condition'] == 'photo'].index] 
    photo_labels = metadata[metadata['condition'] == 'photo']['cat_codes'].values
    text_features = features[metadata[metadata['condition'] == 'text'].index]
    text_labels = metadata[metadata['condition'] == 'text']['cat_codes'].values
    classes = ['airplane', 'bike', 'bird', 'cat', 'car', 'chair', 'cup', 'hat', 'house', 'rabbit', 'tree', 'watch']

    # initialize lists to be appended to (order is *_traindata_testdata)
    prob_arrs_photo_photo, prob_arrs_photo_text, prob_arrs_text_text, prob_arrs_text_photo  = [], [], [], []
    prediction_photo_photo, prediction_photo_text, prediction_text_text, prediction_text_photo = [], [], [], []
    truths_photo_photo, truths_photo_text, truths_text_text, truths_text_photo = [], [], [], []
    class_probs_photo_photo, class_probs_photo_text, class_probs_text_text, class_probs_text_photo = [], [], [], []

    rkf = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats)
    logit = LogisticRegression(max_iter=1000)

    # perform repeated cross validation to extract class probabilities and prediction scores
    for train_index, test_index in rkf.split(photo_features): 
        model = logit.fit(photo_features[train_index], photo_labels[train_index])

        probs_photo, probs_text = model.predict_proba(photo_features[test_index]), model.predict_proba(text_features)
        preds_photo, preds_text = model.predict(photo_features[test_index]), model.predict(text_features)
        actual_photo, actual_text = photo_labels[test_index], text_labels

        prob_arrs_photo_photo.extend(list(zip(actual_photo, probs_photo)))
        prob_arrs_photo_text.extend(list(zip(actual_text, probs_text)))

        prediction_photo_photo.extend(preds_photo)
        prediction_photo_text.extend(preds_text)

        truths_photo_photo.extend(actual_photo)
        truths_photo_text.extend(actual_text)

    for train_index, test_index in rkf.split(text_features):
        model = logit.fit(text_features[train_index], text_labels[train_index])

        probs_text, probs_photo = model.predict_proba(text_features[test_index]), model.predict_proba(photo_features)
        preds_text, preds_photo = model.predict(text_features[test_index]), model.predict(photo_features)
        actual_text, actual_photo = text_labels[test_index], photo_labels

        prob_arrs_text_text.extend(list(zip(actual_text, probs_text)))
        prob_arrs_text_photo.extend(list(zip(actual_photo, probs_photo)))

        prediction_text_text.extend(preds_text)
        prediction_text_photo.extend(preds_photo)

        truths_text_text.extend(actual_text)
        truths_text_photo.extend(actual_photo)

    # turn raw lists into a more manageable dataframe to make heatmaps from
    frame_photo_photo = pd.DataFrame(prob_arrs_photo_photo, columns=['category','class_probs']) 
    frame_photo_text = pd.DataFrame(prob_arrs_photo_text, columns=['category','class_probs'])
    frame_text_text = pd.DataFrame(prob_arrs_text_text, columns=['category','class_probs'])
    frame_text_photo = pd.DataFrame(prob_arrs_text_photo, columns=['category','class_probs'])
    for i in range(12):
        class_probs_photo_photo.append(frame_photo_photo[frame_photo_photo.category == i]['class_probs'].values.mean(axis=0))
        class_probs_photo_text.append(frame_photo_text[frame_photo_text.category == i]['class_probs'].values.mean(axis=0))
        class_probs_text_text.append(frame_text_text[frame_text_text.category == i]['class_probs'].values.mean(axis=0))
        class_probs_text_photo.append(frame_text_photo[frame_text_photo.category == i]['class_probs'].values.mean(axis=0))
    class_probs_photo_photo, class_probs_photo_text, class_probs_text_text, class_probs_text_photo = np.array(class_probs_photo_photo), np.array(class_probs_photo_text), np.array(class_probs_text_text), np.array(class_probs_text_photo)
    class_probs_photo_photo, class_probs_photo_text = pd.DataFrame(class_probs_photo_photo, columns = classes, index=classes), pd.DataFrame(class_probs_photo_text, columns = classes, index=classes)
    class_probs_text_text, class_probs_text_photo = pd.DataFrame(class_probs_text_text, columns = classes, index=classes), pd.DataFrame(class_probs_text_photo, columns = classes, index=classes)

    acc_scores_photo_photo = pd.DataFrame(confusion_matrix(truths_photo_photo,prediction_photo_photo), columns=classes, index=classes)
    acc_scores_photo_text = pd.DataFrame(confusion_matrix(truths_photo_text,prediction_photo_text), columns=classes, index=classes)
    acc_scores_text_photo = pd.DataFrame(confusion_matrix(truths_text_photo,prediction_text_photo), columns=classes, index=classes)
    acc_scores_text_text = pd.DataFrame(confusion_matrix(truths_text_text,prediction_text_text), columns=classes, index=classes)
    
    # turn these big dataframes into a more acessible dict
    probslist = [class_probs_photo_photo, class_probs_photo_text, class_probs_text_photo, class_probs_text_text]
    accslist = [acc_scores_photo_photo, acc_scores_photo_text, acc_scores_text_photo, acc_scores_text_text]
    class_probs_dict = dict((a+'_'+b, lol) for lol,(a,b) in zip(probslist, product(['photo','text'],['photo','text'])))
    acc_scores_dict = dict((a+'_'+b, lol) for lol,(a,b) in zip(accslist, product(['photo','text'],['photo','text'])))
    
    return class_probs_dict, acc_scores_dict


def generate_2x2_plots(data, abstraction, metric):
    fig, axs = plt.subplots(nrows=2,ncols=2, figsize=(14,12), constrained_layout=True)
    sns.heatmap(data['photo_photo'], ax=axs[0, 0])
    sns.heatmap(data['photo_text'], ax=axs[0, 1])
    sns.heatmap(data['text_photo'], ax=axs[1, 0])
    sns.heatmap(data['text_text'], ax=axs[1, 1])
    axs[0, 0].set_ylabel('photo-cue trained classifier'), axs[1, 0].set_ylabel('text-cue trained classifier')
    axs[0, 0].set_xlabel('photo-cue test data'), axs[0, 1].set_xlabel('text-cue test data')
    axs[0, 0].xaxis.set_label_position('top'), axs[0, 1].xaxis.set_label_position('top')
    fig.text(0.52, -0.04, 'Predicted label probabilities', ha='center')
    fig.text(-0.04, 0.54, 'Correct label', va='center', rotation='vertical')
    axs[0, 0].xaxis.labelpad = 10
    axs[0, 1].xaxis.labelpad = 10
    if metric == 'probabilities':
        plt.suptitle(f'Prediction probabilities vs category label, by training & prediction data ({abstraction})');
    elif metric == 'confusion matrix':
        plt.suptitle(f'Confusion matrix for category predictions, by training & prediction data ({abstraction})');


def perform_cross_validation(features, labels, num_folds, input_type, prediction_type, output=False):
    # Perform cross-validation on an n-way softmax classifier to get baseline accuracies and to visualize data 
    logit = LogisticRegression(max_iter=1000)
    cv_results = cross_validate(logit,features,labels, cv=num_folds, return_estimator=True,return_train_score =True)
    print(f'Performing {num_folds}-fold cross validation:')
    print(f"This is a {len(set(list(labels)))}-way classifier trained on {input_type}, predicting {prediction_type}.")
    print(f"The average train accuracy of this logistic regression is {cv_results['train_score'].mean()}")
    print(f"The average test accuracy of this logistic regression is {round(cv_results['test_score'].mean(),3)}")
    print(f"This model took {round((cv_results['fit_time'] + cv_results['score_time']).sum(),3)} seconds to train")
    if output == True:
        return cv_results


def perform_cross_validation_twice(features, metadata, labels, num_folds, input_type, prediction_type, output=False):
    # Cross-validation on two 12-way softmax classifiers (photo-cue/text-cue trained) 
    # What is the average accuracy from 10-fold cross vaidation?
    logit_photo = LogisticRegression(max_iter=1000)
    logit_text = LogisticRegression(max_iter=1000)
    photo_features, photo_labels = features[metadata[metadata['condition'] ==  'photo'].index],labels[metadata['condition']=='photo'].values
    text_features, text_labels = features[metadata[metadata['condition'] ==  'text'].index],labels[metadata['condition']=='text'].values
    cv_results_photo = cross_validate(logit_photo, photo_features, photo_labels,
                                      cv=10, return_estimator=True,return_train_score=True)
    cv_results_text = cross_validate(logit_text, text_features, text_labels,
                                     cv=10, return_estimator=True,return_train_score=True)
    photo_results=np.mean([cv_results_photo['estimator'][i].score(text_features, text_labels) for i in range(10)])
    text_results=np.mean([cv_results_text['estimator'][i].score(photo_features, photo_labels) for i in range(10)])
    print(f'Performing {num_folds}-fold cross validation on two classifiers:')
    print(f"These are {len(set(list(labels)))}-way classifiers trained solely on photo-cue/text-cue {input_type}, predicting {prediction_type}.")
    print(f"The average train accuracy of the logistic regression trained on photo-cue data is {cv_results_photo['train_score'].mean()}")
    print(f"The average train accuracy of the logistic regression trained on text-cue data is {cv_results_text['train_score'].mean()}\n")

    print(f"When predicting photo-cue sketches, the average test accuracy of the logistic regression trained on photo-cue data is {round(cv_results_photo['test_score'].mean(),3)}")
    print(f'When predicting text -cue sketches, the average test accuracy of the logistic regression trained on photo-cue data is {round(photo_results,3)}')
    print(f"When predicting text -cue sketches, the average test accuracy of the logistic regression trained on text -cue data is {round(cv_results_text['test_score'].mean(),3)}")
    print(f'When predicting photo-cue sketches, the average test accuracy of the logistic regression trained on text -cue data is {round(text_results,3)}\n')

    print(f"This model took {round((cv_results_photo['fit_time'] + cv_results_photo['score_time'] + cv_results_text['fit_time'] + cv_results_text['score_time']).sum(),3)} seconds to train")
    
    if output == True:
        return cv_results_photo, cv_results_text

# basic level plots that show relations for either condition or category
def adjacent_plots(data1, data2, x=None, y=None, plottype=None):
    pretty_dict = {'activeSketchTime': 'Active sketching time (ms)', 
                   'numStrokes': 'Number of strokes', 
                   'totalInk': 'Total ink used', 
                   'arcLength': 'Arc length',
                   'prob_true_predict_fc6': 'Prediction probability',
                   'prob_true_predict_pixel': 'Prediction probability',
                   'prob_true_predict_fc6_logodds': 'Log-odd ratio',
                   'prob_true_predict_logodds': 'Log-odd ratio'} 
    if x == 'condition':
        fig, ax = plt.subplots(1,2, figsize=(6,5), sharey = True)
        getattr(sns, plottype)(data=data1, x=x, y=y, ax=ax[0]).set_title('with outliers')
        ax[0].set_ylabel(pretty_dict[y])
        
        getattr(sns, plottype)(data=data2, x=x, y=y, ax=ax[1]).set_title('outliers removed')
        ax[1].set_ylabel('') #pretty_dict[y]
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]), plt.suptitle(f'{pretty_dict[y]} by {x}');
    elif x == 'category':
        fig, axs = plt.subplots(2, figsize=(12,8))
        result = data1.groupby([x])[y].aggregate(np.mean).reset_index().sort_values(y)
        getattr(sns, plottype)(data=data1, x=x, y=y, order=result[x], ax=axs[0]).set(xlabel='', title='with outliers')

        result = data2.groupby([x])[y].aggregate(np.mean).reset_index().sort_values(y)
        getattr(sns, plottype)(data=data2, x=x, y=y, order=result[x], ax=axs[1]).set_title('outliers removed')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]), plt.suptitle(f'{pretty_dict[y]} by {x}');
    elif y == 'density':
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        sns.distplot(data1.loc[data1['condition'] == 'text'][[x]], hist=False, rug=True, ax=ax[0], label='text')
        sns.distplot(data1.loc[data1['condition'] == 'photo'][[x]], hist=False, rug=True, ax=ax[0], label='photo').set_title('with outliers')
        ax[0].set(xlabel=pretty_dict[x], ylabel=y), ax[0].legend()
        
        sns.distplot(data2.loc[data2['condition'] == 'text'][[x]], hist=False, rug=True, ax=ax[1], label='text')
        sns.distplot(data2.loc[data2['condition'] == 'photo'][[x]], hist=False, rug=True, ax=ax[1], label='photo').set_title('outliers removed')
        ax[1].set(xlabel=pretty_dict[x], ylabel=y), ax[1].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]), plt.suptitle(f'Density of {pretty_dict[x]}, by condition');
    elif plottype == 'regplot':
        fig, ax = plt.subplots(1,2, figsize=(12,5))

        result = data1.groupby(["gameID",'condition'])[x].aggregate(np.mean).reset_index().sort_values('gameID')
        result = result.pivot(index='gameID', columns='condition')[x].reset_index()
        sns.regplot(data=result, x='text', y='photo', marker='.', ax=ax[0]).set(title='with outliers')
        xline = np.linspace(result[['photo', 'text']].min().max(), result[['photo', 'text']].max().min(), 1000)
        ax[0].plot(xline, xline, color='gray')
        
        result = data2.groupby(["gameID",'condition'])[x].aggregate(np.mean).reset_index().sort_values('gameID')
        result = result.pivot(index='gameID', columns='condition')[x].reset_index()
        sns.regplot(data=result, x='text', y='photo', marker='.', ax=ax[1]).set(title='outliers removed')
        ax[1].plot(xline, xline, color='gray')

        fig.suptitle(f'Average {pretty_dict[x].lower()} per participant, over condition'), plt.tight_layout(rect=[0, 0.03, 1, 0.95]);

        return (result.photo - result.text).values
    else:
        print("Doesn't look like the input is valid.")


def cat_cond_diffplots(data1, data2, dv, plottype):
    pretty_dict = {'activeSketchTime': 'Active sketching time (ms)', 
                   'numStrokes': 'Number of strokes', 
                   'totalInk': 'Total ink used', 
                   'arcLength': 'Arc length'} 

    ordering = data1.groupby(['category'])[dv].aggregate(np.mean).reset_index().sort_values([dv]).reset_index(drop=True)['category']

    fig2 = plt.figure(constrained_layout=False, figsize=(12,14))
    spec2 = gridspec.GridSpec(ncols=1, nrows=10, figure=fig2, wspace=0.4)

    f2_ax1 = fig2.add_subplot(spec2[0:3, 0])
    sns.barplot(data = data1, x='category', y=dv, hue='condition', order=ordering, ci=95,ax=f2_ax1)
    f2_ax1.set_xlabel(''), f2_ax1.get_xaxis().set_visible(False), f2_ax1.set_ylabel(pretty_dict[dv])
    f2_ax1.legend(bbox_to_anchor=(1.175, 1.05), loc='upper right', prop={'size': 16})

    f2_ax2 = fig2.add_subplot(spec2[3:5, 0])
    result1 = data1.groupby(['category','condition'])[dv].aggregate(np.mean).reset_index().sort_values(['category','condition']).reset_index(drop=True)
    diff = result1.iloc[::2][dv].values - result1.iloc[1::2][dv].values
    sns.barplot(sorted(data1['category'].unique()), diff,ax=f2_ax2, order = ordering)
    f2_ax2.set_ylabel('difference'), f2_ax2.get_xaxis().set_visible(False)
    f2_ax2.axhline(alpha = 0.6, color = 'black', lw=.5)
    
    f2_ax3 = fig2.add_subplot(spec2[5:8, 0])
    sns.barplot(data = data2, x='category', y=dv, hue='condition', order=ordering, ci=95,ax=f2_ax3)
    f2_ax3.set_xlabel(''), f2_ax3.get_xaxis().set_visible(False), f2_ax3.set_ylabel(pretty_dict[dv]), f2_ax3.get_legend().remove()

    f2_ax4 = fig2.add_subplot(spec2[8:10, 0])
    result2 = data2.groupby(['category','condition'])[dv].aggregate(np.mean).reset_index().sort_values(['category','condition']).reset_index(drop=True)
    diff = result2.iloc[::2][dv].values - result2.iloc[1::2][dv].values
    sns.barplot(sorted(data2['category'].unique()), diff, ax=f2_ax4, order = ordering)
    f2_ax4.set_xlabel('category'),f2_ax4.set_ylabel('difference')
    f2_ax4.axhline(alpha = 0.6, color = 'black', lw=.5)
    
    plt.text(0.91, 0.62, "with outliers", rotation=270, transform=fig2.transFigure)
    plt.text(0.91, 0.2, "outliers removed", rotation=270, transform=fig2.transFigure)
    warnings.simplefilter("ignore")
    plt.suptitle(f'{pretty_dict[dv]} by category, over condition').set_position([.5, 0.92]), plt.tight_layout(rect=[0, 0.03, 1, 0.95]);

# Converts one pngData from K to flattened numpy image array
def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='double' )

def pngToArray(pngData):
    arr = np.array(Image.open(BytesIO(base64.b64decode(pngData))).resize((224,224)))
    rbgarr = np.array(rgba2rgb(arr)).flatten()
    return rbgarr



