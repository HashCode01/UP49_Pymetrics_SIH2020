import sys
sys.path.insert(1, 'your_path/src/')
from definitions import *
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
sys.path.insert(1, 'your_path/src/')
from data.transformers import *
from helper.functions import *

if __name__ == '__main__':
    # Main dataframe generator
    locations = [DATA_DIR + '/raw/' + x for x in ['Tset_A','Tset_B']] #, 'training_B']]

    df = pd.read_csv(DATA_DIR + '/raw/' + 'Sep_AB_imputed.csv')
    df = df.rename(columns = {'Unnamed: 0':'time','pid':'id'})
    df = df.reset_index().set_index(['id', 'time']).sort_index(ascending=True)
    #obj1=CreateDataframe()
    #df=obj1.transform(location=locations)
    #df = load_pickle(DATA_DIR + '/preprocessed/from_raw/df.pickle')

    # Run full pipe
    data_pipeline = Pipeline([
        # ('create_dataframe', CreateDataframe(save=True)),
        #('input_count', AddRecordingCount()),
        ('imputation', CarryForwardImputation()),
        ('derive_features', DerivedFeatures()),
        ('fill_missing', FillMissing()),
        ('drop_features', DropFeatures()),
        ('min_maxxer', MinMaxSmoother()),
    ])
    df = data_pipeline.fit_transform(df)
    # Save
    df.to_csv(DATA_DIR + '/preprocessed/formatted/Sep_AB.csv')
    save_pickle(df, DATA_DIR + '/preprocessed/formatted/df.pickle')
    save_pickle(data_pipeline, MODELS_DIR + '/pipelines/data_pipeline.dill', use_dill=True)

    # Labels -> scores
    labels = load_pickle(DATA_DIR + '/processed/labels/original.pickle')
    scores = LabelsToScores().transform(labels)
    save_pickle(scores['utility'], DATA_DIR + '/processed/labels/utility_scores.pickle')
    save_pickle(scores, DATA_DIR + '/processed/labels/full_scores.pickle')

    # Save the ids of those who eventually develop sepsis
    if_sepsis = labels.groupby('pid').apply(lambda x: x.sum()) > 1
    ids = if_sepsis[if_sepsis].index
    save_pickle(ids, DATA_DIR + '/processed/labels/ids_eventual.pickle')

    idx = df[df['SOFA_deterioration'] == 1].index
    labels.loc[idx].sum() / labels.loc[idx].shape[0]
    idx = df[df['SOFA_deterioration'] != 1].index
    labels.loc[idx].sum() / labels.loc[idx].shape[0]
