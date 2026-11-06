"""Develop, freeze, and evaluate the Bank Marketing coursework models."""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
from time import perf_counter
from functools import wraps

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'bank-marketing-matplotlib'))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from bank_data import BOUNDARY, TOTAL, sha, source, write_json

ROOT = Path(__file__).resolve().parents[1]
SEED, THRESHOLD, THREADS = 42, .5, 3
FIELDS = ('age','job','marital','education','default','housing','loan','previous','poutcome')
ECON = ('emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed')
CONTACT = ('contact','month','day_of_week','campaign')
NAMES = ('logistic_regression','decision_tree','knn')
MODELS = ('dummy_prior', *NAMES, 'soft_voting')
RESULTS = Path('results/model_experiments')
MODEL_DIR = Path('models/model_experiments')
CLEANED = Path('data/processed/bank_marketing_cleaned.csv')
DEVELOPMENT_FILES = ('folds.csv', 'preprocessing_folds.csv', 'preprocessing_summary.csv', 'baseline_folds.csv',
                     'tuning_folds.csv', 'tuned_and_ensemble_folds.csv', 'development_summary.csv')
HOLDOUT_FILES = ('holdout_metrics.csv', 'holdout_predictions.csv', 'holdout_block_sensitivity.csv',
                 'holdout_precision_recall.png')
OUTPUTS = (*[RESULTS / file for file in (*DEVELOPMENT_FILES, *HOLDOUT_FILES)], CLEANED)
PROFILES = {
    'historical_full': dict(conditional=True, interval=True, fill_unknown=False),
    'without_economic': dict(conditional=True, interval=True, fill_unknown=False),
    'precall_interval': dict(conditional=False, interval=True, fill_unknown=False),
    'precall_no_interval': dict(conditional=False, interval=False, fill_unknown=False),
    'precall_impute_unknown': dict(conditional=False, interval=True, fill_unknown=True),
}
BASE = {'logistic_regression': {'C':1.,'class_weight':'balanced'},
        'decision_tree': {'max_depth':None,'min_samples_leaf':1,'class_weight':None},
        'knn': {'n_neighbors':5,'weights':'uniform'}}
GRIDS = {'logistic_regression': {'C':[.01,.1,1.,10.],'class_weight':[None,'balanced']},
         'decision_tree': {'max_depth':[3,6,12,None],'min_samples_leaf':[20,100],'class_weight':[None,'balanced']},
         'knn': {'n_neighbors':[5,15,45,101],'weights':['uniform','distance']}}


def single_run(operation):
    """Serialize operations on a project. The OS releases the lock on exit."""
    @wraps(operation)
    def run(root, *args, **kwargs):
        with (root / '.bank-experiment.lock').open('a+b') as lock:
            try:
                if os.name == 'nt':
                    import msvcrt
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Another experiment is using this project folder. Retry when it finishes.') from exc
            return operation(root, *args, **kwargs)
    return run


def target(frame):
    y = frame.y.map({'no':0,'yes':1})
    if y.isna().any() or y.nunique()!=2:
        raise ValueError('Expected both valid binary labels.')
    return y.astype(int)


def features(frame, profile):
    spec = PROFILES[profile]
    fields = list(FIELDS)
    if spec['conditional']:
        fields += list(CONTACT)
    if profile == 'historical_full':
        fields += list(ECON)
    x = frame[fields].copy()
    if spec['interval']:
        x['pdays'] = frame.pdays.where(frame.pdays.ne(999),np.nan)
        x['pdays_recorded'] = x.pdays.notna().astype('int64')
    if spec['fill_unknown']:
        for col in x.select_dtypes(exclude='number'):
            x[col] = x[col].astype(object).replace('unknown',np.nan)
    assert not {'y','duration','source_row','partition'} & set(x)
    return x


def make_model(name, params, x):
    if name == 'dummy_prior':
        return DummyClassifier(strategy='prior')
    estimator = {'logistic_regression': lambda: LogisticRegression(max_iter=2000,random_state=SEED,**params),
                 'decision_tree': lambda: DecisionTreeClassifier(random_state=SEED,**params),
                 'knn': lambda: KNeighborsClassifier(algorithm='brute',metric='euclidean',n_jobs=1,**params)}[name]()
    num = x.select_dtypes(include='number').columns.tolist()
    cat = x.select_dtypes(exclude='number').columns.tolist()
    numeric = [('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
    if name in ('logistic_regression','knn'):
        numeric.append(('scale',StandardScaler()))
    pre = ColumnTransformer([('numeric',Pipeline(numeric),num),('categorical',Pipeline([
        ('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
        ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
    return Pipeline([('preprocess',pre),('model',estimator)])


def positive(model,x):
    return model.predict_proba(x)[:,list(model.classes_).index(1)]


def metrics(y,p):
    # Use the same threshold rule for all models; scores are not claimed calibrated.
    yhat = (p > THRESHOLD).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,yhat,labels=[0,1]).ravel()
    return dict(average_precision=average_precision_score(y,p),roc_auc=roc_auc_score(y,p),
                precision=precision_score(y,yhat,zero_division=0),recall=recall_score(y,yhat,zero_division=0),
                f1=f1_score(y,yhat,zero_division=0),balanced_accuracy=balanced_accuracy_score(y,yhat),
                brier=brier_score_loss(y,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def cv_run(factory,x,y,folds,train_score=False):
    rows=[]; preds=[]
    for number,(train,val) in enumerate(folds,1):
        assert x.index[train[-1]] < x.index[val[0]] and x.index[val[-1]] < BOUNDARY
        model=factory(); start=perf_counter()
        model.fit(x.iloc[train],y.iloc[train]); fit_time=perf_counter()-start
        start=perf_counter(); p=positive(model,x.iloc[val]); score_time=perf_counter()-start
        result=dict(fold=number,fit_seconds=fit_time,predict_seconds=score_time,**metrics(y.iloc[val],p))
        if train_score:
            result['train_average_precision']=average_precision_score(y.iloc[train],positive(model,x.iloc[train]))
        rows.append(result); preds.append(p)
    return rows,preds


def mean_ap(rows):
    return float(np.mean([r['average_precision'] for r in rows]))


def factory_for(name,params,x):
    return lambda: make_model(name,params,x)


@single_run
def develop(root):
    result=root/RESULTS; models=root/MODEL_DIR
    result.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True)
    if any((result / name).exists() for name in ('holdout_started.json', 'holdout_metrics.csv', 'holdout_outputs.json')):
        raise RuntimeError('This evaluation is already opened. Preserve it; use a separate research workspace for further development.')
    if (result / 'frozen_protocol.json').exists():
        frozen_protocol(root, require_holdout=False)
        print('Development is already frozen and verified. Run evaluate to score the holdout.', flush=True)
        return
    path,digest=source(root)
    raw=pd.read_csv(path,sep=';',nrows=BOUNDARY)
    assert raw.shape==(BOUNDARY,21)
    data=raw.drop_duplicates(keep='first').copy(); y=target(data)
    folds=list(TimeSeriesSplit(n_splits=3).split(data))
    assert len(data)==32940 and int(y.sum())==2100
    fold_rows=[dict(fold=i,train_rows=len(tr),validation_rows=len(va),train_last_source_row=int(data.index[tr[-1]]),
                    validation_first_source_row=int(data.index[va[0]]),validation_positive_rate=float(y.iloc[va].mean()))
               for i,(tr,va) in enumerate(folds,1)]
    pd.DataFrame(fold_rows).to_csv(result/'folds.csv',index=False)
    prep_rows=[]
    for profile in PROFILES:
        x=features(data,profile)
        rows,_=cv_run(factory_for('logistic_regression',BASE['logistic_regression'],x),x,y,folds)
        prep_rows.extend(dict(profile=profile,eligible=not PROFILES[profile]['conditional'],**r) for r in rows)
    prep=pd.DataFrame(prep_rows)
    prep.to_csv(result/'preprocessing_folds.csv',index=False)
    prep_summary=prep.groupby(['profile','eligible'],sort=False).average_precision.agg(['mean','std']).reset_index()
    prep_summary.to_csv(result/'preprocessing_summary.csv',index=False)
    chosen=str(prep_summary[prep_summary.eligible].sort_values('mean',ascending=False,kind='stable').iloc[0].profile)
    x=features(data,chosen)
    print(f'Frozen eligible preprocessing profile for model comparisons: {chosen}',flush=True)
    baseline_rows=[]; grid_rows=[]; tuned_rows=[]; best={}; scores={}; tuned_predictions={}
    for name in ('dummy_prior',*NAMES):
        rows,_=cv_run(factory_for(name,BASE.get(name,{}),x),x,y,folds)
        baseline_rows.extend(dict(model=name,**r) for r in rows)
        if name=='dummy_prior':
            continue
        start=perf_counter()
        candidates=list(ParameterGrid(GRIDS[name]))
        if BASE[name] not in candidates:
            candidates.insert(0,BASE[name])
        high=-1.
        for i,params in enumerate(candidates,1):
            # A baseline combination is reused when the grid contains it.
            trial=rows if params==BASE[name] else cv_run(factory_for(name,params,x),x,y,folds)[0]
            ap=mean_ap(trial)
            grid_rows.extend(dict(model=name,candidate=i,params=json.dumps(params,sort_keys=True),mean_average_precision=ap,**r) for r in trial)
            if ap>high+1e-12:
                high=ap; best[name]=params
        vals,tuned_predictions[name]=cv_run(factory_for(name,best[name],x),x,y,folds,train_score=True)
        scores[name]=mean_ap(vals)
        tuned_rows.extend(dict(model=name,**r) for r in vals)
        print(f'{name}: {len(candidates)} candidates in {perf_counter()-start:.1f}s; AP {mean_ap(rows):.4f} -> {scores[name]:.4f}; {best[name]}',flush=True)
    pd.DataFrame(baseline_rows).to_csv(result/'baseline_folds.csv',index=False)
    pd.DataFrame(grid_rows).to_csv(result/'tuning_folds.csv',index=False)
    ranking=sorted(NAMES,key=lambda name:-scores[name]); pair=ranking[:2]
    ensemble=lambda: VotingClassifier([(name,make_model(name,best[name],x)) for name in pair],voting='soft',n_jobs=1)
    rows,ensemble_predictions=cv_run(ensemble,x,y,folds,train_score=True)
    for p,a,b in zip(ensemble_predictions,*[tuned_predictions[n] for n in pair]):
        np.testing.assert_allclose(p,(a+b)/2,atol=1e-12,rtol=0)
    tuned_rows.extend(dict(model='soft_voting',**r) for r in rows)
    pd.DataFrame(tuned_rows).to_csv(result/'tuned_and_ensemble_folds.csv',index=False)
    summary=[]
    for name in MODELS:
        b=[r for r in baseline_rows if r['model']==name]
        t=[r for r in tuned_rows if r['model']==name] or b
        ap=mean_ap(t)
        summary.append(dict(model=name,baseline_ap=mean_ap(b) if b else np.nan,development_ap=ap,
                            fold_ap_sd=np.std([r['average_precision'] for r in t],ddof=1),
                            gain_vs_baseline=ap-mean_ap(b) if b else np.nan,
                            mean_train_ap=np.mean([r['train_average_precision'] for r in t]) if 'train_average_precision' in t[0] else np.nan,
                            selected_params=json.dumps(best.get(name,{}),sort_keys=True)))
    pd.DataFrame(summary).to_csv(result/'development_summary.csv',index=False)
    model_manifest={}
    for name in MODELS:
        model=ensemble() if name=='soft_voting' else make_model(name,best.get(name,{}),x)
        start=perf_counter(); model.fit(x,y); fit_time=perf_counter()-start
        f=models/f'{name}.joblib'; joblib.dump(model,f,compress=3)
        np.testing.assert_allclose(positive(model,x.iloc[:32]),positive(joblib.load(f),x.iloc[:32]))
        model_manifest[name]={'file':f.name,'sha256':sha(f),'bytes':f.stat().st_size,'development_refit_seconds':fit_time}
    spec=dict(raw_sha256=digest,source_rows=TOTAL,holdout_boundary=BOUNDARY,development_rows=len(data),
              profile=chosen,features=list(x.columns),candidates=best,ranking=ranking,ensemble_members=pair,
              excluded_model=ranking[-1],ensemble_weights=[1,1],threshold=THRESHOLD,
              threshold_rule='positive when score strictly greater than threshold',
              selection_metric='unweighted mean fold average precision',
              development_files={name: sha(result / name) for name in DEVELOPMENT_FILES},
              model_manifest=model_manifest,seed=SEED,threads=THREADS,python=platform.python_version(),
              sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__)
    write_json(result/'frozen_protocol.json',spec)
    print(f'Protocol frozen before reading holdout outcomes: {pair}; omitted {ranking[-1]}',flush=True)


def frozen_protocol(root, *, require_holdout):
    result = root / RESULTS
    protocol = result / 'frozen_protocol.json'
    spec = json.loads(protocol.read_text(encoding='utf-8'))
    protocol_hash = sha(protocol)
    marker = result / 'holdout_started.json'
    if marker.exists():
        expected = {'protocol_sha256': protocol_hash, 'first_test_row': BOUNDARY, 'test_rows': TOTAL - BOUNDARY}
        saved_marker = json.loads(marker.read_text(encoding='utf-8'))
        if saved_marker not in (expected, {**expected, 'output_receipt_required': True}):
            raise ValueError('The frozen protocol or holdout marker changed after the test was opened.')
    elif require_holdout or any((result / name).exists() for name in ('holdout_metrics.csv', 'holdout_outputs.json')):
        raise FileNotFoundError('The holdout marker is missing. Restore it from the saved run before using these results.')
    path, digest = source(root)
    if (digest, BOUNDARY, THRESHOLD, TOTAL) != (spec['raw_sha256'], spec['holdout_boundary'], spec['threshold'], spec['source_rows']):
        raise ValueError('The frozen protocol does not match the source data or evaluation settings.')
    if spec['sklearn'] != sklearn.__version__:
        raise RuntimeError(f"The frozen models require scikit-learn {spec['sklearn']}. Use the pinned environment.")
    if set(spec['model_manifest']) != set(MODELS):
        raise ValueError('The frozen protocol must describe all five models.')
    for name, detail in spec['model_manifest'].items():
        filename = f'{name}.joblib'
        if detail['file'] != filename or sha(root / MODEL_DIR / filename) != detail['sha256']:
            raise ValueError(f'Model file differs from the frozen protocol: {filename}.')
    hashes = spec.get('development_files')
    if hashes is not None and any(hashes.get(name) != sha(result / name) for name in DEVELOPMENT_FILES):
        raise ValueError('Development evidence differs from the frozen protocol.')
    return path, spec, protocol_hash


def record_holdout_outputs(root, protocol_hash):
    files = {path.as_posix(): sha(root / path) for path in OUTPUTS}
    write_json(root / RESULTS / 'holdout_outputs.json', {'protocol_sha256': protocol_hash, 'files': files})
    marker = root / RESULTS / 'holdout_started.json'
    state = json.loads(marker.read_text(encoding='utf-8'))
    if not state.get('output_receipt_required'):
        write_json(marker, {**state, 'output_receipt_required': True})


def read_verified_holdout(root):
    path, spec, protocol_hash = frozen_protocol(root, require_holdout=True)
    result = root / RESULTS
    receipt = result / 'holdout_outputs.json'
    if receipt.exists():
        expected = json.loads(receipt.read_text(encoding='utf-8'))
        if expected['protocol_sha256'] != protocol_hash or set(expected['files']) != {p.as_posix() for p in OUTPUTS}:
            raise ValueError('The saved results receipt does not match the frozen protocol or output inventory.')
        for output in OUTPUTS:
            if sha(root / output) != expected['files'][output.as_posix()]:
                raise ValueError(f'Saved experiment file changed: {output.as_posix()}.')
    elif json.loads((result / 'holdout_started.json').read_text(encoding='utf-8')).get('output_receipt_required'):
        raise FileNotFoundError('The saved results receipt is missing. Run evaluate to recover the holdout outputs.')
    predictions = pd.read_csv(result / 'holdout_predictions.csv', float_precision='round_trip')
    if not receipt.exists():
        # Older completed runs did not save an output receipt. Compare their scores with the frozen estimators.
        test = pd.read_csv(path, sep=';').iloc[BOUNDARY:]
        x = features(test, spec['profile'])
        y = target(test)
        np.testing.assert_array_equal(predictions.source_row, test.index)
        np.testing.assert_array_equal(predictions.y, y)
        table = pd.read_csv(result / 'holdout_metrics.csv').set_index('model')
        for name in MODELS:
            score = positive(joblib.load(root / MODEL_DIR / f'{name}.joblib'), x)
            np.testing.assert_allclose(predictions[name], score, atol=1e-12, rtol=0)
            for metric, value in metrics(y, score).items():
                np.testing.assert_allclose(table.loc[name, metric], value, atol=1e-12, rtol=0)
    return spec, predictions


@single_run
def verify(root):
    read_verified_holdout(root)
    print('Frozen model files and saved holdout results verified.', flush=True)


@single_run
def evaluate(root):
    result = root / RESULTS
    path, spec, protocol_hash = frozen_protocol(root, require_holdout=False)
    receipt = result / 'holdout_outputs.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if saved['protocol_sha256'] != protocol_hash or set(saved['files']) != {p.as_posix() for p in OUTPUTS}:
            raise ValueError('The saved results receipt does not match the frozen protocol or output inventory.')
        for name in DEVELOPMENT_FILES:
            file = RESULTS / name
            if sha(root / file) != saved['files'].get(file.as_posix()):
                raise ValueError(f'Development evidence changed: {file.as_posix()}.')
        if all((root / file).is_file() and sha(root / file) == saved['files'][file.as_posix()]
               for file in (*[RESULTS / name for name in HOLDOUT_FILES], CLEANED)):
            print('Saved holdout evaluation verified and reused.', flush=True)
            return
    started = result / 'holdout_started.json'
    if not started.exists():
        write_json(started, {'protocol_sha256': protocol_hash, 'first_test_row': BOUNDARY, 'test_rows': TOTAL - BOUNDARY})
    with tempfile.TemporaryDirectory(prefix='.holdout-', dir=result) as directory:
        staging = Path(directory)
        score_holdout(root, path, spec, staging)
        (root / CLEANED.parent).mkdir(parents=True, exist_ok=True)
        for output in (*[RESULTS / name for name in HOLDOUT_FILES], CLEANED):
            (staging / output.name).replace(root / output)
        record_holdout_outputs(root, protocol_hash)


def score_holdout(root, path, spec, staging):
    models = root / MODEL_DIR
    raw=pd.read_csv(path,sep=';'); assert raw.shape==(TOTAL,21)
    dev=raw.iloc[:BOUNDARY].drop_duplicates(keep='first'); test=raw.iloc[BOUNDARY:]
    x=features(test,spec['profile']); y=target(test)
    assert list(x.columns)==spec['features'] and len(test)==8238
    prediction=pd.DataFrame({'source_row':test.index,'y':y.to_numpy()}); rows=[]; saved={}
    for name,detail in spec['model_manifest'].items():
        model=joblib.load(models/detail['file']); start=perf_counter(); p=positive(model,x)
        elapsed=perf_counter()-start; saved[name]=p
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        rows.append(dict(model=name,**metrics(y,p),test_rows=len(y),positives=int(y.sum()),
                         predict_seconds=elapsed,fit_seconds=detail['development_refit_seconds'],model_bytes=detail['bytes']))
        prediction[name]=p
    a,b=spec['ensemble_members']; np.testing.assert_allclose(saved['soft_voting'],(saved[a]+saved[b])/2,atol=1e-12,rtol=0)
    assert abs(rows[0]['average_precision']-y.mean())<1e-12
    assert all(sum(r[k] for k in ('tn','fp','fn','tp'))==len(y) for r in rows)
    pd.DataFrame(rows).to_csv(staging/'holdout_metrics.csv',index=False)
    prediction.to_csv(staging/'holdout_predictions.csv',index=False)
    groups=[]
    for index,part in enumerate(np.array_split(np.arange(len(y)),3),1):
        for name,p in saved.items():
            groups.append(dict(block=index,rows=len(part),positive_rate=float(y.iloc[part].mean()),model=name,**metrics(y.iloc[part],p[part])))
    pd.DataFrame(groups).to_csv(staging/'holdout_block_sensitivity.csv',index=False)
    dataset=[]
    for partition,frame in (('development',dev),('holdout',test)):
        clean=features(frame,spec['profile']).copy(); clean.insert(0,'source_row',frame.index); clean.insert(1,'partition',partition)
        clean['y']=target(frame); dataset.append(clean)
    cleaned=pd.concat(dataset,ignore_index=True)
    assert cleaned.source_row.is_unique and len(cleaned)==41178
    cleaned.to_csv(staging/CLEANED.name,index=False)
    fig,ax=plt.subplots(figsize=(7.5,4.7))
    for row in rows[1:]:
        name=row['model']; precision,recall,_=precision_recall_curve(y,saved[name])
        ax.plot(recall,precision,label=f"{name.replace('_',' ')} (AP {row['average_precision']:.3f})")
    ax.axhline(y.mean(),color='0.35',linestyle='--',label=f'Positive rate {y.mean():.3f}')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Recall',ylabel='Precision',title='Reserved holdout precision and recall')
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(staging/'holdout_precision_recall.png',dpi=180); plt.close(fig)
    print(pd.DataFrame(rows)[['model','average_precision','roc_auc','f1','balanced_accuracy']].to_string(index=False),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=['develop','evaluate','verify'])
    cli.add_argument('--root',type=Path,default=ROOT)
    args=cli.parse_args()
    with threadpool_limits(limits=THREADS):
        {'develop':develop,'evaluate':evaluate,'verify':verify}[args.phase](args.root.resolve())
