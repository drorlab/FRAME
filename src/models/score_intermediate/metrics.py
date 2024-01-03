import wandb
import numpy as np
import sklearn.metrics as sm
from sklearn.calibration import calibration_curve


def compute_metrics(task, targets, preds, current_epoch=0, prefix='', suffix=''):
    if task == 'regression':
        return regression_metrics(targets, preds, prefix, suffix)
    elif task == 'binary':
        return binary_metrics(targets, preds, current_epoch, prefix, suffix)
    else:
        raise ValueError(f"Invalid task {task:}")


def binary_metrics(targets, probs, current_epoch, prefix='', suffix=''):
    # probs = pos_prob and targets = y_true
    if len(targets) < 1:
        return {}
    metrics = {}
    ### Scalar metrics: AUROC etc2
    metrics[f'{prefix}auroc{suffix}'] = sm.roc_auc_score(targets, probs)
    metrics[f'{prefix}auprc{suffix}'] = sm.average_precision_score(targets, probs)
    metrics[f'{prefix}acc{suffix}'] = sm.accuracy_score(targets, probs.round())
    ### Plot metrics
    y_prob = np.vstack([1-probs, probs]).T # [neg_prob, pos_prob]
    # Confusion matrix
    metrics[f'{prefix}conf_mat{suffix}'] = wandb.plot.confusion_matrix(
        probs=y_prob, y_true=targets, class_names=['neg', 'pos'],
        title=f'{prefix} epoch {current_epoch}')
    # Precision-recall and ROC curves
    metrics[f'{prefix}pr_curve{suffix}'] = wandb.plot.pr_curve(
        targets, y_prob, labels=['neg', 'pos'],
        title=f'{prefix} epoch {current_epoch}: Precision v. Recall')
    metrics[f'{prefix}roc{suffix}'] = wandb.plot.roc_curve(
        targets, y_prob, labels=['neg', 'pos'],
        title=f'{prefix} epoch {current_epoch}: ROC')
    # Reliability diagram
    prob_true, prob_pred = calibration_curve(targets, probs, n_bins=10, normalize=False)
    metrics[f'{prefix}calibration_curve{suffix}'] = wandb.plot.line_series(
        xs=[[0.0, 1.0], prob_pred],
        ys=[[0.0, 1.0], prob_true],
        keys=['Perfectly Calibrated', 'Empirical Probabilty'],
        title='Calibration Curve',
        xname='Predicted Probability')
    # table = wandb.Table(data=[[x] for x in probs], columns=["pred_prob"])
    # metrics[f'{prefix}predicted_prob_hist{suffix}'] = wandb.plot.histogram(
    #     table, "pred_prob", title="Predicted Probability")

    return metrics


def regression_metrics(targets, preds, prefix='', suffix=''):
    if len(targets) < 1:
        return {}
    metrics = {}
    targets = targets.astype(float)
    preds = preds.astype(float)
    mse = ((targets - preds)**2).mean()
    metrics[f'{prefix}mse{suffix}'] = mse
    metrics[f'{prefix}rmse{suffix}'] = mse**0.5
    # Skip if only 2 data points or less since the correlations are
    # not really meaningful.
    if len(targets) > 2:
        metrics[f'{prefix}pearson{suffix}'] = targets.corr(preds, method='pearson')
        metrics[f'{prefix}spearman{suffix}'] = targets.corr(preds, method='spearman')
        metrics[f'{prefix}kendall{suffix}'] = targets.corr(preds, method='kendall')
    return metrics

