"""

"""

import pandas as pd
from dataclasses import dataclass, field
from functools import cached_property
from scipy.stats import fisher_exact

@dataclass
class FairnessClassEval : 
    """
    Binary classification metrics for a single subgroup.

    Parameters
    ----------
    class_name : str
        Name of the subgroup being evaluated.
    y_true : pd.Series
        Ground-truth boolean labels indicating poverty status.
    y_pred : pd.Series
        Model-predicted boolean labels indicating poverty status.
    """
    class_name : str 
    class_df : pd.DataFrame 
    y_true : pd.Series 
    y_pred : pd.Series 

    def __post_init__(self) : 
        self.n = len(self.y_pred)

        # Confusion matrix 
        self.TP, self.FP, self.TN, self.FN = self._compute_confusion()

    def _compute_confusion(self) -> tuple[int]:
        """Compute TP/FP/TN/FN counts for the subgroup."""
        TP = ((self.y_pred == True) & (self.y_true == True)).sum()
        FP = ((self.y_pred == True) & (self.y_true != True)).sum()
        TN = ((self.y_pred != True) & (self.y_true != True)).sum()
        FN = ((self.y_pred != True) & (self.y_true == True)).sum()
        return int(TP), int(FP), int(TN), int(FN)
    
    def metrics(self) -> dict[str, float]:
        """Return core binary metrics and counts as a dict."""
        recall = self.TP / (self.TP + self.FN) if (self.TP + self.FN) else None  
        FNR = self.FN / (self.TP + self.FN) if (self.TP + self.FN) else None
        precision = self.TP / (self.TP + self.FP) if (self.TP + self.FP) else None 
        FPR = self.FP / (self.FP + self.TN) if (self.FP + self.TN) else None
        ACC = (self.TP + self.TN) / (self.TP + self.FP + self.TN + self.FN) 

        return {
            "Class" : self.class_name,
            "N" : self.n, 
            "TP": self.TP,
            "FP": self.FP,
            "TN": self.TN,
            "FN": self.FN,
            "Recall": recall,
            "Precision": precision,
            "FNR": FNR,
            "FPR": FPR,
            "Accuracy": ACC
        }

@dataclass
class FairnessGroupEval : 
    """
    Evaluate fairness metrics across demographic groups and visualize bias.

    Parameters
    ----------
    df : pd.DataFrame
        Data with income columns (`Actual_KES`, `Predicted_KES`),
        poverty classifications (`Actually_Poor`, `Predicted_Poor`),
        and grouping attributes.
    group_col : str
        Column defining the group attribute to audit.
    threshold : float or None, optional
        If None (default), uses the upstream `Actually_Poor` and `Predicted_Poor`
        columns from training.py. If a float (e.g. 0.25), computes poverty labels
        from that quantile of actual consumption, with separate urban/rural lines.
    group_name : str, optional
        Human-readable name for the grouping (defaults to `group_col`).
    reference_group : str, optional
        Comparison group; 'Rest' compares against all other groups.
    min_group_size : int, optional
        Minimum rows required for a group to be included.
    """
    df: pd.DataFrame
    group_col: str
    threshold: float | None = None
    group_name: str | None = None
    reference_group: str = "Rest"
    min_group_size: int = 30

    def __post_init__(self) -> None:
        self.group_name = self.group_name or self.group_col

        if self.threshold is not None:
            self.true_col = self._set_labels()
            self.pred_col = self._set_preds()
        else:
            self.true_col = 'Actually_Poor'
            self.pred_col = 'Predicted_Poor'

        self.evaluators = self._set_classes()
        self.df['Residual_KES'] = self.df['Predicted_KES'] - self.df['Actual_KES']

    @cached_property
    def threshold_rural(self) -> float:
        """Compute and cache the rural poverty threshold based on quantile."""
        return (
            self.df[self.df["urban_rural_classification"] == "Rural"]["Actual_KES"]
            .quantile(self.threshold)
        )

    @cached_property
    def threshold_urban(self) -> float:
        """Compute and cache the urban poverty threshold based on quantile."""
        return (
            self.df[self.df["urban_rural_classification"] == "Urban"]["Actual_KES"]
            .quantile(self.threshold)
        )

    def _set_labels(self) -> str:
        """Create boolean ground-truth poverty label per urban/rural threshold."""
        col_name = f'Actual_Poor_{str(self.threshold).replace("0.", "")}'

        self.df[col_name] = self.df.apply(
            lambda row: (
                row['Actual_KES'] <= self.threshold_rural
                if row['urban_rural_classification'] == 'Rural'
                else row['Actual_KES'] <= self.threshold_urban
            ),
            axis=1
        )

        return col_name

    def _set_preds(self) -> str:
        """Create boolean predicted poverty label per urban/rural threshold."""
        col_name = f'Predicted_Poor_{str(self.threshold).replace("0.", "")}'

        self.df[col_name] = self.df.apply(
            lambda row: (
                row['Predicted_KES'] <= self.threshold_rural
                if row['urban_rural_classification'] == 'Rural'
                else row['Predicted_KES'] <= self.threshold_urban
            ),
            axis=1
        )

        return col_name
    
    def _set_classes(self) -> dict[str, FairnessClassEval] : 
        """Instantiate evaluators for each eligible subgroup."""
        evaluator_df = {}

        classes = self._get_classes()

        for c in classes : 
            # Class df 
            class_df = self.df[self.df[self.group_col] == c]

            # Create a fairness evaluation for the class 
            evaluator_df[c] = FairnessClassEval(
                c, 
                class_df, 
                class_df[self.true_col],
                class_df[self.pred_col]
            )
        
        return evaluator_df 

    def _get_classes(self) -> list[str] : 
        """Return groups meeting the minimum size requirement."""
        class_counts = self.df[self.group_col].value_counts().reset_index()
        classes = class_counts[class_counts['count'] >= self.min_group_size][self.group_col].to_list()

        return classes 
    
    def _get_ref_eval(self, group : str) -> FairnessClassEval : 
        """Build evaluator for the chosen reference set (Rest or named group)."""
        
        # Determine reference group 
        if self.reference_group == 'Rest' : 
            ref_df = self.df[self.df[self.group_col] != group]
        
        else : 
            ref_df = self.df[self.df[self.group_col] == self.reference_group]

        # Compute reference metrics 
        ref_eval = FairnessClassEval(
            self.reference_group, 
            ref_df, 
            ref_df[self.true_col], 
            ref_df[self.pred_col]
        )

        return ref_eval
    
    def _compute_significance(self, group_eval: FairnessClassEval, ref_eval: FairnessClassEval) -> dict:
        """
        Compute Fisher's exact test p-values for FNR and FPR differences.
        """

        def p_to_sig(p):
            if p is None:
                return ""
            if p < 0.01:
                return "***"
            elif p < 0.05:
                return "**"
            elif p < 0.1:
                return "*"
            else:
                return "insig"

        fnr_table = [
            [group_eval.FN, group_eval.TP],  
            [ref_eval.FN, ref_eval.TP]       
        ]
        try:
            _, p_fnr = fisher_exact(fnr_table)
        except ValueError:
            p_fnr = None

        # --- FPR table (Inclusion Error) ---
        fpr_table = [
            [group_eval.FP, group_eval.TN],
            [ref_eval.FP, ref_eval.TN]
        ]
        try:
            _, p_fpr = fisher_exact(fpr_table)
        except ValueError:
            p_fpr = None

        return {"FNR_p": p_to_sig(p_fnr), "FPR_p": p_to_sig(p_fpr)}
    
    def get_metrics_df(self, percent : bool = True) -> pd.DataFrame:
        """Aggregate metrics, ratios, and counts for every subgroup."""
        results = []

        for group, group_eval in self.evaluators.items() : 
            # Get ref eval 
            ref_eval = self._get_ref_eval(group)
    
            # Get metrics 
            g_met = group_eval.metrics()
            r_met = ref_eval.metrics()

            # Calculate significance
            sig = self._compute_significance(group_eval, ref_eval)

            # FNR Ratio 
            if (g_met['FNR'] != None) and (r_met['FNR'] != None) and (r_met['FNR'] > 0) : 
                fnr_ratio = g_met['FNR'] / r_met['FNR'] if r_met['FNR'] > 0 else 0
            else : 
                fnr_ratio = None 

            if percent : 
                g_met['FNR'] = round(g_met['FNR'] * 100, 2)
                g_met['Recall'] = round(g_met['Recall'] * 100, 2)
                g_met['FPR'] = round(g_met['FPR'] * 100, 2)
                g_met['Precision'] = round(g_met['Precision'] * 100, 2) if g_met['Precision'] != None else None 

                r_met['FNR'] = round(r_met['FNR'] * 100, 2)
                r_met['Recall'] = round(r_met['Recall'] * 100, 2)
                r_met['FPR'] = round(r_met['FPR'] * 100, 2)
                r_met['Precision'] = round(r_met['Precision'] * 100, 2) if r_met['Precision'] != None else None 

            row = {
                'Group' : self.group_name,
                'Attribute': group,
                'Ref_Group': self.reference_group,
                
                # Confusion Matrix 
                'TP' : g_met['TP'], 
                'FP' : g_met['FP'], 
                'TN' : g_met['TN'],
                'FN' : g_met['FN'],
                'N': g_met['N'],

                # FNR (Exclusion Error Rate)
                'Group Exclusion Error (FNR)': g_met['FNR'],
                'Ref Exclusion Error (FNR)': r_met['FNR'],
                'Exclusion Ratio': round(fnr_ratio, 2),
                'Exclusion Sig' : sig['FNR_p'],

                # FPR (Inclusion Error Rate)
                'Group Inclusion Error (FPR)': g_met['FPR'],
                'Ref Inclusion Error (FPR)': r_met['FPR'],

                # True poverty share vs predicted poverty share 
                'Actual Poverty Share' : round(100 * ((g_met['TP'] + g_met['FN']) / g_met['N']), 2), 
                'Predicted Poverty Share' : round(100 * ((g_met['TP'] + g_met['FP']) / g_met['N']), 2), 
                
                # Recall
                'Recall_Group': g_met['Recall'],
                'Recall_Ref': r_met['Recall'],
                
                # Precision
                'Precision_Group': g_met['Precision'],
                'Precision_Ref': r_met['Precision']
            }
            results.append(row)
        
        return pd.DataFrame(results)
    