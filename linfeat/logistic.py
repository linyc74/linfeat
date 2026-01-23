import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Sequence, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from .basic import Parameters, config_matplotlib_font_for_language


import builtins
from functools import partial
print = partial(builtins.print, flush=True)  # always flush the output


class LogisticL1FeatureSelection:

    df: pd.DataFrame
    features: List[str]
    outcome: str
    param: Parameters

    X: np.ndarray
    y: np.ndarray

    C_grid: Sequence[float]
    best_C: float
    selected_features: List[str]

    def main(self, df: pd.DataFrame, features: List[str], outcome: str, parameters: Parameters):
        print(f'--- Logistic L1 Feature Selection ---\n')
        self.df = df
        self.features = features
        self.outcome = outcome
        self.param = parameters

        self.X = self.df[self.features].values
        self.y = self.df[self.outcome].values

        self.print_summary()
        self.estimate_C_by_cross_validation()
        self.plot_feature_paths_over_C()
        self.select_features_at_best_C()
        self.refit_with_selected_features()

    def print_summary(self):
        print(f'Samples: {len(self.df)}')
        print(f'Features: {len(self.features)}')
        print(f'Outcome: {self.outcome}\n')

    def estimate_C_by_cross_validation(self):
        self.C_grid = np.logspace(np.log10(self.param.l1_c_min), np.log10(self.param.l1_c_max), self.param.l1_c_grid_steps)
 
        k_fold = StratifiedKFold(n_splits=self.param.cv_folds, shuffle=True, random_state=self.param.random_state)

        e = np.full(shape=(self.param.l1_c_grid_steps,), fill_value=np.nan)
        mean_train_acc, std_train_acc = e.copy(), e.copy()
        mean_test_acc, std_test_acc = e.copy(), e.copy()
        mean_train_loss, std_train_loss = e.copy(), e.copy()
        mean_test_loss, std_test_loss = e.copy(), e.copy()

        for i, C in enumerate(self.C_grid):
            print(f'{i+1:>3d} / {self.param.l1_c_grid_steps}  C = {C:.5g}...')
            e = np.full(shape=(self.param.cv_folds,), fill_value=np.nan)  # empty array of length CV_FOLDS
            fold_test_acc, fold_train_acc, fold_test_loss, fold_train_loss = e.copy(), e.copy(), e.copy(), e.copy()

            for fold, (train_idx, test_idx) in enumerate(k_fold.split(self.X, self.y)):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]

                # scale INSIDE the fold
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)  # use training set's scaler for test set

                clf = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    C=float(C),
                    class_weight=self.param.class_weight,
                    max_iter=self.param.max_iter,
                    tol=self.param.tol,
                    random_state=self.param.random_state,
                )
                clf.fit(X_train_scaled, y_train)
                if clf.n_iter_ == self.param.max_iter:
                    print(f'C = {C:.5g}: The max_iter was reached which means the coef_ did not converge')

                y_hat_test = clf.predict(X_test_scaled)
                y_hat_train = clf.predict(X_train_scaled)
                fold_test_acc[fold] = accuracy_score(y_test, y_hat_test)
                fold_train_acc[fold] = accuracy_score(y_train, y_hat_train)
                fold_test_loss[fold] = log_loss(y_test, y_hat_test)
                fold_train_loss[fold] = log_loss(y_train, y_hat_train)

            mean_test_acc[i] = fold_test_acc.mean()
            std_test_acc[i] = fold_test_acc.std(ddof=1)  # ddof=1: delta degree of freedom = 1, that is n-1 for sample std
            mean_train_acc[i] = fold_train_acc.mean()
            std_train_acc[i] = fold_train_acc.std(ddof=1)
            mean_test_loss[i] = fold_test_loss.mean()
            std_test_loss[i] = fold_test_loss.std(ddof=1)
            mean_train_loss[i] = fold_train_loss.mean()
            std_train_loss[i] = fold_train_loss.std(ddof=1)

        best_idx = int(np.argmin(mean_test_loss))
        self.best_C = float(self.C_grid[best_idx])
        best_acc = mean_test_acc[best_idx]
        best_acc_std = std_test_acc[best_idx]
        best_loss = mean_test_loss[best_idx]
        best_loss_std = std_test_loss[best_idx]
        print(f'\nEstimated best C = {self.best_C:.5g}, mean accuracy = {best_acc:.3f} ± {best_acc_std:.3f}, mean log loss = {best_loss:.3f} ± {best_loss_std:.3f}\n')

        plt.figure(figsize=(self.param.fig_width, self.param.fig_height))
        plt.rcParams.update({'font.size': self.param.font_size})
        x = np.log10(self.C_grid)
        plt.errorbar(x, mean_train_acc, yerr=std_train_acc, fmt="o-", capsize=3, label='train', markersize=3, linewidth=1, color='dimgray')
        plt.errorbar(x, mean_test_acc, yerr=std_test_acc, fmt="o-", capsize=3, label='test', markersize=3, linewidth=1, color='orangered')
        plt.legend(fontsize=self.param.font_size)
        plt.xlabel('log10(C)  (larger = weaker regularization)')
        plt.ylabel('CV Accuracy (mean ± SD over folds)')
        plt.grid(True, color='lightgray', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'{self.param.outdir}/L1_accuracy.png', dpi=600)
        plt.close()

        plt.figure(figsize=(self.param.fig_width, self.param.fig_height))
        plt.rcParams.update({'font.size': self.param.font_size})
        x = np.log10(self.C_grid)
        plt.errorbar(x, mean_train_loss, yerr=std_train_loss, fmt="o-", capsize=3, label='train', markersize=3, linewidth=1, color='dimgray')
        plt.errorbar(x, mean_test_loss, yerr=std_test_loss, fmt="o-", capsize=3, label='test', markersize=3, linewidth=1, color='orangered')
        plt.legend(fontsize=self.param.font_size)
        plt.xlabel('log10(C)  (larger = weaker regularization)')
        plt.ylabel('CV Log Loss (mean ± SD over folds)')
        plt.grid(True, color='lightgray', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'{self.param.outdir}/L1_log_loss.png', dpi=600)
        plt.close()

    def plot_feature_paths_over_C(self):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(self.X)  # scale the whole dataset without train-test split

        clf = LogisticRegression(
            penalty='l1',
            solver='saga',
            fit_intercept=True,
            class_weight=self.param.class_weight,
            max_iter=self.param.max_iter,
            tol=self.param.tol,
            random_state=self.param.random_state,
            warm_start=True
        )

        # fit the model for each C in the grid
        coefs: List[np.ndarray] = []
        for C in self.C_grid:
            clf.C = float(C)  # set C for each iteration
            clf.fit(Xs, self.y)
            coefs.append(clf.coef_.ravel().copy())
        coefs = np.vstack(coefs)  # shape: (len(self.C_grid), n_features)

        # select top-m features
        abs_max = np.max(np.abs(coefs), axis=0)
        idx_sorted = np.argsort(abs_max)[::-1]
        idx_keep = idx_sorted[:min(self.param.l1_show_top_m_features, coefs.shape[1])]
        coefs_keep = coefs[:, idx_keep]  # shape: (len(self.C_grid), self.param.l1_show_top_m_features)
        names_keep = [self.features[j] for j in idx_keep]
        print(f'Plotting {self.param.l1_show_top_m_features} feature paths: {names_keep}\n')

        config_matplotlib_font_for_language(names_keep)

        # plot
        plt.figure(figsize=(self.param.fig_width, self.param.fig_height))
        plt.rcParams.update({'font.size': self.param.font_size})

        x = np.log10(self.C_grid)
        for j, name in enumerate(names_keep):  # plot each feature's path
            plt.plot(x, coefs_keep[:, j], label=name, linewidth=1)
        plt.axhline(0, linestyle='--', linewidth=1, color='black')

        if self.best_C is not None:  # if best C is set, mark it on the plot
            plt.axvline(np.log10(self.best_C), linestyle='--', linewidth=1, color='black')

        plt.xlabel('log10(C)  (larger = weaker regularization)')
        plt.ylabel('Coefficient Value')
        plt.grid(True, color='lightgray', linewidth=0.5)

        plt.tight_layout()  # tight layout before saving the figure

        plt.savefig(f'{self.param.outdir}/L1_feature_paths_without_legend.png', dpi=600)
        plt.legend(loc='best', fontsize=self.param.font_size, ncol=1)
        plt.savefig(f'{self.param.outdir}/L1_feature_paths_with_legend.png', dpi=600)        

        plt.close()

    def select_features_at_best_C(self):
        # note that this is not the same as the feature paths plot, 
        # as the best C could be set independently, so it's not among the values of the C_grid
        # so we need to fit the model at exactly the best C
        scaler = StandardScaler()
        Xs = scaler.fit_transform(self.X)  # scale the whole dataset without train-test split

        clf = LogisticRegression(
            penalty='l1',
            solver='saga',
            fit_intercept=True,
            class_weight=self.param.class_weight,
            max_iter=self.param.max_iter,
            tol=self.param.tol,
            random_state=self.param.random_state,
            warm_start=True
        )

        clf.C = self.best_C
        clf.fit(Xs, self.y)
        coefs = clf.coef_.ravel().copy()  # shape: (n_features,)

        # select features with absolute coefficient value > 0
        idx = np.where(np.abs(coefs) > 0)[0]
        self.selected_features = [self.features[j] for j in idx]
        print(f'{len(self.selected_features)} features selected at best C = {self.best_C:.5g}: {self.selected_features}\n')

    def refit_with_selected_features(self):
        # use the selected features
        # no scaling, because it's for interpreting the features
        idx = [self.features.index(feature) for feature in self.selected_features]
        X = self.X[:, idx]

        clf = LogisticRegression(
            penalty=None,  # no regularization, because it's not for prediction, just for interpretability
            solver='lbfgs',
            fit_intercept=True,
            class_weight=self.param.class_weight,
            max_iter=self.param.max_iter,
            tol=self.param.tol,
            random_state=self.param.random_state,
        )
        clf.fit(X, self.y)
        print(f'Refitted with selected features: {self.selected_features}')
        print(f'Intercept: {clf.intercept_.ravel()[0]:.3f}')
        print('Coefficients:')
        for name, coef in zip(self.selected_features, clf.coef_.ravel()):
            print(f"  '{name}': {coef:.3f}")
        print()


class LogisticStepwiseFeatureSelection:

    df: pd.DataFrame
    core_features: List[str]
    candidate_features: List[str]
    outcome: str
    param: Parameters

    y: np.ndarray
    k_fold: StratifiedKFold
    selected_features: List[str]
    results: List[Dict[str, float]]

    def main(
            self,
            df: pd.DataFrame,
            core_features: List[str],
            candidate_features: List[str],
            outcome: str,
            parameters: Parameters):
        print(f'--- Logistic Stepwise Feature Selection ---\n')
        self.df = df.copy()
        self.core_features = core_features
        self.candidate_features = candidate_features
        self.outcome = outcome
        self.param = parameters

        self.print_summary()

        # globally constant
        self.y = self.df[self.outcome].values
        self.k_fold = StratifiedKFold(n_splits=self.param.cv_folds, shuffle=True, random_state=self.param.random_state)
        
        # stepwise
        self.selected_features = []
        self.results = [None] * self.param.stepwise_n_features
        for ith_step in range(self.param.stepwise_n_features):
            print(f'Step {ith_step + 1} of {self.param.stepwise_n_features}...')
            self.select_one_feature(ith_step=ith_step)

            assert len(self.selected_features) == ith_step + 1
            assert self.results[ith_step] is not None
        
        self.plot_results()
        self.write_table_of_results()

    def print_summary(self):
        print(f'Samples: {len(self.df)}')
        print(f'{len(self.core_features)} core features: {self.core_features}')
        print(f'{len(self.candidate_features)} candidate features: {self.candidate_features}')
        print(f'Outcome: {self.outcome}\n')

    def select_one_feature(self, ith_step: int):
        best_feature = None  # of this step
        for candidate_feature in self.candidate_features:
            X = self.df[self.core_features + self.selected_features + [candidate_feature]].values
    
            cv_results = self.run_k_fold_cv(X=X, y=self.y, C=self.param.stepwise_c)
    
            if best_feature is None or cv_results['train_loss'] < self.results[ith_step]['train_loss']:
                best_feature = candidate_feature
                self.results[ith_step] = cv_results
        
        assert best_feature is not None  # now we should have found the best feature for this step
        self.selected_features.append(best_feature)
        self.candidate_features.remove(best_feature)
        print(f'Selected features: {self.selected_features}')
        print(f'Train loss: {self.results[ith_step]["train_loss"]}')
        print(f'Train accuracy: {self.results[ith_step]["train_acc"]}')
        print(f'Test loss: {self.results[ith_step]["test_loss"]}')
        print(f'Test accuracy: {self.results[ith_step]["test_acc"]}\n')
                
    def run_k_fold_cv(self, X: np.ndarray, y: np.ndarray, C: float) -> Dict[str, float]:
        e = np.full(shape=(self.param.cv_folds,), fill_value=np.nan)  # empty array of length CV_FOLDS
        train_acc, test_acc, train_loss, test_loss = e.copy(), e.copy(), e.copy(), e.copy()
        for fold, (train_idx, test_idx) in enumerate(self.k_fold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # scale INSIDE the fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)  # use training set's scaler for test set

            clf = LogisticRegression(
                penalty='l2',
                C=C,
                solver='saga',
                fit_intercept=True,
                class_weight=self.param.class_weight,
                max_iter=self.param.max_iter,
                tol=self.param.tol,
                random_state=self.param.random_state,
                warm_start=True
            )

            clf.fit(X_train_scaled, y_train)

            y_hat_test = clf.predict(X_test_scaled)
            y_hat_train = clf.predict(X_train_scaled)
            train_acc[fold] = accuracy_score(y_train, y_hat_train)
            test_acc[fold] = accuracy_score(y_test, y_hat_test)
            train_loss[fold] = log_loss(y_train, y_hat_train)
            test_loss[fold] = log_loss(y_test, y_hat_test)

        return {
            'train_acc': train_acc.mean(),
            'train_acc_std': train_acc.std(ddof=1),  # ddof=1: delta degree of freedom = 1, that is n-1 for sample std
            'test_acc': test_acc.mean(),
            'test_acc_std': test_acc.std(ddof=1),
            'train_loss': train_loss.mean(),
            'train_loss_std': train_loss.std(ddof=1),
            'test_loss': test_loss.mean(),
            'test_loss_std': test_loss.std(ddof=1),
        }

    def plot_results(self):
        train_acc = [result['train_acc'] for result in self.results]
        train_acc_std = [result['train_acc_std'] for result in self.results]
        test_acc = [result['test_acc'] for result in self.results]
        test_acc_std = [result['test_acc_std'] for result in self.results]
        train_loss = [result['train_loss'] for result in self.results]
        train_loss_std = [result['train_loss_std'] for result in self.results]
        test_loss = [result['test_loss'] for result in self.results]
        test_loss_std = [result['test_loss_std'] for result in self.results]

        config_matplotlib_font_for_language(self.selected_features)

        plt.figure(figsize=self.__figsize())
        plt.rcParams.update({'font.size': self.param.font_size})
        plt.errorbar(range(len(train_acc)), train_acc, yerr=train_acc_std, fmt="o-", capsize=3, label='train', markersize=3, linewidth=1, color='dimgray')
        plt.errorbar(range(len(test_acc)), test_acc, yerr=test_acc_std, fmt="o-", capsize=3, label='test', markersize=3, linewidth=1, color='orangered')
        plt.legend(fontsize=self.param.font_size)
        plt.xlabel('Features')
        plt.ylabel('CV Accuracy (mean ± SD over folds)')
        plt.xticks(range(len(self.selected_features)), self.selected_features)
        plt.xticks(rotation=45)
        plt.xlim(-0.5, len(self.selected_features) - 0.5)
        plt.grid(True, color='lightgray', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'{self.param.outdir}/stepwise_accuracy.png', dpi=600)
        plt.close()

        plt.figure(figsize=self.__figsize())
        plt.rcParams.update({'font.size': self.param.font_size})
        plt.errorbar(range(len(train_loss)), train_loss, yerr=train_loss_std, fmt="o-", capsize=3, label='train', markersize=3, linewidth=1, color='dimgray')
        plt.errorbar(range(len(test_loss)), test_loss, yerr=test_loss_std, fmt="o-", capsize=3, label='test', markersize=3, linewidth=1, color='orangered')
        plt.xticks(range(len(self.selected_features)), self.selected_features)
        plt.xticks(rotation=45)
        plt.xlim(-0.5, len(self.selected_features) - 0.5)
        plt.legend(fontsize=self.param.font_size)
        plt.xlabel('Features')
        plt.ylabel('CV Log Loss (mean ± SD over folds)')
        plt.grid(True, color='lightgray', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'{self.param.outdir}/stepwise_log_loss.png', dpi=600)
        plt.close()

    def __figsize(self) -> Tuple[float, float]:
        longest_feature_name = max(len(feature) for feature in self.selected_features)
        h_per_char_per_font_size = 0.1 / 7  # 0.1 cm per character at font size 7
        h = self.param.fig_height + (longest_feature_name * h_per_char_per_font_size * self.param.font_size / 2.54)
        return self.param.fig_width, h

    def write_table_of_results(self):
        df = pd.DataFrame(self.results)
        df.index = self.selected_features
        df.index.name = 'Feature'
        df = df.rename(columns={
            'train_acc': 'Train Accuracy',
            'train_acc_std': 'Train Accuracy SD',
            'test_acc': 'Test Accuracy',
            'test_acc_std': 'Test Accuracy SD',
            'train_loss': 'Train Log Loss',
            'train_loss_std': 'Train Log Loss SD',
            'test_loss': 'Test Log Loss',
            'test_loss_std': 'Test Log Loss Std',
        })
        df.to_csv(f'{self.param.outdir}/stepwise_results.csv')
