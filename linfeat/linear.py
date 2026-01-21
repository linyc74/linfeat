import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Sequence, Dict, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from .basic import Parameters


import builtins
from functools import partial
print = partial(builtins.print, flush=True)  # always flush the output


class LinearL1FeatureSelection:

    df: pd.DataFrame
    features: List[str]
    outcome: str
    param: Parameters

    X: np.ndarray
    y: np.ndarray

    alpha_grid: Sequence[float]
    best_alpha: float
    selected_features: List[str]

    def main(self, df: pd.DataFrame, features: List[str], outcome: str, parameters: Parameters):
        print(f'--- Linear L1 Feature Selection ---\n')
        self.df = df
        self.features = features
        self.outcome = outcome
        self.param = parameters

        self.X = self.df[self.features].values
        self.y = self.df[self.outcome].values

        self.print_summary()
        self.estimate_alpha_by_cross_validation()
        self.plot_feature_paths_over_alpha()
        self.manually_set_best_alpha()
        self.select_features_at_best_alpha()
        if len(self.selected_features) > 0:
            self.refit_with_selected_features()
        else:
            print('No features were selected. Abort refitting.')

    def print_summary(self):
        print(f'Samples: {len(self.df)}')
        print(f'Features: {len(self.features)}')
        print(f'Outcome: {self.outcome}\n')

    def estimate_alpha_by_cross_validation(self):
        self.alpha_grid = np.logspace(np.log10(self.param.l1_alpha_min), np.log10(self.param.l1_alpha_max), self.param.l1_alpha_grid_steps)

        k_fold = KFold(n_splits=self.param.cv_folds, shuffle=True, random_state=self.param.random_state)

        e = np.full(shape=(self.param.l1_alpha_grid_steps,), fill_value=np.nan)  # empty array of length l1_alpha_grid_steps
        mean_train_mse, std_train_mse = e.copy(), e.copy()
        mean_test_mse, std_test_mse = e.copy(), e.copy()

        for i, alpha in enumerate(self.alpha_grid):
            print(f'{i+1:>3d} / {self.param.l1_alpha_grid_steps}  alpha = {alpha:.5g}...')

            e = np.full(shape=(self.param.cv_folds,), fill_value=np.nan)  # empty array of length cv_folds
            fold_test_mse, fold_train_mse = e.copy(), e.copy()

            for fold, (train_idx, test_idx) in enumerate(k_fold.split(self.X, self.y)):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]

                # scale INSIDE the fold
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)  # use training set's scaler for test set

                lasso = Lasso(
                    alpha=alpha,
                    fit_intercept=True,
                    max_iter=self.param.max_iter,
                    tol=self.param.tol,
                    random_state=self.param.random_state,
                )
                lasso.fit(X_train_scaled, y_train)

                y_hat_test = lasso.predict(X_test_scaled)
                y_hat_train = lasso.predict(X_train_scaled)
                fold_test_mse[fold] = mean_squared_error(y_test, y_hat_test)
                fold_train_mse[fold] = mean_squared_error(y_train, y_hat_train)

            mean_test_mse[i] = fold_test_mse.mean()
            std_test_mse[i] = fold_test_mse.std(ddof=1)  # ddof=1: delta degree of freedom = 1, that is n-1 for sample std
            mean_train_mse[i] = fold_train_mse.mean()
            std_train_mse[i] = fold_train_mse.std(ddof=1)

        best_idx = int(np.argmin(mean_test_mse))
        self.best_alpha = float(self.alpha_grid[best_idx])
        best_mse = mean_test_mse[best_idx]
        best_mse_std = std_test_mse[best_idx]
        print(f'\nEstimated best alpha = {self.best_alpha:.5g}, MSE = {best_mse:.3f} ± {best_mse_std:.3f}\n')

        plt.figure(figsize=(self.param.fig_width, self.param.fig_height))
        plt.rcParams.update({'font.size': self.param.font_size})
        C_grid = 1 / self.alpha_grid
        x = np.log10(C_grid)
        plt.errorbar(x, mean_train_mse, yerr=std_train_mse, fmt="o-", capsize=3, label='train', markersize=3, linewidth=1, color='dimgray')
        plt.errorbar(x, mean_test_mse, yerr=std_test_mse, fmt="o-", capsize=3, label='test', markersize=3, linewidth=1, color='orangered')
        plt.legend()
        plt.xlabel('log10(1/alpha)  (larger = weaker regularization)')
        plt.ylabel('CV Mean Squared Error (mean ± SD over folds)')
        plt.grid(True, color='lightgray', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'{self.param.outdir}/L1_mean_squared_error.png', dpi=600)
        plt.close()

    def plot_feature_paths_over_alpha(self):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(self.X)  # scale the whole dataset without train-test split

        coefs: List[np.ndarray] = []
        for alpha in self.alpha_grid:
            lasso = Lasso(
                alpha=alpha,
                fit_intercept=True,
                max_iter=self.param.max_iter,
                tol=self.param.tol,
                random_state=self.param.random_state)
            lasso.fit(Xs, self.y)
            coefs.append(lasso.coef_.ravel().copy())
        coefs = np.vstack(coefs)  # shape: (len(self.alpha_grid), n_features)

        # select top-m features
        abs_max = np.max(np.abs(coefs), axis=0)
        idx_sorted = np.argsort(abs_max)[::-1]
        idx_keep = idx_sorted[:min(self.param.l1_show_top_m_features, coefs.shape[1])]
        coefs_keep = coefs[:, idx_keep]  # shape: (len(self.alpha_grid), self.l1_show_top_m_features)
        names_keep = [self.features[j] for j in idx_keep]

        # plot
        plt.figure(figsize=(self.param.fig_width, self.param.fig_height))
        plt.rcParams.update({'font.size': self.param.font_size})

        C = 1 / self.alpha_grid
        x = np.log10(C)
        for j, name in enumerate(names_keep):  # plot each feature's path
            plt.plot(x, coefs_keep[:, j], label=name, linewidth=1)
        plt.axhline(0, linestyle='--', linewidth=1, color='black')

        if self.best_alpha is not None:  # if best alpha is set, mark it on the plot
            plt.axvline(np.log10(1/self.best_alpha), linestyle='--', linewidth=1, color='black')
            # plt.text(np.log10(1/self.best_alpha), plt.ylim()[1]*0.9, 'Best alpha', ha='center', va='top', fontsize=self.FONT_SIZE, color='black')

        plt.xlabel('log10(1/alpha)  (larger = weaker regularization)')
        plt.ylabel('Coefficient Value')
        plt.grid(True, color='lightgray', linewidth=0.5)

        plt.tight_layout()  # tight layout before saving the figure

        plt.savefig(f'{self.param.outdir}/L1_feature_paths_without_legend.png', dpi=600)
        plt.legend(loc='best', fontsize=self.param.font_size, ncol=1)
        plt.savefig(f'{self.param.outdir}/L1_feature_paths_with_legend.png', dpi=600)        

        plt.close()

    def manually_set_best_alpha(self):
        # self.best_alpha = ?
        # print(f'Manually set best alpha = {self.best_alpha:.5g}\n')
        pass

    def select_features_at_best_alpha(self):
        # note that this is not the same as the feature paths plot, 
        # as the best alpha could be manually set, so it's not among the values of the alpha_grid
        # so we need to fit the model at exactly the best alpha
        scaler = StandardScaler()
        Xs = scaler.fit_transform(self.X)  # scale the whole dataset without train-test split

        lasso = Lasso(
            alpha=self.best_alpha,
            fit_intercept=True,
            max_iter=self.param.max_iter,
            tol=self.param.tol,
            random_state=self.param.random_state)

        lasso.fit(Xs, self.y)
        coefs = lasso.coef_.ravel().copy()  # shape: (n_features,)

        # select features with absolute coefficient value > 0
        idx = np.where(np.abs(coefs) > 0)[0]
        self.selected_features = [self.features[j] for j in idx]
        print(f'Selected features at best alpha = {self.best_alpha:.5g}: {self.selected_features}\n')

    def refit_with_selected_features(self):
        # use the selected features
        # no scaling, because it's for interpreting the features
        idx = [self.features.index(feature) for feature in self.selected_features]
        X = self.X[:, idx]

        lr = LinearRegression()  # no regularization, because it's not for prediction, just for interpretability
        lr.fit(X, self.y)
        print(f'Refitted with selected features: {self.selected_features}')
        print(f'Intercept: {lr.intercept_.ravel()[0]:.3f}')
        print('Coefficients:')
        for name, coef in zip(self.selected_features, lr.coef_.ravel()):
            print(f"  '{name}': {coef:.3f}")
        print()


class LinearStepwiseFeatureSelection:

    param: Parameters

    df: pd.DataFrame
    core_features: List[str]
    candidate_features: List[str]
    outcome: str

    y: np.ndarray
    k_fold: KFold
    selected_features: List[str]
    results: List[Dict[str, float]]

    def main(
            self,
            df: pd.DataFrame,
            core_features: List[str],
            candidate_features: List[str],
            outcome: str,
            parameters: Parameters):
        self.df = df.copy()
        self.core_features = core_features
        self.candidate_features = candidate_features
        self.outcome = outcome
        self.param = parameters

        print(f'--- Linear Stepwise Feature Selection ---\n')
        self.print_summary()

        # globally constant
        self.y = self.df[self.outcome].values
        self.k_fold = KFold(n_splits=self.param.cv_folds, shuffle=True, random_state=self.param.random_state)  # outcome is not binary, so we can't use StratifiedKFold
        
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
    
            cv_results = self.run_k_fold_cv(X=X, y=self.y)
    
            if best_feature is None or cv_results['train_mse'] < self.results[ith_step]['train_mse']:
                best_feature = candidate_feature
                self.results[ith_step] = cv_results
        
        assert best_feature is not None  # now we should have found the best feature for this step
        self.selected_features.append(best_feature)
        self.candidate_features.remove(best_feature)
        print(f'Selected features: {self.selected_features}')
        print(f'Train MSE: {self.results[ith_step]["train_mse"]}')
        print(f'Test MSE: {self.results[ith_step]["test_mse"]}\n')
                
    def run_k_fold_cv(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        e = np.full(shape=(self.param.cv_folds,), fill_value=np.nan)  # empty array of length cv_folds
        train_mse, test_mse = e.copy(), e.copy()
        for fold, (train_idx, test_idx) in enumerate(self.k_fold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # scale INSIDE the fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)  # use training set's scaler for test set

            ridge = Ridge(
                alpha=self.param.stepwise_alpha,
                fit_intercept=True,
                max_iter=self.param.max_iter,
                tol=self.param.tol,
                random_state=self.param.random_state)

            ridge.fit(X_train_scaled, y_train)

            y_hat_test = ridge.predict(X_test_scaled)
            y_hat_train = ridge.predict(X_train_scaled)
            train_mse[fold] = mean_squared_error(y_train, y_hat_train)
            test_mse[fold] = mean_squared_error(y_test, y_hat_test)

        return {
            'train_mse': train_mse.mean(),
            'train_mse_std': train_mse.std(ddof=1),  # ddof=1: delta degree of freedom = 1, that is n-1 for sample std
            'test_mse': test_mse.mean(),
            'test_mse_std': test_mse.std(ddof=1),
        }

    def plot_results(self):
        train_mse = [result['train_mse'] for result in self.results]
        train_mse_std = [result['train_mse_std'] for result in self.results]
        test_mse = [result['test_mse'] for result in self.results]
        test_mse_std = [result['test_mse_std'] for result in self.results]

        plt.figure(figsize=self.__figsize())
        plt.rcParams.update({'font.size': self.param.font_size})
        plt.errorbar(range(len(train_mse)), train_mse, yerr=train_mse_std, fmt="o-", capsize=3, label='train', markersize=3, linewidth=1, color='dimgray')
        plt.errorbar(range(len(test_mse)), test_mse, yerr=test_mse_std, fmt="o-", capsize=3, label='test', markersize=3, linewidth=1, color='orangered')
        plt.legend()
        plt.xlabel('Features')
        plt.ylabel('CV Mean Squared Error (mean ± SD over folds)')
        plt.xticks(range(len(self.selected_features)), self.selected_features)
        plt.xticks(rotation=45)
        plt.xlim(-0.5, len(self.selected_features) - 0.5)
        plt.grid(True, color='lightgray', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'{self.param.outdir}/stepwise_mean_squared_error.png', dpi=600)
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
            'train_mse': 'Train MSE',
            'train_mse_std': 'Train MSE SD',
            'test_mse': 'Test MSE',
            'test_mse_std': 'Test MSE SD',
        })
        df.to_csv(f'{self.param.outdir}/stepwise_results.csv')
