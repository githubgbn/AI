"""
数据处理流水线
实现7种数据处理步骤，每种支持多种算法
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter
import pywt


class DataProcessor:
    """数据处理流水线"""

    def __init__(self):
        self.config = {}

    def process_missing_value(self, df, method='mean'):
        """缺失值处理

        Args:
            df: 输入数据框
            method: 处理方法
                - delete: 删除缺失行
                - mean: 均值填充
                - median: 中位数填充
                - forward_fill: 前向填充
                - backward_fill: 后向填充
                - interpolate: 插值填充
                - knn: KNN填充
                - multivariate_impute: 多重插补
        """
        if df is None or len(df) == 0:
            return df

        df = df.copy()

        if method == 'delete':
            return df.dropna()
        elif method == 'mean':
            return df.fillna(df.mean())
        elif method == 'median':
            return df.fillna(df.median())
        elif method == 'forward_fill':
            return df.fillna(method='ffill')
        elif method == 'backward_fill':
            return df.fillna(method='bfill')
        elif method == 'interpolate':
            return df.interpolate(method='linear')
        elif method == 'knn':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=min(5, len(df)))
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            return df
        elif method == 'multivariate_impute':
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = IterativeImputer(max_iter=10)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            return df
        return df

    def process_anomaly(self, df, method='zscore', threshold=3):
        """异常处理

        Args:
            df: 输入数据框
            method: 处理方法
                - zscore: Z-score方法
                - iqr: IQR方法
                - isolation_forest: 孤立森林
                - percentile: 百分位法
            threshold: 异常阈值
        """
        if df is None or len(df) == 0:
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return df

        if method == 'zscore':
            for col in numeric_cols:
                if df[col].std() == 0:
                    continue
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        elif method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
        elif method == 'isolation_forest':
            iso = IsolationForest(contamination=0.1, random_state=42)
            try:
                mask = iso.fit_predict(df[numeric_cols]) != -1
                df = df[mask]
            except:
                pass
        elif method == 'percentile':
            for col in numeric_cols:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df = df[(df[col] >= lower) & (df[col] <= upper)]

        return df

    def process_filter(self, df, method='median_filter', window_size=3):
        """滤波降噪

        Args:
            df: 输入数据框
            method: 处理方法
                - median_filter: 中值滤波
                - mean_filter: 均值滤波
                - gaussian_filter: 高斯滤波
                - kalman_filter: 卡尔曼滤波
                - wavelet_denoise: 小波降噪
            window_size: 滤波窗口大小
        """
        if df is None or len(df) == 0:
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            try:
                if method == 'median_filter':
                    df[col] = median_filter(df[col].values, size=window_size)
                elif method == 'mean_filter':
                    df[col] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
                elif method == 'gaussian_filter':
                    df[col] = gaussian_filter(df[col].values, sigma=window_size)
                elif method == 'kalman_filter':
                    # 简化的卡尔曼滤波
                    from statsmodels.tsa.api import SimpleExpSmoothing
                    try:
                        ses = SimpleExpSmoothing(df[col].values, initialization_method="estimated")
                        df[col] = ses.fit(smoothing_level=0.3).fittedvalues
                    except:
                        df[col] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
                elif method == 'wavelet_denoise':
                    if len(df[col]) >= 4:
                        coeffs = pywt.wavedec(df[col].values, 'db4', level=2)
                        threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(df)))
                        if threshold > 0:
                            coeffs[-1] = pywt.threshold(coeffs[-1], threshold, mode='soft')
                            reconstructed = pywt.waverec(coeffs, 'db4')
                            df[col] = reconstructed[:len(df)]
            except:
                continue

        return df

    def process_diff(self, df, method='first_order', target_column=None):
        """差分算子

        Args:
            df: 输入数据框
            method: 处理方法
                - first_order: 一阶差分
                - second_order: 二阶差分
                - seasonal_diff: 季节性差分
            target_column: 目标变量列名（差分时跳过）
        """
        if df is None or len(df) == 0:
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 跳过目标列
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)

        for col in numeric_cols:
            try:
                if method == 'first_order':
                    df[col] = df[col].diff()
                elif method == 'second_order':
                    df[col] = df[col].diff().diff()
                elif method == 'seasonal_diff':
                    df[col] = df[col].diff(periods=7)  # 假设季节周期为7
            except:
                continue

        return df.dropna()

    def process_resample(self, df, method='downsample', target_size=None, random_state=42):
        """重采样

        Args:
            df: 输入数据框
            method: 处理方法
                - upsample: 上采样
                - downsample: 下采样
                - interpolate_resample: 插值重采样
                - smote: SMOTE过采样
            target_size: 目标样本数量
            random_state: 随机种子
        """
        if df is None or len(df) == 0:
            return df

        df = df.copy()

        if len(df) <= 10:
            return df

        if target_size is None:
            target_size = max(int(len(df) * 0.8), 10)

        if method == 'downsample':
            if target_size >= len(df):
                return df
            np.random.seed(random_state)
            indices = np.random.choice(len(df), target_size, replace=False)
            return df.iloc[sorted(indices)]
        elif method == 'upsample':
            if target_size <= len(df):
                return df
            indices = np.linspace(0, len(df)-1, target_size)
            result = pd.DataFrame()
            for col in df.columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    result[col] = np.interp(indices, np.arange(len(df)), df[col].values)
                else:
                    result[col] = df[col].iloc[np.clip(np.round(indices).astype(int), 0, len(df)-1)].values
            return result
        elif method == 'interpolate_resample':
            if target_size == len(df):
                return df
            indices = np.linspace(0, len(df)-1, target_size)
            result = pd.DataFrame()
            for col in df.columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    result[col] = np.interp(indices, np.arange(len(df)), df[col].values)
                else:
                    result[col] = df[col].iloc[np.clip(np.round(indices).astype(int), 0, len(df)-1)].values
            return result
        elif method == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0 and self._has_categorical(df):
                    # 分离数值和分类列
                    cat_cols = df.select_dtypes(exclude=[np.number]).columns
                    X = df[numeric_cols].values
                    y = df.iloc[:, -1].values

                    if len(np.unique(y)) > 1 and len(X) > len(numeric_cols):
                        smote = SMOTE(random_state=random_state)
                        X_res, y_res = smote.fit_resample(X, y)
                        result = pd.DataFrame(X_res, columns=numeric_cols)
                        if len(cat_cols) > 0:
                            for col in cat_cols:
                                result[col] = df[col].iloc[0]
                        result[df.columns[-1]] = y_res
                        return result
            except ImportError:
                pass
            return df

        return df

    def _has_categorical(self, df):
        """检查是否有分类列"""
        return len(df.select_dtypes(exclude=[np.number]).columns) > 0

    def process_normalize(self, df, method='standard', target_column=None):
        """标准化/归一化

        Args:
            df: 输入数据框
            method: 处理方法
                - minmax: MinMax归一化
                - standard: Standard标准化
                - zscore: Z-score标准化
                - l2: L2归一化
                - robust: RobustScaler
            target_column: 目标变量列名（标准化时跳过）
        """
        if df is None or len(df) == 0:
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # 标准化时跳过目标列
        if target_column and target_column in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != target_column]

        if len(numeric_cols) == 0:
            return df

        data = df[numeric_cols].values

        try:
            if method == 'minmax':
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(data)
            elif method == 'standard':
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(data)
            elif method == 'zscore':
                for i, col in enumerate(numeric_cols):
                    mean_val = np.nanmean(data[:, i])
                    std_val = np.nanstd(data[:, i])
                    if std_val > 0:
                        df[col] = (data[:, i] - mean_val) / std_val
            elif method == 'l2':
                norm = np.linalg.norm(data, axis=0)
                norm[norm == 0] = 1
                df[numeric_cols] = data / norm
            elif method == 'robust':
                scaler = RobustScaler()
                df[numeric_cols] = scaler.fit_transform(data)
        except:
            pass

        return df

    def process(self, df, config, target_column=None):
        """执行完整的数据处理流水线

        Args:
            df: 输入数据框
            config: 处理配置，包含各步骤的算法选择
            target_column: 目标变量列名

        Returns:
            处理后的数据框
        """
        if df is None or len(df) == 0:
            return df

        # 按顺序执行各处理步骤
        df = self.process_missing_value(df, config.get('missing_method', 'mean'))
        if df is None or len(df) < 5:
            return pd.DataFrame(columns=df.columns) if df is not None else df

        df = self.process_anomaly(df, config.get('anomaly_method', 'zscore'))
        if df is None or len(df) < 5:
            return pd.DataFrame(columns=df.columns) if df is not None else df

        df = self.process_filter(df, config.get('filter_method', 'median_filter'))
        if df is None or len(df) < 5:
            return pd.DataFrame(columns=df.columns) if df is not None else df

        df = self.process_diff(df, config.get('diff_method', 'first_order'), target_column)
        if df is None or len(df) < 5:
            return pd.DataFrame(columns=df.columns) if df is not None else df

        df = self.process_resample(df, config.get('resample_method', 'downsample'))
        if df is None or len(df) < 5:
            return pd.DataFrame(columns=df.columns) if df is not None else df

        df = self.process_normalize(df, config.get('normalize_method', 'standard'), target_column)

        return df

    def get_default_config(self):
        """获取默认配置"""
        return {
            'missing_method': 'mean',
            'anomaly_method': 'zscore',
            'filter_method': 'median_filter',
            'diff_method': 'first_order',
            'resample_method': 'downsample',
            'normalize_method': 'standard'
        }
