"""
预测模型模块
集成随机森林、LightGBM、XGBoost三种模型
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class Predictor:
    """多模型预测器

    支持随机森林、LightGBM、XGBoost三种模型的训练和评估
    """

    def __init__(self, test_size=0.2, random_state=42, n_estimators=100, verbose=True):
        """初始化预测器

        Args:
            test_size: 测试集比例
            random_state: 随机种子
            n_estimators: 基础树数量
            verbose: 是否打印详细信息
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _prepare_data(self, df, target_column):
        """准备训练和测试数据"""
        X = df.drop(columns=[target_column], errors='ignore')
        y = df[target_column] if target_column in df.columns else df.iloc[:, -1]

        # 处理缺失值（简单删除）
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]

        # 处理无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # 分离训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        return X, y

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """训练随机森林模型"""
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'model': rf,
            'predictions': y_pred
        }

    def train_lightgbm(self, X_train, X_test, y_train, y_test):
        """训练LightGBM模型"""
        try:
            import lightgbm as lgb

            # 将标签转换为整数类型
            y_train_int = y_train.astype(int)
            y_test_int = y_test.astype(int)

            lgb_model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                verbose=-1,
                force_col_wise=True
            )
            lgb_model.fit(X_train, y_train_int)
            y_pred = lgb_model.predict(X_test)

            return {
                'accuracy': accuracy_score(y_test_int, y_pred),
                'f1': f1_score(y_test_int, y_pred, average='weighted'),
                'model': lgb_model,
                'predictions': y_pred
            }
        except Exception as e:
            if self.verbose:
                print(f"LightGBM训练失败: {e}")
            return None

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """训练XGBoost模型"""
        try:
            import xgboost as xgb

            # 将标签转换为整数类型
            y_train_int = y_train.astype(int)
            y_test_int = y_test.astype(int)

            xgb_model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            xgb_model.fit(X_train, y_train_int)
            y_pred = xgb_model.predict(X_test)

            return {
                'accuracy': accuracy_score(y_test_int, y_pred),
                'f1': f1_score(y_test_int, y_pred, average='weighted'),
                'model': xgb_model,
                'predictions': y_pred
            }
        except Exception as e:
            if self.verbose:
                print(f"XGBoost训练失败: {e}")
            return None

    def train_and_evaluate(self, df, target_column):
        """训练并评估所有模型

        Args:
            df: 处理后的数据框
            target_column: 目标变量列名

        Returns:
            各模型的评估结果字典
        """
        if self.verbose:
            print("\n" + "=" * 50)
            print("开始训练和评估模型...")
            print("=" * 50)

        # 准备数据
        X, y = self._prepare_data(df, target_column)

        if len(X.columns) == 0:
            if self.verbose:
                print("警告: 没有可用特征")
            return {}

        if len(np.unique(y)) < 2:
            if self.verbose:
                print("警告: 目标变量只有一类")
            return {}

        if self.verbose:
            print(f"训练集大小: {len(self.X_train)}, 测试集大小: {len(self.X_test)}")
            print(f"特征数量: {len(X.columns)}")
            print(f"目标变量类别: {np.unique(y)}")

        # 训练随机森林
        if self.verbose:
            print("\n训练随机森林...")
        rf_result = self.train_random_forest(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        if rf_result:
            self.models['random_forest'] = rf_result['model']
            self.results['random_forest'] = {
                'accuracy': rf_result['accuracy'],
                'f1': rf_result['f1']
            }
            if self.verbose:
                print(f"随机森林 - 准确率: {rf_result['accuracy']:.4f}, F1: {rf_result['f1']:.4f}")

        # 训练LightGBM
        if self.verbose:
            print("\n训练LightGBM...")
        lgb_result = self.train_lightgbm(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        if lgb_result:
            self.models['lightgbm'] = lgb_result['model']
            self.results['lightgbm'] = {
                'accuracy': lgb_result['accuracy'],
                'f1': lgb_result['f1']
            }
            if self.verbose:
                print(f"LightGBM - 准确率: {lgb_result['accuracy']:.4f}, F1: {lgb_result['f1']:.4f}")

        # 训练XGBoost
        if self.verbose:
            print("\n训练XGBoost...")
        xgb_result = self.train_xgboost(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        if xgb_result:
            self.models['xgboost'] = xgb_result['model']
            self.results['xgboost'] = {
                'accuracy': xgb_result['accuracy'],
                'f1': xgb_result['f1']
            }
            if self.verbose:
                print(f"XGBoost - 准确率: {xgb_result['accuracy']:.4f}, F1: {xgb_result['f1']:.4f}")

        if self.verbose:
            print("\n" + "=" * 50)
            print("模型训练完成!")
            print("=" * 50)

        return self.results

    def get_best_model(self):
        """获取最佳模型

        Returns:
            (模型名称, 模型对象, 评估指标)
        """
        if not self.results:
            return None, None, None

        best_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        return best_name, self.models[best_name], self.results[best_name]

    def get_all_results(self):
        """获取所有模型结果"""
        return self.results

    def print_summary(self):
        """打印结果摘要"""
        if not self.results:
            print("没有可用的模型结果")
            return

        print("\n" + "=" * 60)
        print("模型评估结果摘要")
        print("=" * 60)
        print(f"{'模型名称':<20} {'准确率':<15} {'F1分数':<15}")
        print("-" * 60)

        for name, metrics in sorted(self.results.items(), key=lambda x: -x[1]['accuracy']):
            print(f"{name:<20} {metrics['accuracy']:<15.4f} {metrics['f1']:<15.4f}")

        best_name, _, best_metrics = self.get_best_model()
        print("-" * 60)
        print(f"最佳模型: {best_name} (准确率: {best_metrics['accuracy']:.4f})")
        print("=" * 60)
