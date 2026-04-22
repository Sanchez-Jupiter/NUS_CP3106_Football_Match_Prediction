"""
Reproduce the 2017 EPL Bayesian-network paper in a controlled way.

Paper:
Predicting Football Matches Results using Bayesian Networks for English Premier League (2017)

What this script does:
1. Downloads EPL CSV files for seasons 2010-2011, 2011-2012, 2012-2013 from football-data.co.uk.
2. Builds a Bayesian-network classifier using pgmpy.
3. Evaluates season by season with K-fold cross validation, matching the paper's setup as closely as practical.
4. Reports two experimental views:
   - paper_like: uses the match-summary attributes shown in the paper table.
   - paper_like_with_ft_goals: additionally includes full-time goals, which is likely label leakage.

Important caveat:
Most attributes listed in the paper table are not true pre-match features. Shots, shots on target,
corners, fouls, cards, and halftime goals are only known during or after the match. The reported
75.09% accuracy is therefore not directly comparable to a strict pre-match prediction task.
"""

from __future__ import annotations

# ==================== 导入 ====================
import argparse  # 命令行参数解析
import json  # JSON文件处理
from dataclasses import dataclass  # 数据类装饰器
from pathlib import Path  # 路径处理
from typing import Iterable  # 类型提示

import numpy as np  # 数值计算
import pandas as pd  # 数据处理
from pgmpy.estimators import BayesianEstimator, HillClimbSearch  # 贝叶斯网络估计和学习
from pgmpy.inference import VariableElimination  # 贝叶斯网络推理
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # 评估指标
from sklearn.model_selection import KFold  # K折交叉验证

# 处理不同版本的pgmpy库兼容性
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
except ImportError:
    from pgmpy.models import BayesianNetwork as BayesianModel

# 处理不同版本的BDeu评分方法
try:
    from pgmpy.estimators import BDeu
except ImportError:
    from pgmpy.estimators import BDeuScore as BDeu


# ==================== 路径和目录配置 ====================
ROOT_DIR = Path(__file__).resolve().parent  # 当前脚本所在目录
RAW_DIR = ROOT_DIR / "data" / "raw" / "football_data_uk"  # 原始数据目录
REPORT_DIR = ROOT_DIR / "reports"  # 报告输出目录
RAW_DIR.mkdir(parents=True, exist_ok=True)  # 创建目录（如不存在）
REPORT_DIR.mkdir(parents=True, exist_ok=True)  # 创建目录（如不存在）

# ==================== 全局配置常量 ====================
SEED = 42  # 随机种子，确保结果可复现
TARGET_COL = "FTR"  # 目标列：全场比赛结果 (FTR = Full Time Result)
LABEL_ORDER = ["A", "D", "H"]  # 标签顺序：客队胜/平/主队胜

# ==================== 数据源URL配置 ====================
SEASON_URLS = {
    "2010-2011": "https://www.football-data.co.uk/mmz4281/1011/E0.csv",  # EPL赛季数据
    "2011-2012": "https://www.football-data.co.uk/mmz4281/1112/E0.csv",  # EPL赛季数据
    "2012-2013": "https://www.football-data.co.uk/mmz4281/1213/E0.csv",  # EPL赛季数据
}

# ==================== 论文使用的特征列 ====================
PAPER_TABLE_FEATURES = [
    "HomeTeam",
    "AwayTeam",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
    "HTHG",  # 主队中场进球
    "HTAG",  # 客队中场进球
]
# 注意：论文使用的这些特征大部分是赛后特征（如射门数、射正数、角球数等），
# 不是严格的赛前特征，因此报告的准确度不适用于真实的赛前预测任务

# ==================== 数据类：实验结果 ====================
@dataclass
class ExperimentResult:
    """存储单个赛季的实验结果"""
    season: str  # 赛季
    feature_mode: str  # 特征模式 (paper_like 或 paper_like_with_ft_goals)
    accuracy: float  # 准确率
    macro_f1: float  # 宏平均F1分数
    support: int  # 样本数量
    confusion: list[list[int]]  # 混淆矩阵


def _load_or_download_season(season: str) -> pd.DataFrame:
    """
    加载或下载指定赛季的EPL数据。
    
    如果本地缓存不存在，则从football-data.co.uk下载CSV文件并保存到本地。
    否则直接读取本地缓存的CSV文件。
    
    Args:
        season: 赛季字符串，格式为 "YYYY-YYYY" (如 "2010-2011")
    
    Returns:
        包含该赛季所有比赛数据的DataFrame
    """
    csv_path = RAW_DIR / f"epl_{season}.csv"  # 本地缓存文件路径
    if not csv_path.exists():
        # 如果本地缓存不存在，则从网络下载
        df = pd.read_csv(SEASON_URLS[season])
        df.to_csv(csv_path, index=False)  # 保存到本地以便下次使用
    else:
        # 直接读取本地缓存
        df = pd.read_csv(csv_path)
    return df


def _prepare_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    准备并验证数据集所需的列。
    
    检查所有必需的列是否存在，移除缺失目标值的行，并确保目标列是字符串类型。
    
    Args:
        df: 原始数据DataFrame
        feature_cols: 需要的特征列名列表
    
    Returns:
        清理后的DataFrame，仅包含所需的列
    
    Raises:
        ValueError: 如果缺少必需的列
    """
    # 检查所需列
    required_cols = feature_cols + [TARGET_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 提取所需列
    prepared = df[required_cols].copy()
    # 移除目标列缺失的行
    prepared = prepared.dropna(subset=[TARGET_COL])
    # 转换目标列为字符串类型
    prepared[TARGET_COL] = prepared[TARGET_COL].astype(str)
    return prepared


def _discretize_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    bins: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    离散化一个交叉验证折叠中的特征。
    
    贝叶斯网络要求离散化的特征。该函数将数值特征转换为分类值：
    - 如果数值特征的唯一值少于等于bins数，将其映射为离散值
    - 否则使用等频率分箱法将特征分成若干箱
    
    Args:
        train_df: 训练集数据
        test_df: 测试集数据
        feature_cols: 需要离散化的特征列名
        bins: 分箱数量（默认4）
    
    Returns:
        (离散化后的训练集, 离散化后的测试集)
    """
    # 初始化输出DataFrame
    train_out = pd.DataFrame(index=train_df.index)
    test_out = pd.DataFrame(index=test_df.index)

    for col in feature_cols:
        # 检查是否为数值列
        if pd.api.types.is_numeric_dtype(train_df[col]):
            # 转换为数值，缺失值填充为0.0
            train_values = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)
            test_values = pd.to_numeric(test_df[col], errors="coerce").fillna(0.0)

            # 获取训练集中该列的唯一值
            unique_values = np.unique(train_values.to_numpy())
            
            # 情形1：唯一值少于阈值，直接映射为离散标签
            if len(unique_values) <= bins:
                # 创建映射字典：值 -> 标签 (v_0, v_1, ...)
                mapping = {value: f"v_{idx}" for idx, value in enumerate(sorted(unique_values.tolist()))}
                sorted_values = np.array(sorted(unique_values.tolist()), dtype=float)
                # 训练集：直接按映射转换
                train_out[col] = train_values.map(mapping).astype(str)
                # 测试集：对于未见过的值，找最近的已知值
                test_out[col] = test_values.map(
                    lambda value: mapping.get(value, mapping[sorted_values[np.argmin(np.abs(sorted_values - value))]])
                ).astype(str)
            # 情形2：唯一值较多，使用等频率分箱
            else:
                # 计算分箱边界：使用等频分位数
                quantiles = np.linspace(0, 1, bins + 1)
                edges = np.quantile(train_values, quantiles)
                # 移除重复的边界值
                edges = np.unique(edges)
                
                # 如果边界太少（无法正常分箱），将所有值放入一个箱
                if len(edges) <= 2:
                    train_out[col] = "bin_0"
                    test_out[col] = "bin_0"
                else:
                    # 设置边界为[-inf, ..., +inf]以包含所有值
                    edges = edges.astype(float)
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    # 使用pd.cut根据边界进行分箱
                    train_bins = pd.cut(train_values, bins=edges, include_lowest=True, duplicates="drop")
                    test_bins = pd.cut(test_values, bins=edges, include_lowest=True, duplicates="drop")
                    train_out[col] = train_bins.astype(str)
                    test_out[col] = test_bins.astype(str)
        else:
            # 非数值列：直接转为字符串，缺失值标记为"missing"
            train_out[col] = train_df[col].fillna("missing").astype(str)
            test_out[col] = test_df[col].fillna("missing").astype(str)

    # 添加目标列（已经是字符串）
    train_out[TARGET_COL] = train_df[TARGET_COL].astype(str)
    test_out[TARGET_COL] = test_df[TARGET_COL].astype(str)
    return train_out, test_out


def _fit_bn(train_df: pd.DataFrame, feature_cols: list[str]) -> BayesianModel:
    """
    根据训练数据学习贝叶斯网络的结构和参数。
    
    使用Hill Climbing搜索算法学习网络结构，使用BDeu评分方法。
    然后使用贝叶斯估计器拟合条件概率表（CPT）。
    
    Args:
        train_df: 离散化的训练数据
        feature_cols: 特征列名列表
    
    Returns:
        学习完成的贝叶斯网络模型
    """
    # 步骤1：使用Hill Climbing算法学习网络结构
    search = HillClimbSearch(train_df)  # 创建搜索对象
    learned = search.estimate(scoring_method=BDeu(train_df), max_indegree=3, show_progress=False)
    # max_indegree=3: 每个节点最多有3个入边（父节点）

    # 步骤2：提取边（依赖关系）
    edges = list(learned.edges()) if hasattr(learned, "edges") else list(learned)
    # 创建贝叶斯网络模型
    model = BayesianModel(edges)

    # 步骤3：确保所有特征和目标都作为节点存在
    for col in feature_cols + [TARGET_COL]:
        if col not in model.nodes():
            model.add_node(col)  # 添加孤立节点（如果有的话）

    if TARGET_COL not in model.nodes():
        model.add_node(TARGET_COL)

    # 步骤4：拟合条件概率表（CPT）
    model.fit(
        train_df,
        estimator=BayesianEstimator,
        prior_type="BDeu",  # 使用BDeu先验
        equivalent_sample_size=10,  # 等效样本大小
    )
    return model


def _predict_bn(model: BayesianModel, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    使用贝叶斯网络进行预测。
    
    对于每个测试样本，根据特征值（证据）进行变量消除推理，
    获取目标变量的后验概率分布，并选择概率最高的类别作为预测。
    
    Args:
        model: 训练完成的贝叶斯网络模型
        test_df: 离散化的测试数据
        feature_cols: 特征列名列表
    
    Returns:
        预测标签数组
    """
    # 创建推理对象
    infer = VariableElimination(model)
    predictions: list[str] = []

    # 逐行进行预测
    for _, row in test_df.iterrows():
        # 将特征值转换为证据字典
        evidence = {col: str(row[col]) for col in feature_cols}
        # 进行推理：给定证据，查询目标变量的后验概率
        query = infer.query(variables=[TARGET_COL], evidence=evidence, show_progress=False)
        # 获取目标变量的可能取值
        state_names = query.state_names[TARGET_COL]
        # 选择概率最高的类别
        pred_idx = int(np.argmax(query.values))
        predictions.append(str(state_names[pred_idx]))

    return np.asarray(predictions, dtype=object)


def _run_cv_for_season(season: str, feature_cols: list[str], n_splits: int) -> ExperimentResult:
    """
    对单个赛季执行K折交叉验证，使用贝叶斯网络进行预测。
    
    步骤：
    1. 加载赛季数据
    2. 分割成K个折叠
    3. 对每个折叠：
       - 离散化特征
       - 学习贝叶斯网络
       - 进行预测
    4. 计算汇总的准确率和F1分数
    
    Args:
        season: 赛季字符串
        feature_cols: 特征列名列表
        n_splits: 折叠数
    
    Returns:
        该赛季的实验结果
    """
    # 加载并准备数据
    season_df = _load_or_download_season(season)
    season_df = _prepare_columns(season_df, feature_cols)

    # 创建K折分割器（随机分割）
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    all_true: list[str] = []  # 存储所有真实标签
    all_pred: list[str] = []  # 存储所有预测标签

    # 执行K折交叉验证
    for train_idx, test_idx in splitter.split(season_df):
        # 分割训练集和测试集
        train_df = season_df.iloc[train_idx].reset_index(drop=True)
        test_df = season_df.iloc[test_idx].reset_index(drop=True)
        
        # 离散化特征（贝叶斯网络所需）
        train_disc, test_disc = _discretize_fold(train_df, test_df, feature_cols)

        # 训练贝叶斯网络
        model = _fit_bn(train_disc, feature_cols)
        # 进行预测
        pred = _predict_bn(model, test_disc, feature_cols)
        # 收集真实标签和预测标签
        all_true.extend(test_disc[TARGET_COL].tolist())
        all_pred.extend(pred.tolist())

    # 转换为numpy数组
    y_true = np.asarray(all_true, dtype=object)
    y_pred = np.asarray(all_pred, dtype=object)
    
    # 返回汇总的实验结果
    return ExperimentResult(
        season=season,
        feature_mode="",  # 将由调用者设置
        accuracy=float(accuracy_score(y_true, y_pred)),  # 整体准确率
        macro_f1=float(f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER)),  # 宏平均F1分数
        support=int(len(y_true)),  # 总样本数
        confusion=confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),  # 混淆矩阵
    )


def _format_result_block(title: str, results: Iterable[ExperimentResult]) -> str:
    """
    格式化实验结果为可读的文本块。
    
    为每个赛季的结果创建详细的文本报告，
    包括单个赛季的准确率、F1分数、混淆矩阵，
    以及加权平均的指标。
    
    Args:
        title: 报告标题
        results: ExperimentResult对象的迭代器
    
    Returns:
        格式化后的报告文本
    """
    lines = ["=" * 72, title, "=" * 72, ""]
    # 收集指标用于计算加权平均
    accs = []
    f1s = []
    supports = []

    # 逐个赛季格式化结果
    for result in results:
        accs.append(result.accuracy)
        f1s.append(result.macro_f1)
        supports.append(result.support)
        # 添加赛季信息
        lines.append(f"Season: {result.season}")
        lines.append(f"Accuracy: {result.accuracy:.4f}")
        lines.append(f"Macro-F1: {result.macro_f1:.4f}")
        lines.append(f"Support: {result.support}")
        lines.append("Confusion matrix [A, D, H]:")
        lines.append(str(np.asarray(result.confusion)))
        lines.append("")

    # 计算加权平均（按样本数加权）
    weighted_acc = float(np.average(accs, weights=supports))
    weighted_f1 = float(np.average(f1s, weights=supports))
    lines.append(f"Weighted average accuracy: {weighted_acc:.4f}")
    lines.append(f"Weighted average macro-F1: {weighted_f1:.4f}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """
    主程序：复现2017年EPL贝叶斯网络论文。
    
    流程：
    1. 解析命令行参数
    2. 对指定的赛季执行K折交叉验证
    3. 生成报告（文本和JSON格式）
    4. 保存结果到文件
    """
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="Reproduce the 2017 EPL Bayesian-network paper.")
    parser.add_argument(
        "--folds", 
        type=int, 
        default=10, 
        help="Number of CV folds per season (默认10折)."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(SEASON_URLS.keys()),
        choices=list(SEASON_URLS.keys()),
        help="Subset of seasons to evaluate (默认评估所有赛季).",
    )
    parser.add_argument(
        "--include-fulltime-goals",
        action="store_true",
        help="Include FTHG/FTAG from the paper table. This is direct leakage and only for comparison.",
    )
    args = parser.parse_args()

    # ==================== 特征集合配置 ====================
    paper_like_features = PAPER_TABLE_FEATURES.copy()
    experiment_name = "paper_like"  # 基础特征集合
    
    # 可选：添加全场进球数（这是直接的标签泄露，仅用于对比测试）
    if args.include_fulltime_goals:
        paper_like_features += ["FTHG", "FTAG"]
        experiment_name = "paper_like_with_ft_goals"

    # ==================== 执行实验 ====================
    results: list[ExperimentResult] = []
    # 对每个赛季进行K折交叉验证
    for season in args.seasons:
        print(f"Processing season {season}...")
        result = _run_cv_for_season(season, paper_like_features, args.folds)
        result.feature_mode = experiment_name
        results.append(result)

    # ==================== 生成报告 ====================
    # 格式化结果为文本报告
    report_text = _format_result_block(
        f"BAYESIAN NETWORK EPL PAPER REPRODUCTION - {experiment_name.upper()}",
        results,
    )
    
    # 添加重要说明
    report_text += "\n" + "Notes:\n"
    report_text += "- The paper's reported 75.09% average accuracy uses match-summary attributes, not strict pre-match features.\n"
    report_text += "- HS/HST/HC/HF/HY/HR and their away-team counterparts are only known after the match progresses.\n"
    report_text += "- HTHG/HTAG are halftime features, still not pre-match.\n"
    if args.include_fulltime_goals:
        report_text += "- FTHG/FTAG are direct label leakage and should only be used to test whether the paper setup is leaky.\n"

    # ==================== 保存结果 ====================
    # 保存文本报告
    report_path = REPORT_DIR / f"paper_bayesian_epl_{experiment_name}.txt"
    report_path.write_text(report_text, encoding="utf-8")

    # 创建JSON格式的摘要
    summary = {
        "experiment": experiment_name,  # 实验名称
        "folds": args.folds,  # 折叠数
        "seasons": args.seasons,  # 评估的赛季
        "features": paper_like_features,  # 使用的特征
        "results": [result.__dict__ for result in results],  # 详细结果
    }
    # 保存JSON摘要
    summary_path = REPORT_DIR / f"paper_bayesian_epl_{experiment_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ==================== 输出结果 ====================
    print(report_text)
    print(f"\nReport saved to: {report_path}")
    print(f"Summary saved to: {summary_path}")


# ==================== 入口点 ====================
if __name__ == "__main__":
    # 运行主程序
    main()