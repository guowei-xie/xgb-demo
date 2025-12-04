"""
预测结果分析模块

生成预测概率分布与真实续报率关系图，便于快速诊断模型效果。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import platform
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

# 配置matplotlib中文字体支持
_configured_chinese_font = False
_season_order_map = {"寒": 0, "春": 1, "暑": 2, "秋": 3}


def _extract_term_sort_key(term: str) -> tuple[int, int, int, str]:
    """
    生成学期排序键：年份优先，其次季节（寒/春/暑/秋），最后按期次数字。
    """
    if term is None:
        return (0, len(_season_order_map), 0, "")

    text = str(term)
    year_match = re.search(r"(20\d{2})", text)
    year = int(year_match.group(1)) if year_match else 0
    remainder = text
    if year_match:
        remainder = remainder.replace(year_match.group(1), "", 1)

    season_idx = len(_season_order_map)
    for season, idx in _season_order_map.items():
        if season in remainder:
            season_idx = idx
            break

    remainder_digits = re.sub(r"(20\d{2})", "", remainder)
    period_match = re.search(r"(\d+)", remainder_digits)
    period = int(period_match.group(1)) if period_match else 0
    return (year, season_idx, period, text)


def _sort_terms(term_series: pd.Series) -> list[str]:
    """
    对学期标签进行自定义排序，确保呈现顺序符合年-季节-期次。
    """
    unique_terms = term_series.dropna().astype(str).drop_duplicates().tolist()
    return sorted(unique_terms, key=_extract_term_sort_key)


def _configure_chinese_font():
    """
    配置matplotlib使用支持中文的字体，解决中文显示乱码问题。
    """
    global _configured_chinese_font
    if _configured_chinese_font:
        return
    
    system = platform.system()
    chinese_fonts = []
    
    if system == "Darwin":  # macOS
        chinese_fonts = ["PingFang SC", "STHeiti", "Arial Unicode MS", "Heiti TC"]
    elif system == "Windows":
        chinese_fonts = ["SimHei", "Microsoft YaHei", "KaiTi", "FangSong"]
    else:  # Linux
        chinese_fonts = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Droid Sans Fallback"]
    
    # 获取系统可用字体列表
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    
    # 尝试设置中文字体
    font_found = False
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
            font_found = True
            print(f"✓ 已配置中文字体: {font_name}")
            break
    
    if not font_found:
        # 如果预设字体都不可用，尝试从系统字体列表中找到第一个中文字体
        chinese_keywords = ["PingFang", "Heiti", "SimHei", "YaHei", "WenQuanYi", "Noto Sans CJK"]
        for font in available_fonts:
            if any(keyword in font for keyword in chinese_keywords):
                plt.rcParams["font.sans-serif"] = [font]
                plt.rcParams["axes.unicode_minus"] = False
                print(f"✓ 已自动检测并配置中文字体: {font}")
                font_found = True
                break
    
    if not font_found:
        # 最后尝试：设置通用配置，至少解决负号问题
        plt.rcParams["axes.unicode_minus"] = False
        print("⚠️ 警告: 未找到合适的中文字体，图表中的中文可能显示为方框")
    
    _configured_chinese_font = True


def analyze_prediction_quality(
    probabilities: Iterable[float],
    labels: Iterable[float],
    output_path: Path,
    dataset_name: str = "测试集",
) -> Path:
    """
    构建预测概率分析图（直方图+散点拟合），并保存到本地。

    Args:
        probabilities: 预测概率序列。
        labels: 实际标签序列。
        output_path: 图片输出路径。
        dataset_name: 图表标题中使用的名称。

    Returns:
        生成的图片路径。
    """
    _configure_chinese_font()  # 确保中文字体已配置
    proba_series, label_series = _sanitize_inputs(probabilities, labels)
    bucket_summary = _summarize_by_bucket(proba_series, label_series)

    fig, (ax_hist, ax_scatter) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        height_ratios=(2, 1.6),
        constrained_layout=True,
    )

    _plot_histogram(ax_hist, bucket_summary, dataset_name)
    _plot_scatter_with_fit(ax_scatter, bucket_summary)

    # 根据实际数据动态设置横轴范围，截断到实际最大值
    max_prob = proba_series.max()
    # 横轴显示到实际最大值加2%边距，但不超过1.0
    x_max = min(1.0, max_prob * 1.02)
    ax_hist.set_xlim(0, x_max)
    ax_scatter.set_xlabel("预测概率")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✓ 预测分析图已保存至: {output_path}")
    return output_path


def _sanitize_inputs(probabilities: Iterable[float], labels: Iterable[float]) -> Tuple[pd.Series, pd.Series]:
    """
    将输入序列转换为对齐的Series，并完成基础校验。
    """
    proba_series = pd.Series(probabilities, name="probability").astype(float)
    label_series = pd.Series(labels, name="label").astype(float)

    if proba_series.isnull().any():
        raise ValueError("预测概率包含缺失值，无法分析")
    if label_series.isnull().any():
        raise ValueError("实际标签包含缺失值，无法分析")
    if len(proba_series) != len(label_series):
        raise ValueError("预测概率与标签长度不一致")

    return proba_series, label_series


def _summarize_by_bucket(proba_series: pd.Series, label_series: pd.Series) -> pd.DataFrame:
    """
    以1%概率为分桶，统计样本数与实际续报率。
    """
    df = pd.DataFrame({"probability": proba_series, "label": label_series})
    df["bucket"] = (df["probability"] * 100).clip(0, 99).astype(int)

    summary = (
        df.groupby("bucket")
        .agg(
            sample_count=("label", "size"),
            renewal_rate=("label", "mean"),
            avg_probability=("probability", "mean"),
        )
        .reset_index()
    )
    summary["bucket_left"] = summary["bucket"] / 100
    summary["bucket_center"] = summary["bucket_left"] + 0.5 / 100
    return summary


def _plot_histogram(ax: plt.Axes, bucket_summary: pd.DataFrame, dataset_name: str) -> None:
    """
    绘制预测概率分布直方图。
    """
    ax.bar(
        bucket_summary["bucket_left"],
        bucket_summary["sample_count"],
        width=0.01,
        align="edge",
        color="#4C72B0",
        edgecolor="white",
    )
    ax.set_ylabel("样本数")
    ax.set_title(f"{dataset_name} - 预测概率分布（1%分桶）")
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def _plot_scatter_with_fit(ax: plt.Axes, bucket_summary: pd.DataFrame) -> None:
    """
    绘制续报率散点，并叠加y=x对角参考线。
    """
    x = bucket_summary["avg_probability"].to_numpy()
    y = bucket_summary["renewal_rate"].to_numpy()

    if len(x) < 2:
        raise ValueError("有效分桶不足，无法进行线性拟合")

    ax.scatter(x, y, color="#55A868", alpha=0.85, label="实际续报率")
    diag_min = max(0.0, min(x.min(), y.min()))
    diag_max = min(1.0, max(x.max(), y.max()))
    diag_x = np.linspace(diag_min, diag_max, 100)
    ax.plot(
        diag_x,
        diag_x,
        color="#C44E52",
        linewidth=2,
        linestyle="--",
        label="理想对角线 y=x",
    )
    ax.set_ylabel("实际续报率")
    ax.legend()
    ax.grid(axis="both", linestyle="--", alpha=0.4)


def plot_level_performance_by_term(
    df: pd.DataFrame,
    term_col: str,
    level_col: str,
    label_col: str,
    output_path: Path,
    dataset_name: str = "测试集",
) -> Path:
    """
    按学期粒度聚合各等级续报率，并绘制抖动散点+误差棒。

    Args:
        df: 含有level_tag、学期与真实标签的数据集。
        term_col: 学期列名。
        level_col: 等级标签列名。
        label_col: 真实续报标签列名。
        output_path: 图片输出路径。
        dataset_name: 用于标题描述的数据集名称。

    Returns:
        生成的图片路径。
    """
    _configure_chinese_font()
    required_cols = [term_col, level_col, label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必要列: {', '.join(missing_cols)}")

    clean_df = df.dropna(subset=required_cols).copy()
    if clean_df.empty:
        raise ValueError("无法绘制等级-学期图：相关列均为空")

    clean_df[label_col] = clean_df[label_col].astype(float)
    agg_df = (
        clean_df.groupby([term_col, level_col])[label_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "renewal_rate", "count": "sample_count"})
    )

    agg_df["std_err"] = np.sqrt(
        np.clip(agg_df["renewal_rate"] * (1 - agg_df["renewal_rate"]), 0, None)
        / agg_df["sample_count"].clip(lower=1)
    )

    level_candidates = agg_df[level_col].unique().tolist()
    preferred_order = ["S", "A", "B", "C", "D"]
    levels = [lvl for lvl in preferred_order if lvl in level_candidates]
    levels.extend(sorted(lvl for lvl in level_candidates if lvl not in levels))
    if not levels:
        raise ValueError("未找到任何等级标签，无法绘制图表")

    level_positions = {lvl: idx for idx, lvl in enumerate(levels)}
    agg_df["x_base"] = agg_df[level_col].map(level_positions).astype(float)
    
    summary_df = (
        agg_df
        .groupby("l1_term_name")
        .apply(lambda g: g.assign(
            续报率=g["renewal_rate"].round(4),
            占比=(g["sample_count"] / g["sample_count"].sum()).round(4),
            level_tag=pd.Categorical(g["level_tag"], categories=["S", "A", "B", "C", "D"], ordered=True)
            ))
        .reset_index(drop=True)
        .pivot(index="l1_term_name", columns="level_tag", values=["续报率", "占比"])
    )
    summary_df.to_csv(output_path.with_suffix(".csv"))

    rng = np.random.default_rng(42)
    agg_df["x_jitter"] = agg_df["x_base"] + rng.normal(0, 0.08, size=len(agg_df))
    agg_df["x_jitter"] = np.clip(
        agg_df["x_jitter"], agg_df["x_base"] - 0.25, agg_df["x_base"] + 0.25
    )

    total_samples = agg_df["sample_count"].sum()
    level_summary = (
        agg_df.groupby(level_col)
        .apply(
            lambda g: pd.Series(
                {
                    "weighted_rate": np.average(
                        g["renewal_rate"], weights=g["sample_count"].clip(lower=1)
                    ),
                    "total_samples": g["sample_count"].sum(),
                }
            )
        )
        .reset_index()
    )
    level_summary["sample_ratio"] = level_summary["total_samples"] / max(total_samples, 1)
    level_summary["x"] = level_summary[level_col].map(level_positions)
    y_max_data = max(agg_df["renewal_rate"].max(), level_summary["weighted_rate"].max())
    y_upper = min(1.05, y_max_data + 0.12)
    ratio_label_y = y_upper - 0.02
    weighted_label_dx = 0.16

    fig, ax = plt.subplots(figsize=(10, 6))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4C72B0"])
    color_map = {lvl: color_cycle[idx % len(color_cycle)] for idx, lvl in enumerate(levels)}
    scatter_colors = agg_df[level_col].map(color_map)

    ax.scatter(
        agg_df["x_jitter"],
        agg_df["renewal_rate"],
        s=agg_df["sample_count"].clip(lower=20, upper=260),
        c=scatter_colors,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.6,
    )

    ax.scatter(
        level_summary["x"],
        level_summary["weighted_rate"],
        marker="D",
        s=110,
        color="#333333",
        edgecolors="white",
        linewidths=0.9,
        zorder=4,
        label="加权均值",
    )

    for _, row in level_summary.iterrows():
        ratio_text = f"占比{row['sample_ratio']:.1%}"
        weighted_text = f"{row['weighted_rate']:.1%}"
        ax.text(
            row["x"],
            ratio_label_y,
            ratio_text,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
            zorder=5,
        )
        ax.text(
            min(max(row["x"] + weighted_label_dx, row["x"] - 0.2), row["x"] + 0.3),
            row["weighted_rate"],
            weighted_text,
            ha="left",
            va="center",
            fontsize=10,
            color="#333333",
            zorder=5,
        )

    ax.set_xticks(list(level_positions.values()))
    ax.set_xticklabels(levels)
    ax.set_xlabel("等级标签")
    ax.set_ylabel("实际续报率")
    ax.set_title(f"{dataset_name} - 学期维度等级续报率")
    ax.set_ylim(0, y_upper)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.text(
        0.01,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=11,
        color="#333333",
    )

    scatter_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=8,
        markerfacecolor="#9E9E9E",
        markeredgecolor="white",
        label="等级-学期散点",
    )
    weighted_handle = Line2D(
        [],
        [],
        marker="D",
        linestyle="",
        markersize=8,
        markerfacecolor="#333333",
        markeredgecolor="white",
        label="加权均值",
    )
    ax.legend(handles=[scatter_handle, weighted_handle], loc="lower left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✓ 等级-学期散点图已保存至: {output_path}")
    return output_path


def plot_level_ratio_trend(
    df: pd.DataFrame,
    term_col: str,
    level_col: str,
    output_path: Path,
    dataset_name: str = "测试集",
) -> Path:
    """
    绘制等级占比随学期变化的折线趋势图，帮助观察结构变化。

    Args:
        df: 包含学期与等级标签的数据集。
        term_col: 学期字段名称。
        level_col: 等级标签字段名称。
        output_path: 图片输出路径。
        dataset_name: 用于标题描述的数据集名称。

    Returns:
        生成的图片路径。
    """
    _configure_chinese_font()
    required_cols = [term_col, level_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必要列: {', '.join(missing_cols)}")

    clean_df = df.dropna(subset=required_cols).copy()
    if clean_df.empty:
        raise ValueError("无法绘制等级占比趋势图：相关列均为空")

    term_order = _sort_terms(clean_df[term_col])

    agg_df = (
        clean_df.groupby([term_col, level_col])
        .size()
        .reset_index(name="sample_count")
    )
    agg_df["term_total"] = agg_df.groupby(term_col)["sample_count"].transform("sum")
    agg_df["sample_ratio"] = agg_df["sample_count"] / agg_df["term_total"].clip(lower=1)

    preferred_order = ["S", "A", "B", "C", "D"]
    level_candidates = agg_df[level_col].unique().tolist()
    levels = [lvl for lvl in preferred_order if lvl in level_candidates]
    levels.extend(sorted(lvl for lvl in level_candidates if lvl not in levels))
    if not levels:
        raise ValueError("未找到任何等级标签，无法绘制图表")

    pivot_df = (
        agg_df.pivot(index=term_col, columns=level_col, values="sample_ratio")
        .reindex(term_order)
        .fillna(0.0)
    )

    term_labels = [label for label in term_order if label in pivot_df.index]
    pivot_df = pivot_df.reindex(term_labels)
    if pivot_df.empty or not term_labels:
        raise ValueError("无法绘制等级占比趋势图：没有有效的学期与等级数据")
    x_positions = np.arange(len(term_labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4C72B0"])
    bottoms = np.zeros(len(term_labels), dtype=float)

    for idx, level in enumerate(levels):
        if level not in pivot_df.columns:
            continue
        heights = pivot_df[level].to_numpy()
        ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            width=0.98,
            color=color_cycle[idx % len(color_cycle)],
            edgecolor="white",
            label=f"{level}等级",
        )
        bottoms += heights

    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("样本占比")
    ax.set_title(f"{dataset_name} - 各等级占比百分比条形图")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(term_labels, rotation=90)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✓ 等级占比趋势图已保存至: {output_path}")
    return output_path


def plot_term_renewal_trend(
    df: pd.DataFrame,
    term_col: str,
    prob_col: str,
    label_col: str,
    output_path: Path,
    dataset_name: str = "测试集",
) -> Path:
    """
    绘制学期续报率走势与预测概率走势的相关性折线图。
    按学期聚合，计算每个学期的预测续报率（预测概率平均值）与实际续报率（实际标签平均值）。

    Args:
        df: 包含学期、预测概率和实际标签的数据集。
        term_col: 学期字段名称。
        prob_col: 预测概率字段名称。
        label_col: 实际续报标签字段名称。
        output_path: 图片输出路径。
        dataset_name: 用于标题描述的数据集名称。

    Returns:
        生成的图片路径。
    """
    _configure_chinese_font()
    required_cols = [term_col, prob_col, label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必要列: {', '.join(missing_cols)}")

    clean_df = df.dropna(subset=required_cols).copy()
    if clean_df.empty:
        raise ValueError("无法绘制学期续报率走势图：相关列均为空")

    # 确保数据类型正确
    clean_df[prob_col] = clean_df[prob_col].astype(float)
    clean_df[label_col] = clean_df[label_col].astype(float)

    # 按学期聚合，计算预测续报率和实际续报率
    agg_df = (
        clean_df.groupby(term_col)
        .agg(
            pred_renewal_rate=(prob_col, "mean"),  # 预测续报率：预测概率的平均值
            actual_renewal_rate=(label_col, "mean"),  # 实际续报率：实际标签的平均值
            sample_count=(label_col, "size"),  # 样本数量
        )
        .reset_index()
    )

    if agg_df.empty:
        raise ValueError("无法绘制学期续报率走势图：聚合后无数据")

    # 对学期进行排序
    term_order = _sort_terms(agg_df[term_col])
    agg_df["term_order"] = agg_df[term_col].map(
        {term: idx for idx, term in enumerate(term_order)}
    )
    agg_df = agg_df.sort_values("term_order")

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制折线图
    x_positions = np.arange(len(agg_df))
    ax.plot(
        x_positions,
        agg_df["pred_renewal_rate"],
        marker="o",
        linewidth=2,
        markersize=8,
        linestyle="--",  # 预测用虚线
        label="预测续报率",
        color="#4C72B0",
        alpha=0.8,
    )
    ax.plot(
        x_positions,
        agg_df["actual_renewal_rate"],
        marker="s",
        linewidth=2,
        markersize=8,
        linestyle="-",  # 实际用实线
        label="实际续报率",
        color="#55A868",
        alpha=0.8,
    )

    # 设置坐标轴
    ax.set_xticks(x_positions)
    ax.set_xticklabels(agg_df[term_col].tolist(), rotation=90, ha="center")
    ax.set_ylabel("续报率", fontsize=12)
    ax.set_xlabel("学期", fontsize=12)
    ax.set_title(f"{dataset_name} - 学期续报率走势对比", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(agg_df["pred_renewal_rate"].max(), agg_df["actual_renewal_rate"].max()) * 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=11)

    # 添加样本数量标注（可选，如果学期数量不太多）
    if len(agg_df) <= 20:
        for pos_idx, (_, row) in enumerate(agg_df.iterrows()):
            ax.text(
                pos_idx,
                max(row["pred_renewal_rate"], row["actual_renewal_rate"]) + 0.01,
                f"n={row['sample_count']}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#666666",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ 学期续报率走势图已保存至: {output_path}")
    return output_path


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
    title: str = "特征重要性分析",
) -> Path:
    """
    绘制特征重要性条形图，展示模型中最重要特征及其重要性得分。

    Args:
        feature_importance_df: 特征重要性DataFrame，需包含'feature'和'importance'列。
        output_path: 图片输出路径。
        top_n: 显示前N个重要特征，默认20。
        title: 图表标题。

    Returns:
        生成的图片路径。
    """
    _configure_chinese_font()
    
    if feature_importance_df.empty:
        raise ValueError("特征重要性数据为空，无法绘制图表")
    
    required_cols = ['feature', 'importance']
    missing_cols = [col for col in required_cols if col not in feature_importance_df.columns]
    if missing_cols:
        raise ValueError(f"特征重要性DataFrame缺少必要列: {', '.join(missing_cols)}")
    
    # 取前top_n个特征（已经是按重要性降序排序的）
    plot_df = feature_importance_df.head(top_n).copy()
    
    # 为了在水平条形图中让最重要的特征显示在顶部，需要反转顺序
    plot_df = plot_df.sort_values('importance', ascending=True)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    
    # 绘制水平条形图
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_df)))
    bars = ax.barh(
        range(len(plot_df)),
        plot_df['importance'],
        color=colors,
        edgecolor='white',
        linewidth=0.8,
    )
    
    # 设置y轴标签
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['feature'], fontsize=10)
    
    # 设置x轴
    ax.set_xlabel('重要性得分', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加数值标签
    for idx, (bar, importance) in enumerate(zip(bars, plot_df['importance'])):
        ax.text(
            importance + max(plot_df['importance']) * 0.01,
            idx,
            f'{importance:.4f}',
            va='center',
            fontsize=9,
            color='#333333',
        )
    
    # 添加网格线
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 特征重要性图已保存至: {output_path}")
    return output_path

