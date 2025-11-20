"""
预测概率等级标签工具

根据配置的概率区间为样本打上等级标签。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class LevelRule:
    """等级规则定义"""

    label: str
    min_prob: Optional[float] = None
    max_prob: Optional[float] = None
    min_inclusive: bool = True
    max_inclusive: bool = False

    def match(self, value: float) -> bool:
        """判断当前值是否命中规则"""
        if self.min_prob is not None:
            if value < self.min_prob or (value == self.min_prob and not self.min_inclusive):
                return False
        if self.max_prob is not None:
            if value > self.max_prob or (value == self.max_prob and not self.max_inclusive):
                return False
        return True


def parse_rules(rule_configs: Iterable[dict]) -> List[LevelRule]:
    """
    将配置字典解析为 LevelRule 对象列表。
    """
    rules: List[LevelRule] = []
    for config in rule_configs:
        rules.append(
            LevelRule(
                label=config["label"],
                min_prob=config.get("min_prob"),
                max_prob=config.get("max_prob"),
                min_inclusive=config.get("min_inclusive", True),
                max_inclusive=config.get("max_inclusive", False),
            )
        )
    return rules


def assign_level_tags(probabilities: pd.Series, rule_configs: Iterable[dict], default_label: str = "未知") -> pd.Series:
    """
    按配置规则为概率序列打标签。
    """
    rules = parse_rules(rule_configs)
    if not rules:
        raise ValueError("等级规则配置为空，无法生成level_tag")

    def _tag_value(prob: float) -> str:
        for rule in rules:
            if rule.match(prob):
                return rule.label
        return default_label

    return probabilities.apply(_tag_value).rename("level_tag")

