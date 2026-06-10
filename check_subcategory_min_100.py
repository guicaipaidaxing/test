# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from find_terms import parse_docx_category_items


DOCX_NAME = "敏感词（复筛）_去重_去包含_低俗清理_去纯英文数字.docx"
MIN_COUNT = 100


def main() -> int:
    folder = Path(__file__).resolve().parent
    src = folder / DOCX_NAME
    if not src.exists():
        print(f"未找到文件: {src}")
        return 1

    _, cat_order, _, items = parse_docx_category_items(src)

    cat_terms: dict[str, set[str]] = defaultdict(set)
    for _, cat, _, norm in items:
        cat_terms[cat].add(norm)

    print(f"检查文件: {src.name}")
    print(f"最小要求: 每个小类不少于 {MIN_COUNT} 个")
    print()

    fail: list[tuple[str, int]] = []
    for cat in cat_order:
        count = len(cat_terms.get(cat, set()))
        mark = "PASS" if count >= MIN_COUNT else "FAIL"
        print(f"{mark}\t{count}\t{cat}")
        if count < MIN_COUNT:
            fail.append((cat, count))

    print()
    print(f"小类总数: {len(cat_order)}")
    print(f"满足要求的小类数: {len(cat_order) - len(fail)}")
    print(f"不满足要求的小类数: {len(fail)}")

    if fail:
        print("\n以下小类不足 100 个：")
        for cat, count in fail:
            print(f"{count}\t{cat}")
    else:
        print("\n全部小类都满足不少于 100 个。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
