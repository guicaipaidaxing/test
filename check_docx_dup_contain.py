# -*- coding: utf-8 -*-
"""分析「敏感词（复筛）_去重_去包含_低俗清理.docx」中的重复词条与包含关系，并输出 UTF-8 报告。"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from find_terms import parse_docx_category_items


def main() -> int:
    folder = Path(__file__).resolve().parent
    target = folder / "敏感词（复筛）_去重_去包含_低俗清理.docx"
    out_file = folder / "check_result_utf8.txt"

    if not target.exists():
        print(f"未找到目标文件: {target.name}")
        print("当前目录下的 docx：")
        for p in sorted(folder.glob("*.docx")):
            print("  ", p.name)
        return 1

    lines: list[str] = []

    lines.append(f"分析文件: {target.name}")
    lines.append("")

    _, _, _, items = parse_docx_category_items(target)
    norms = [x[3] for x in items]
    counter = Counter(norms)

    by_norm: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for big, cat, raw, norm in items:
        by_norm[norm].append((big, cat, raw))

    dups = {k: v for k, v in counter.items() if v > 1}

    unique_sorted = sorted(counter.keys(), key=len)
    n = len(unique_sorted)
    pairs: list[tuple[str, str]] = []
    for i in range(n):
        a = unique_sorted[i]
        for j in range(i + 1, n):
            b = unique_sorted[j]
            if a in b:
                pairs.append((a, b))

    lines.append("=== 统计 ===")
    lines.append(f"词条条目总数（段落词条）: {len(items)}")
    lines.append(f"归一化后不同词条数: {len(counter)}")
    lines.append(f"重复出现的归一化词条数: {len(dups)}")
    lines.append(f"存在子串包含关系的词对数: {len(pairs)}")
    lines.append("")

    if dups:
        lines.append("=== 重复词条（归一化后相同，出现次数>1）===")
        for term in sorted(dups.keys()):
            lines.append("")
            lines.append(f"「{term}」出现 {counter[term]} 次：")
            for big, cat, raw in by_norm[term]:
                lines.append(f"  - [{big}] / [{cat}]")
                lines.append(f"    原文: {raw}")
        lines.append("")
    else:
        lines.append("=== 重复词条 ===")
        lines.append("无。")
        lines.append("")

    if pairs:
        lines.append("=== 包含关系（较短词是较长词的子串；仅列出词对）===")
        for short, long in pairs:
            lines.append(f"  短「{short}」 是 长「{long}」 的子串")
        lines.append("")
    else:
        lines.append("=== 包含关系 ===")
        lines.append("无。")
        lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")

    print("分析完成。")
    print(f"已写入 UTF-8 结果文件: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
