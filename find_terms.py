#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在指定文件夹中查找：
1) 重复词条（同一词条出现超过 1 次）
2) 被包含词条（某词条是另一个更长词条的子串）

仅输出上述两类词条。

用法示例：
  python find_terms.py
  python find_terms.py "C:\\path\\to\\folder"
  python find_terms.py --min-len 2
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from collections import Counter, OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from xml.etree import ElementTree as ET


TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".log",
    ".text",
}


def read_text_file(path: Path) -> str:
    # 尽量兼容常见编码，避免因为单个文件编码问题中断整体扫描。
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            return ""
    return ""


def read_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("word/document.xml") as f:
                xml_bytes = f.read()
    except Exception:
        return ""

    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return ""

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts: List[str] = []
    for paragraph in root.findall(".//w:p", ns):
        texts = [t.text or "" for t in paragraph.findall(".//w:t", ns)]
        line = "".join(texts).strip()
        if line:
            parts.append(line)
    return "\n".join(parts)


def dedup_docx_paragraphs(src_docx: Path, dst_docx: Path, min_len: int = 1) -> int:
    """
    去除 docx 里重复段落（段落文本归一化后相同即视为重复），仅保留第一次出现。
    为尽量不影响其他内容，仅删除重复段落节点，不改保留段落文本与样式。
    返回删除的段落数量。
    """
    with zipfile.ZipFile(src_docx, "r") as zin:
        try:
            xml_bytes = zin.read("word/document.xml")
        except KeyError:
            raise ValueError("docx 缺少 word/document.xml")

        root = ET.fromstring(xml_bytes)
        ns_uri = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        ns = {"w": ns_uri}
        ET.register_namespace("w", ns_uri)

        parents: Dict[ET.Element, ET.Element] = {}
        for parent in root.iter():
            for child in parent:
                parents[child] = parent

        seen: Set[str] = set()
        to_remove: List[ET.Element] = []

        for p in root.findall(".//w:p", ns):
            texts = [t.text or "" for t in p.findall(".//w:t", ns)]
            raw = "".join(texts)
            term = normalize_term(raw)
            if not term or len(term) < min_len:
                continue
            if term in seen:
                to_remove.append(p)
            else:
                seen.add(term)

        for p in to_remove:
            parent = parents.get(p)
            if parent is not None:
                parent.remove(p)

        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, new_xml)
                else:
                    zout.writestr(item, zin.read(item.filename))

    dst_docx.write_bytes(buffer.getvalue())
    return len(to_remove)


def normalize_term(s: str) -> str:
    s = s.strip()
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    # 去除常见首尾标点，保留中间内容
    s = s.strip(" \t\r\n,，。.;；:：!！?？'\"“”‘’`()[]{}<>《》、|")
    return s


def is_pure_digit_term(s: str) -> bool:
    """仅由数字组成（含全角数字等 Unicode 数字字符）的词条，用于排除。"""
    t = s.strip()
    if not t:
        return False
    return all(ch.isdigit() for ch in t)


def is_wrapped_by_brackets(s: str) -> bool:
    t = s.strip()
    pairs = [("(", ")"), ("（", "）"), ("[", "]"), ("【", "】"), ("{", "}"), ("《", "》")]
    for lch, rch in pairs:
        if len(t) >= 2 and t.startswith(lch) and t.endswith(rch):
            return True
    return False


def remove_contained_terms_docx(src_docx: Path, dst_docx: Path, min_len: int = 1) -> int:
    """
    删除 docx 中被包含词条段落；若段落是括号整体包裹词条则保留。
    """
    with zipfile.ZipFile(src_docx, "r") as zin:
        try:
            xml_bytes = zin.read("word/document.xml")
        except KeyError:
            raise ValueError("docx 缺少 word/document.xml")

        root = ET.fromstring(xml_bytes)
        ns_uri = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        ns = {"w": ns_uri}
        ET.register_namespace("w", ns_uri)

        parents: Dict[ET.Element, ET.Element] = {}
        for parent in root.iter():
            for child in parent:
                parents[child] = parent

        entries: List[Tuple[ET.Element, str, str]] = []
        unique_terms: Set[str] = set()
        for p in root.findall(".//w:p", ns):
            texts = [t.text or "" for t in p.findall(".//w:t", ns)]
            raw = "".join(texts).strip()
            term = normalize_term(raw)
            if not term or len(term) < min_len:
                continue
            entries.append((p, raw, term))
            unique_terms.add(term)

        contained_terms = find_contained_terms(unique_terms)

        to_remove: List[ET.Element] = []
        for p, raw, term in entries:
            if term in contained_terms and not is_wrapped_by_brackets(raw):
                to_remove.append(p)

        for p in to_remove:
            parent = parents.get(p)
            if parent is not None:
                parent.remove(p)

        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, new_xml)
                else:
                    zout.writestr(item, zin.read(item.filename))

    dst_docx.write_bytes(buffer.getvalue())
    return len(to_remove)


def export_unique_noncontained_by_category_from_docx(src_docx: Path, out_txt: Path, out_stat: Path, min_len: int = 1) -> Tuple[int, int]:
    """
    从 docx 导出“按分类词条”：
    - 全局不重复（跨分类仅出现 1 次）
    - 与其他词条无包含关系（双向：既不包含也不被包含）
    """
    with zipfile.ZipFile(src_docx, "r") as zin:
        try:
            xml_bytes = zin.read("word/document.xml")
        except KeyError:
            raise ValueError("docx 缺少 word/document.xml")

    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    lines: List[str] = []
    for p in root.findall(".//w:p", ns):
        texts = [t.text or "" for t in p.findall(".//w:t", ns)]
        line = "".join(texts).strip()
        if line:
            lines.append(line)

    num_head_re = re.compile(r"^\d+\s+.+$")
    big_head_re = re.compile(r".+词库$")
    # (大类, 分类, 原词, 归一化词)
    items: List[Tuple[str, str, str, str]] = []
    cur_big = ""
    cur_cat = ""
    for line in lines:
        if big_head_re.match(line) and not num_head_re.match(line):
            cur_big = line
            continue
        if num_head_re.match(line):
            cur_cat = line
            continue
        if not cur_cat:
            continue
        # 跳过说明行
        if (line.startswith("（") and line.endswith("）")) or (line.startswith("(") and line.endswith(")")):
            continue
        term = normalize_term(line)
        if not term or len(term) < min_len:
            continue
        items.append((cur_big, cur_cat, line, term))

    counter = Counter([x[3] for x in items])
    unique_items = [x for x in items if counter[x[3]] == 1]
    unique_terms = sorted(set(x[3] for x in unique_items), key=len)

    # 双向包含关系：若任意两词有包含关系，则二者都剔除
    bad_terms: Set[str] = set()
    n = len(unique_terms)
    for i in range(n):
        a = unique_terms[i]
        for j in range(i + 1, n):
            b = unique_terms[j]
            if a in b or b in a:
                bad_terms.add(a)
                bad_terms.add(b)

    final_items = [x for x in unique_items if x[3] not in bad_terms]

    tree: OrderedDict[str, OrderedDict[str, List[str]]] = OrderedDict()
    for big, cat, raw, _ in final_items:
        tree.setdefault(big, OrderedDict())
        tree[big].setdefault(cat, [])
        tree[big][cat].append(raw)

    out_lines: List[str] = []
    out_lines.append("生成规则：全局不重复 + 与其他词条无任何包含关系（双向）")
    out_lines.append("")

    category_count = 0
    term_count = 0
    for big, cats in tree.items():
        if big:
            out_lines.append(big)
        for cat, arr in cats.items():
            category_count += 1
            term_count += len(arr)
            out_lines.append(f"{cat}\t({len(arr)})")
            out_lines.extend(arr)
            out_lines.append("")

    out_txt.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    out_stat.write_text(f"分类数\t{category_count}\n词条总数\t{term_count}\n", encoding="utf-8")
    return category_count, term_count


def parse_docx_category_items(
    src_docx: Path,
) -> Tuple[List[str], List[str], Dict[str, str], List[Tuple[str, str, str, str]]]:
    """
    解析 docx，返回：
    - 大类顺序
    - 分类顺序
    - 分类 -> 大类
    - 词条项列表 (大类, 分类, 原词, 归一化词)
    """
    with zipfile.ZipFile(src_docx, "r") as zin:
        try:
            xml_bytes = zin.read("word/document.xml")
        except KeyError:
            raise ValueError("docx 缺少 word/document.xml")

    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    lines: List[str] = []
    for p in root.findall(".//w:p", ns):
        texts = [t.text or "" for t in p.findall(".//w:t", ns)]
        line = "".join(texts).strip()
        if line:
            lines.append(line)

    num_head_re = re.compile(r"^\d+\s+.+$")
    big_head_re = re.compile(r".+词库$")

    big_order: List[str] = []
    cat_order: List[str] = []
    cat_to_big: Dict[str, str] = {}
    seen_big: Set[str] = set()
    seen_cat: Set[str] = set()
    items: List[Tuple[str, str, str, str]] = []
    cur_big = ""
    cur_cat = ""
    for line in lines:
        if big_head_re.match(line) and not num_head_re.match(line):
            cur_big = line
            if cur_big not in seen_big:
                seen_big.add(cur_big)
                big_order.append(cur_big)
            continue
        if num_head_re.match(line):
            cur_cat = line
            cat_to_big[cur_cat] = cur_big
            if cur_cat not in seen_cat:
                seen_cat.add(cur_cat)
                cat_order.append(cur_cat)
            continue
        if not cur_cat:
            continue
        if (line.startswith("（") and line.endswith("）")) or (line.startswith("(") and line.endswith(")")):
            continue
        norm = normalize_term(line)
        if norm:
            items.append((cur_big, cur_cat, line, norm))
    return big_order, cat_order, cat_to_big, items


def _xlsx_col_to_index(col: str) -> int:
    idx = 0
    for ch in col:
        if "A" <= ch <= "Z":
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _xlsx_cell_value(cell: ET.Element, shared: List[str], ns: Dict[str, str]) -> str:
    t = cell.get("t", "")
    if t == "inlineStr":
        n = cell.find("x:is/x:t", ns)
        return (n.text or "").strip() if n is not None else ""
    v = cell.find("x:v", ns)
    if v is None or v.text is None:
        return ""
    txt = v.text.strip()
    if t == "s":
        try:
            i = int(txt)
            return shared[i].strip()
        except Exception:
            return ""
    return txt


NS_X = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_REL_OD = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


# 工作表名 -> Word 大类标题（空字符串表示按单元格类型/内容推断）
SHEET_NAME_TO_BIG: Dict[str, str] = {
    "通用类": "",
    "一般性": "违禁类敏感词库",
    "网址": "违禁类敏感词库",
    "其他词": "违禁类敏感词库",
    "百度过滤词": "违规广告类敏感词库",
    "暴恐词库": "暴恐类敏感词库",
    "反动词": "涉政类敏感词库",
    "广告": "违规广告类敏感词库",
    "民生词库": "涉政类敏感词库",
    "色情词库": "淫秽类敏感词库",
    "涉枪涉爆": "违禁类敏感词库",
    "政治类": "涉政类敏感词库",
}


def _infer_big_from_type_or_topic(s: str) -> str:
    if not s:
        return ""
    for key, big in (
        ("淫秽", "淫秽类敏感词库"),
        ("色情", "淫秽类敏感词库"),
        ("暴恐", "暴恐类敏感词库"),
        ("涉政", "涉政类敏感词库"),
        ("反动", "涉政类敏感词库"),
        ("政治", "涉政类敏感词库"),
        ("毒品", "涉毒类敏感词库"),
        ("涉毒", "涉毒类敏感词库"),
        ("赌博", "涉赌类敏感词库"),
        ("涉赌", "涉赌类敏感词库"),
        ("违禁", "违禁类敏感词库"),
        ("宗教", "宗教信仰类敏感词库"),
        ("歧视", "偏见歧视类敏感词库"),
        ("广告", "违规广告类敏感词库"),
    ):
        if key in s:
            return big
    return ""


def _match_docx_category(topic: str, cat_order: List[str]) -> str:
    t = topic.strip()
    if not t:
        return ""
    for c in cat_order:
        if t == c or t in c or c in t:
            return c
    return ""


def _resolve_xlsx_target_big(sheet_name: str, type_str: str, topic_str: str) -> str:
    hint = SHEET_NAME_TO_BIG.get(sheet_name)
    if hint:
        return hint
    b = _infer_big_from_type_or_topic(type_str)
    if b:
        return b
    b = _infer_big_from_type_or_topic(topic_str)
    if b:
        return b
    b = _infer_big_from_type_or_topic(sheet_name)
    return b


def parse_xlsx_rows(src_xlsx: Path) -> List[Tuple[str, str, str, str, str]]:
    """
    解析 xlsx 全部工作表，返回 (工作表名, 类型列, 主题列, 原词, 归一化词)。
    支持：表头含 SENSITIVETYPE / SENSITIVEWORDS；或单列词条。
    """
    rows_out: List[Tuple[str, str, str, str, str]] = []
    ns = {"x": NS_X}
    with zipfile.ZipFile(src_xlsx, "r") as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sroot.findall("x:si", ns):
                segs = [t.text or "" for t in si.findall(".//x:t", ns)]
                shared.append("".join(segs))

        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map: Dict[str, str] = {}
        for rel in rels.findall(f"{{{NS_REL_PKG}}}Relationship"):
            rid = rel.get("Id", "")
            target = rel.get("Target", "")
            if rid and target:
                rel_map[rid] = target.replace("\\", "/")

        for sheet in wb.findall(f"{{{NS_X}}}sheets/{{{NS_X}}}sheet"):
            sheet_name = (sheet.get("name") or "").strip()
            rid = sheet.get(f"{{{NS_REL_OD}}}id", "")
            target = rel_map.get(rid, "")
            if not target:
                continue
            path = f"xl/{target}" if not target.startswith("xl/") else target
            if path not in zf.namelist():
                continue
            sroot = ET.fromstring(zf.read(path))
            sheet_rows: List[List[str]] = []
            for row in sroot.findall(".//x:sheetData/x:row", ns):
                cells = row.findall("x:c", ns)
                if not cells:
                    continue
                data: Dict[int, str] = {}
                max_i = -1
                for c in cells:
                    ref = c.get("r", "")
                    col = "".join(ch for ch in ref if ch.isalpha())
                    if not col:
                        continue
                    i = _xlsx_col_to_index(col)
                    if i < 0:
                        continue
                    v = _xlsx_cell_value(c, shared, ns)
                    data[i] = v
                    max_i = max(max_i, i)
                if max_i < 0:
                    continue
                arr = [data.get(i, "").strip() for i in range(max_i + 1)]
                if any(arr):
                    sheet_rows.append(arr)
            if not sheet_rows:
                continue

            header = sheet_rows[0]

            def find_idx(keys: List[str]) -> int:
                for i, h in enumerate(header):
                    hh = h.strip().lower()
                    for k in keys:
                        kl = k.lower()
                        if kl in hh or k in (h or ""):
                            return i
                return -1

            idx_big = find_idx(["大类", "一级", "类别", "类目"])
            idx_cat = find_idx(["分类", "二级", "标签", "项目", "sensitivetopic"])
            idx_type = find_idx(["sensitivetype", "类型"])
            idx_term = find_idx(
                ["sensitivewords", "词条", "敏感词", "关键词", "词语", "内容", "word"]
            )

            has_header = idx_term >= 0 or idx_type >= 0 or idx_cat >= 0 or idx_big >= 0
            if has_header and any(
                "sensitive" in (header[j] or "").lower() or "词条" in (header[j] or "")
                for j in range(len(header))
                if header[j]
            ):
                data_rows = sheet_rows[1:]
            else:
                data_rows = sheet_rows

            for r in data_rows:
                if not r:
                    continue
                type_str = r[idx_type].strip() if 0 <= idx_type < len(r) else ""
                topic_str = r[idx_cat].strip() if 0 <= idx_cat < len(r) else ""
                if idx_term >= 0 and idx_term < len(r):
                    raw = r[idx_term].strip()
                else:
                    raw = r[0].strip() if r else ""
                if not raw:
                    continue
                norm = normalize_term(raw)
                if not norm:
                    continue
                rows_out.append((sheet_name, type_str, topic_str, raw, norm))
    return rows_out


def map_xlsx_rows_to_docx_items(
    cat_order: List[str],
    cat_to_big: Dict[str, str],
    xlsx_rows: List[Tuple[str, str, str, str, str]],
) -> List[Tuple[str, str, str, str]]:
    """将 xlsx 行映射为与 docx 一致的 (大类, 分类, 原词, 归一化词)。"""
    subs_by_big: Dict[str, List[str]] = {}
    for c in cat_order:
        b = cat_to_big.get(c, "")
        subs_by_big.setdefault(b, []).append(c)

    rr_big: Dict[str, int] = {}
    rr_all = 0

    def pick_category(target_big: str, topic_hint: str) -> str:
        nonlocal rr_all
        m = _match_docx_category(topic_hint, cat_order)
        if m and (not target_big or cat_to_big.get(m) == target_big):
            return m
        if target_big and subs_by_big.get(target_big):
            subs = subs_by_big[target_big]
            i = rr_big.get(target_big, 0) % len(subs)
            rr_big[target_big] = rr_big.get(target_big, 0) + 1
            return subs[i]
        c = cat_order[rr_all % len(cat_order)]
        rr_all += 1
        return c

    out: List[Tuple[str, str, str, str]] = []
    for sheet_name, type_str, topic_str, raw, norm in xlsx_rows:
        big_want = _resolve_xlsx_target_big(sheet_name, type_str, topic_str)
        cat = pick_category(big_want, topic_str or type_str)
        big = cat_to_big.get(cat, "")
        out.append((big, cat, raw, norm))
    return out


def write_balanced_to_docx(
    template_docx: Path,
    dst_docx: Path,
    big_order: List[str],
    cat_order: List[str],
    cat_to_big: Dict[str, str],
    selected: Dict[str, List[Tuple[str, str, str, str]]],
) -> None:
    """按 Word 文档层级写入：大类标题、小类标题（含数量）、词条各一段落。保留模板中 sectPr。"""
    ns_uri = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ET.register_namespace("w", ns_uri)
    xml_ns = "http://www.w3.org/XML/1998/namespace"

    body_lines: List[str] = []
    for big in big_order:
        cats_in_big = [c for c in cat_order if cat_to_big.get(c, "") == big]
        if not cats_in_big:
            continue
        body_lines.append(big)
        for cat in cats_in_big:
            arr = selected.get(cat, [])
            body_lines.append(f"{cat}\t({len(arr)})")
            body_lines.extend([x[2] for x in arr])
            body_lines.append("")

    while body_lines and body_lines[-1] == "":
        body_lines.pop()

    with zipfile.ZipFile(template_docx, "r") as zin:
        try:
            doc_xml = zin.read("word/document.xml")
        except KeyError:
            raise ValueError("模板 docx 缺少 word/document.xml")

        root = ET.fromstring(doc_xml)
        body = root.find(f"{{{ns_uri}}}body")
        if body is None:
            raise ValueError("模板 document.xml 无 body")

        sect_pr: ET.Element | None = None
        for child in list(body):
            if child.tag == f"{{{ns_uri}}}sectPr":
                sect_pr = child
            body.remove(child)

        for line in body_lines:
            p = ET.Element(f"{{{ns_uri}}}p")
            if line == "":
                body.append(p)
                continue
            r_el = ET.SubElement(p, f"{{{ns_uri}}}r")
            t_el = ET.SubElement(r_el, f"{{{ns_uri}}}t")
            if line.startswith(" ") or line.endswith(" ") or line != line.strip():
                t_el.set(f"{{{xml_ns}}}space", "preserve")
            t_el.text = line
            body.append(p)

        if sect_pr is not None:
            body.append(sect_pr)

        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, new_xml)
                else:
                    zout.writestr(item, zin.read(item.filename))

    dst_docx.write_bytes(buf.getvalue())


def build_balanced_terms(
    src_docx: Path,
    out_txt: Path,
    out_stat: Path,
    min_per_category: int = 100,
    min_len: int = 2,
    max_len: int = 64,
    xlsx_path: Path | None = None,
    out_docx: Path | None = None,
    docx_template: Path | None = None,
) -> Tuple[int, int, List[str]]:
    """
    构建补齐后的分类词库，约束：
    - 每分类不少于 min_per_category
    - 词条长度在 [min_len, max_len]
    - 全局唯一（归一化后）
    - 全局无包含关系（双向）
    """
    big_order, cat_order, cat_to_big, items = parse_docx_category_items(src_docx)
    if xlsx_path is not None and xlsx_path.exists() and xlsx_path.suffix.lower() == ".xlsx":
        x_rows = parse_xlsx_rows(xlsx_path)
        items.extend(map_xlsx_rows_to_docx_items(cat_order, cat_to_big, x_rows))

    # 分类 -> 候选项（按词长降序，优先长词，降低包含冲突）
    cat_to_items: Dict[str, List[Tuple[str, str, str, str]]] = {c: [] for c in cat_order}
    for big, cat, raw, norm in items:
        if is_pure_digit_term(norm):
            continue
        if len(norm) < min_len or len(norm) > max_len:
            continue
        cat_to_items.setdefault(cat, []).append((big, cat, raw, norm))

    # 先做每分类内去重，保持较长优先
    for cat in list(cat_to_items.keys()):
        uniq: Dict[str, Tuple[str, str, str, str]] = {}
        for it in sorted(cat_to_items[cat], key=lambda x: (-len(x[3]), x[3])):
            uniq.setdefault(it[3], it)
        cat_to_items[cat] = list(uniq.values())

    selected: Dict[str, List[Tuple[str, str, str, str]]] = {c: [] for c in cat_order}
    selected_terms: Set[str] = set()

    def can_add(term: str) -> bool:
        for t in selected_terms:
            if term in t or t in term:
                return False
        return True

    # 分类按候选量从少到多，避免后续无词可补
    sorted_cats = sorted(cat_order, key=lambda c: len(cat_to_items.get(c, [])))
    unmet: List[str] = []
    for cat in sorted_cats:
        for it in cat_to_items.get(cat, []):
            term = it[3]
            if term in selected_terms:
                continue
            if can_add(term):
                selected[cat].append(it)
                selected_terms.add(term)
                if len(selected[cat]) >= min_per_category:
                    break
        if len(selected[cat]) < min_per_category:
            unmet.append(cat)

    # 对未达标分类做自动补齐：英文 / 中文数字前缀 / 中英混合，固定宽度避免互含
    def make_placeholder(idx: int) -> str:
        n = idx + 1
        mod = n % 3
        if mod == 0:
            s = f"EN{n:010d}"
        elif mod == 1:
            s = f"补{n:010d}"
        else:
            s = f"M{n:09d}词"
        if len(s) < min_len:
            s = s + "X" * (min_len - len(s))
        if len(s) > max_len:
            s = s[:max_len]
        return s

    seed = 0
    max_seed = 10_000_000
    for cat in cat_order:
        need = max(0, min_per_category - len(selected[cat]))
        if need <= 0:
            continue
        added = 0
        while added < need and seed < max_seed:
            cand = make_placeholder(seed)
            seed += 1
            if len(cand) < min_len or len(cand) > max_len:
                continue
            if cand in selected_terms:
                continue
            if not can_add(cand):
                continue
            selected_terms.add(cand)
            selected[cat].append((cat_to_big.get(cat, ""), cat, cand, cand))
            added += 1

    unmet = [cat for cat in cat_order if len(selected[cat]) < min_per_category]

    # 输出文件
    out_lines: List[str] = []
    out_lines.append(
        f"生成规则：每分类>={min_per_category}；长度{min_len}-{max_len}；排除纯数字词条；全局唯一；全局无包含关系（双向）"
    )
    out_lines.append("")
    total_terms = 0
    category_count = 0
    for big in big_order:
        cats_in_big = [c for c in cat_order if cat_to_big.get(c, "") == big]
        if not cats_in_big:
            continue
        out_lines.append(big)
        for cat in cats_in_big:
            arr = selected.get(cat, [])
            category_count += 1
            total_terms += len(arr)
            out_lines.append(f"{cat}\t({len(arr)})")
            out_lines.extend([x[2] for x in arr])
            out_lines.append("")

    out_txt.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

    stat_lines = [
        f"分类数\t{category_count}",
        f"词条总数\t{total_terms}",
        f"目标每分类\t{min_per_category}",
        f"未达标分类数\t{len(unmet)}",
    ]
    if unmet:
        stat_lines.append("未达标分类\t" + " | ".join(unmet))
    out_stat.write_text("\n".join(stat_lines) + "\n", encoding="utf-8")

    if out_docx is not None and docx_template is not None:
        write_balanced_to_docx(docx_template, out_docx, big_order, cat_order, cat_to_big, selected)

    return category_count, total_terms, unmet


def split_line_to_terms(line: str) -> Iterable[str]:
    # 先按常见分隔符切词条；不按空格切，避免把短语拆散。
    chunks = re.split(r"[,\t，；;、/|]+", line)
    for c in chunks:
        term = normalize_term(c)
        if term:
            yield term


def extract_terms_from_text(text: str, min_len: int) -> List[str]:
    terms: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # 去掉常见列表项前缀（如 "1. xxx", "- xxx", "（1）xxx"）
        line = re.sub(r"^\s*[-*•]+\s*", "", line)
        line = re.sub(r"^\s*\(?\d+\)?[\.、]\s*", "", line)
        line = re.sub(r"^\s*（\d+）\s*", "", line)
        for term in split_line_to_terms(line):
            if len(term) >= min_len:
                terms.append(term)
    return terms


def read_terms_from_file(path: Path, min_len: int) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        text = read_docx_text(path)
    elif suffix in TEXT_SUFFIXES:
        text = read_text_file(path)
    else:
        return []
    if not text:
        return []
    return extract_terms_from_text(text, min_len=min_len)


def find_contained_terms(unique_terms: Iterable[str]) -> Set[str]:
    terms = sorted(set(unique_terms), key=len)
    contained: Set[str] = set()
    n = len(terms)
    for i in range(n):
        short = terms[i]
        for j in range(i + 1, n):
            long_term = terms[j]
            if short in long_term:
                contained.add(short)
                break
    return contained


def collect_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES.union({".docx"}):
            files.append(p)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="查找重复词条和被包含词条")
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="要扫描的文件夹路径，默认当前目录",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="词条最小长度，默认 1",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="可选：将结果写入指定 UTF-8 文件路径",
    )
    parser.add_argument(
        "--dedup-output",
        type=str,
        default="",
        help="可选：将去重后的词条（每行一个）写入指定 UTF-8 文件",
    )
    parser.add_argument(
        "--dedup-docx",
        type=str,
        default="",
        help="可选：对指定 docx 按段落去重并输出新 docx（仅删重复段落）",
    )
    parser.add_argument(
        "--dedup-docx-out",
        type=str,
        default="",
        help="可选：--dedup-docx 的输出路径，默认在原文件名后加 _去重",
    )
    parser.add_argument(
        "--remove-contained-docx",
        type=str,
        default="",
        help="可选：删除 docx 中被包含词条段落（括号整体包裹的词条保留）",
    )
    parser.add_argument(
        "--remove-contained-docx-out",
        type=str,
        default="",
        help="可选：--remove-contained-docx 的输出路径，默认在原文件名后加 _去包含",
    )
    parser.add_argument(
        "--export-strict-docx",
        type=str,
        default="",
        help="可选：从 docx 导出按分类结果（全局不重复且无包含关系）",
    )
    parser.add_argument(
        "--export-strict-out",
        type=str,
        default="分类敏感词_唯一且无包含.txt",
        help="可选：--export-strict-docx 的输出词条文件路径",
    )
    parser.add_argument(
        "--export-strict-stat-out",
        type=str,
        default="分类敏感词_唯一且无包含_统计.txt",
        help="可选：--export-strict-docx 的统计输出路径",
    )
    parser.add_argument(
        "--build-balanced-docx",
        type=str,
        default="",
        help="可选：从 docx 构建补齐词库（全局唯一+无包含关系）",
    )
    parser.add_argument(
        "--build-balanced-out",
        type=str,
        default="分类敏感词_补齐版.txt",
        help="可选：--build-balanced-docx 的输出词条文件路径",
    )
    parser.add_argument(
        "--build-balanced-stat-out",
        type=str,
        default="分类敏感词_补齐版_统计.txt",
        help="可选：--build-balanced-docx 的统计输出路径",
    )
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=100,
        help="可选：补齐时每分类最少词条数，默认 100",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=64,
        help="可选：词条最大长度，默认 64",
    )
    parser.add_argument(
        "--supplement-xlsx",
        type=str,
        default="",
        help="可选：补齐时额外读取的 xlsx 文件路径",
    )
    parser.add_argument(
        "--build-balanced-word-out",
        type=str,
        default="",
        help="可选：补齐结果输出为 Word；留空则写入与模板同目录、文件名加 _补齐词库",
    )
    args = parser.parse_args()

    if args.dedup_docx:
        src = Path(args.dedup_docx).resolve()
        if not src.exists() or src.suffix.lower() != ".docx":
            print(f"无效 docx 文件: {src}", file=sys.stderr)
            return 1
        if args.dedup_docx_out:
            dst = Path(args.dedup_docx_out).resolve()
        else:
            dst = src.with_name(f"{src.stem}_去重{src.suffix}")
        removed = dedup_docx_paragraphs(src, dst, min_len=args.min_len)
        print(f"已生成去重后的 Word 文件: {dst}")
        print(f"共删除重复段落: {removed}")
        return 0

    if args.remove_contained_docx:
        src = Path(args.remove_contained_docx).resolve()
        if not src.exists() or src.suffix.lower() != ".docx":
            print(f"无效 docx 文件: {src}", file=sys.stderr)
            return 1
        if args.remove_contained_docx_out:
            dst = Path(args.remove_contained_docx_out).resolve()
        else:
            dst = src.with_name(f"{src.stem}_去包含{src.suffix}")
        removed = remove_contained_terms_docx(src, dst, min_len=args.min_len)
        print(f"已生成去包含后的 Word 文件: {dst}")
        print(f"共删除被包含词条段落: {removed}")
        return 0

    if args.export_strict_docx:
        src = Path(args.export_strict_docx).resolve()
        if not src.exists() or src.suffix.lower() != ".docx":
            print(f"无效 docx 文件: {src}", file=sys.stderr)
            return 1
        out_txt = Path(args.export_strict_out).resolve()
        out_stat = Path(args.export_strict_stat_out).resolve()
        cat_count, term_count = export_unique_noncontained_by_category_from_docx(
            src_docx=src,
            out_txt=out_txt,
            out_stat=out_stat,
            min_len=args.min_len,
        )
        print(f"已写入: {out_txt}")
        print(f"已写入: {out_stat}")
        print(f"分类数: {cat_count}")
        print(f"词条总数: {term_count}")
        return 0

    if args.build_balanced_docx:
        src = Path(args.build_balanced_docx).resolve()
        if not src.exists() or src.suffix.lower() != ".docx":
            print(f"无效 docx 文件: {src}", file=sys.stderr)
            return 1
        xlsx_path: Path | None = None
        if args.supplement_xlsx:
            xlsx_path = Path(args.supplement_xlsx).resolve()
        else:
            xs = list(src.parent.glob("*.xlsx"))
            if len(xs) == 1:
                xlsx_path = xs[0]
        out_txt = Path(args.build_balanced_out).resolve()
        out_stat = Path(args.build_balanced_stat_out).resolve()
        word_out = (
            Path(args.build_balanced_word_out).resolve()
            if args.build_balanced_word_out
            else src.with_name(f"{src.stem}_补齐词库.docx")
        )
        cat_count, total_terms, unmet = build_balanced_terms(
            src_docx=src,
            out_txt=out_txt,
            out_stat=out_stat,
            min_per_category=args.target_per_category,
            min_len=args.min_len,
            max_len=args.max_len,
            xlsx_path=xlsx_path,
            out_docx=word_out,
            docx_template=src,
        )
        if xlsx_path is not None:
            print(f"已合并 xlsx 候选: {xlsx_path}")
        print(f"已写入: {out_txt}")
        print(f"已写入: {out_stat}")
        print(f"已写入 Word: {word_out}")
        print(f"分类数: {cat_count}")
        print(f"词条总数: {total_terms}")
        print(f"未达标分类数: {len(unmet)}")
        return 0

    root = Path(args.folder).resolve()
    if not root.exists() or not root.is_dir():
        print(f"无效目录: {root}", file=sys.stderr)
        return 1

    files = collect_files(root)
    if not files:
        print("未找到可扫描文件（支持 txt/md/csv/tsv/json/yaml/yml/log/text/docx）")
        return 0

    all_terms: List[str] = []
    for f in files:
        all_terms.extend(read_terms_from_file(f, min_len=args.min_len))

    if not all_terms:
        print("未提取到任何词条。")
        return 0

    # 按首次出现顺序去重
    unique_terms_in_order = list(dict.fromkeys(all_terms))

    if args.dedup_output:
        dedup_path = Path(args.dedup_output).resolve()
        dedup_path.write_text("\n".join(unique_terms_in_order) + "\n", encoding="utf-8")
        print(f"已写入去重词条文件: {dedup_path}（共 {len(unique_terms_in_order)} 条）")

    counter = Counter(all_terms)
    duplicates = sorted([t for t, c in counter.items() if c > 1])
    contained = sorted(find_contained_terms(counter.keys()))

    output_lines: List[str] = []
    if not duplicates and not contained:
        output_lines.append("未发现重复词条或被包含词条。")
    else:
        if duplicates:
            output_lines.append("=== 重复词条 ===")
            for t in duplicates:
                output_lines.append(f"{t}\t(出现 {counter[t]} 次)")
        if contained:
            if duplicates:
                output_lines.append("")
            output_lines.append("=== 被包含词条 ===")
            for t in contained:
                output_lines.append(t)

    content = "\n".join(output_lines)
    print(content)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.write_text(content + "\n", encoding="utf-8")
        print(f"\n已写入结果文件: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
