# -*- coding: utf-8 -*-
import os
import re
import sys
import json
import time
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer


# =========================
# 0) 설정: 여기만 고치면 됨
# =========================
BASE_DIR = r"C:\삼성증권\POPHTSN\data\finddata"
TREECOMMON_NAME = "treecommon.xml"
MAP_DIR_NAME = "map"

EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 임베딩 캐시
EMB_ITEMS_PATH = os.path.join(BASE_DIR, "emb_items.pkl")
EMB_VECTORS_PATH = os.path.join(BASE_DIR, "emb_vectors.npy")

# 컴파일 산출물(첫 실행에서만 XML 읽고 생성)
COMPILED_ITEMS_PATH = os.path.join(BASE_DIR, "compiled_items.pkl")
COMPILED_TEXTS_PATH = os.path.join(BASE_DIR, "compiled_texts.pkl")
COMPILED_MANIFEST_PATH = os.path.join(BASE_DIR, "compiled_manifest.json")

# 임베딩 캐시 메타(컴파일 결과가 바뀌면 emb 캐시 무효화)
EMB_META_PATH = os.path.join(BASE_DIR, "emb_meta.json")

# 학습(리뷰) 저장 파일(JSON)
LEARN_PATH = os.path.join(BASE_DIR, "learn.json")
LEARN_ALPHA = 0.20

def set_base_dir(base_dir: str):
    """
    서버에서 base_dir 주입 시, 경로 상수들을 함께 갱신한다.
    (원래 코드는 import 시점의 BASE_DIR로 경로가 고정되므로 필요)
    """
    global BASE_DIR
    global EMB_ITEMS_PATH, EMB_VECTORS_PATH
    global COMPILED_ITEMS_PATH, COMPILED_TEXTS_PATH, COMPILED_MANIFEST_PATH
    global EMB_META_PATH, LEARN_PATH

    BASE_DIR = base_dir

    EMB_ITEMS_PATH = os.path.join(BASE_DIR, "emb_items.pkl")
    EMB_VECTORS_PATH = os.path.join(BASE_DIR, "emb_vectors.npy")

    COMPILED_ITEMS_PATH = os.path.join(BASE_DIR, "compiled_items.pkl")
    COMPILED_TEXTS_PATH = os.path.join(BASE_DIR, "compiled_texts.pkl")
    COMPILED_MANIFEST_PATH = os.path.join(BASE_DIR, "compiled_manifest.json")

    EMB_META_PATH = os.path.join(BASE_DIR, "emb_meta.json")

    LEARN_PATH = os.path.join(BASE_DIR, "learn.json")

# =========================
# 1) 유틸
# =========================
def tokenize_ko(s: str) -> List[str]:
    s = s.lower()
    toks = re.findall(r"[0-9]+|[a-zA-Z]+|[가-힣]+", s)
    return [t for t in toks if len(t) >= 2]


def now_ms() -> int:
    return int(time.time() * 1000)


def load_json_if_exists(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def tokenize_rules(s: str) -> List[str]:
    """
    룰 매칭용 토큰화:
    - 한글/영문/숫자만 뽑고
    - 길이 2 이상만
    """
    if not s:
        return []
    s = s.lower()
    s = s.replace("p/e", "pe").replace("pbr", "pbr").replace("per", "per")
    toks = re.findall(r"[0-9]+|[a-zA-Z]+|[가-힣]+", s)
    return [t for t in toks if len(t) >= 2]



def build_query_token_set(query: str) -> set:
    """
    쿼리를 토큰 set으로 만들어 룰 매칭에 사용
    """
    return set(tokenize_rules(query))


# =========================
# 2) XML 읽기: 인코딩 이슈 회피 + 로그
# =========================
def parse_xml_safely(path: str, log: bool = False) -> ET.Element:
    with open(path, "rb") as f:
        data = f.read()

    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            text = data.decode(enc)
            root = ET.fromstring(text)
            if log:
                print(f"[XML] loaded: {os.path.basename(path)} bytes={len(data)} enc={enc} root=<{root.tag}>")
            return root
        except Exception:
            pass

    text = data.decode("utf-8", errors="replace")
    root = ET.fromstring(text)
    if log:
        print(f"[XML] loaded: {os.path.basename(path)} bytes={len(data)} enc=utf-8(replace) root=<{root.tag}>")
    return root


# =========================
# 3) 데이터 구조
# =========================
@dataclass
class CondItem:
    code: str
    name: str
    file_name: str
    path: str = ""


@dataclass
class Control:
    caption: str = ""


@dataclass
class MapDetail:
    complete_index_name: str = ""
    map_name: str = ""
    controls: List[Control] = None

    def __post_init__(self):
        if self.controls is None:
            self.controls = []


@dataclass
class CompiledItem:
    code: str
    file_name: str
    name: str
    path: str
    display_name: str
    search_text: str
    controls: List[str]
    has_numeric_control: bool


# =========================
# 4) treecommon/map 파서
# =========================
class MapIndex:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.treecommon_path = os.path.join(base_dir, TREECOMMON_NAME)
        self.map_dir = os.path.join(base_dir, MAP_DIR_NAME)

        if not os.path.exists(self.treecommon_path):
            raise FileNotFoundError(f"treecommon.xml not found: {self.treecommon_path}")

        self.cond_items: List[CondItem] = self._load_treecommon()
        self._map_cache: Dict[str, Dict[str, MapDetail]] = {}

    def _load_treecommon(self) -> List[CondItem]:
        """
        treecommon.xml 파싱:
        - attrib 키를 UPPER로 정규화해서 케이스 이슈 제거
        - FILE_NAME은 반드시 있어야 함
        - CODE가 없으면 태그명(el.tag)을 code로 사용
        - NAME은 반드시 있어야 함(없으면 스킵)
        """
        print(f"[TREE] loading: {self.treecommon_path}")
        root = parse_xml_safely(self.treecommon_path, log=True)

        code_keys = ["CODE", "INDEXCODE", "INDEX_CODE", "ID", "IDX", "INDEXID", "INDEX_ID"]
        name_keys = ["NAME", "INDEXNAME", "INDEX_NAME", "TITLE", "CAPTION"]
        file_keys = ["FILE_NAME", "FILENAME", "FILE", "MAPFILE", "MAP_FILE"]
        path_keys = ["PATH", "FOLDER", "CATEGORY", "GROUP", "MENU"]

        items: List[CondItem] = []
        seen = set()

        for el in root.iter():
            attrib = el.attrib or {}
            if not attrib:
                continue

            a: Dict[str, str] = {}
            for k, v in attrib.items():
                kk = str(k).strip().upper()
                vv = v.strip() if isinstance(v, str) else str(v)
                a[kk] = vv

            file_name = ""
            for k in file_keys:
                if k in a and a[k]:
                    file_name = a[k].strip()
                    break
            if not file_name:
                continue

            code = ""
            for k in code_keys:
                if k in a and a[k]:
                    code = a[k].strip()
                    break
            if not code and el.tag:
                code = str(el.tag).strip()
            if not code:
                continue

            name = ""
            for k in name_keys:
                if k in a and a[k]:
                    name = a[k].strip()
                    break
            if not name and (el.text or "").strip():
                name = (el.text or "").strip()
            if not name:
                continue

            path = ""
            for k in path_keys:
                if k in a and a[k]:
                    path = a[k].strip()
                    break

            key = (file_name.lower(), code)
            if key in seen:
                continue
            seen.add(key)

            items.append(
                CondItem(
                    code=code,
                    name=name,
                    file_name=file_name.lower(),
                    path=path
                )
            )

        print(f"[TREE] loaded indicators: {len(items)}")
        empty_name = sum(1 for x in items if not (x.name and x.name.strip()))
        print(f"[TREE][DEBUG] empty name: {empty_name}/{len(items)}")
        for i in range(min(5, len(items))):
            x = items[i]
            print(f"[TREE][DEBUG] sample[{i}] code={x.code} file={x.file_name} name='{x.name}' path='{x.path}'")

        return items

    def map_path(self, file_name: str) -> str:
        return os.path.join(self.map_dir, f"map{file_name.lower()}.xml")

    def _load_map_file(self, path: str) -> Dict[str, MapDetail]:
        """
        map 파일 파싱 규칙 (mapd-e.xml 기준):
        - 코드 = 엘리먼트 tag (예: D1_1, D1_2 ...)
        - MAP_NAME / COMPLETE_INDEX_NAME 는 해당 엘리먼트 attrib
        - controls 캡션 = 하위 <CONTROL>들의 STATIC_CAPTION + (ComboBox면 COMBO_ITEM도 참고)
        """
        root = parse_xml_safely(path, log=True)

        def canon_code(tag: str) -> str:
            t = (tag or "").strip().upper()
            t = t.replace("-", "_").replace(".", "_").replace(" ", "")
            t = re.sub(r"[^A-Z0-9_]", "", t)
            t = re.sub(r"_+", "_", t)
            return t

        def looks_like_code(tag: str) -> bool:
            # A1, B1_7, C1_10_4 ... 전부 허용
            t = canon_code(tag)
            return bool(re.fullmatch(r"[A-Z]\d+(?:_\d+)*", t))


        out: Dict[str, MapDetail] = {}

        for el in root.iter():
            # CONTROL은 파라미터 정의라 제외
            if (el.tag or "").strip().upper() == "CONTROL":
                continue

            attrib = el.attrib or {}

            # 1) MAP_NAME/COMPLETE_INDEX_NAME 있으면 무조건 코드 노드로 인정 (가장 안전)
            has_map_attr = bool((attrib.get("MAP_NAME") or attrib.get("MapName") or "").strip())
            has_complete_attr = bool((attrib.get("COMPLETE_INDEX_NAME") or attrib.get("CompleteIndexName") or "").strip())

            # 2) 아니면 tag 패턴으로 판정
            if not (has_map_attr or has_complete_attr or looks_like_code(el.tag)):
                continue

            code = canon_code(el.tag)
            attrib = el.attrib or {}

            detail = out.get(code)
            if detail is None:
                detail = MapDetail()
                out[code] = detail

            # COMPLETE_INDEX_NAME / MAP_NAME
            if not detail.complete_index_name:
                v = (attrib.get("COMPLETE_INDEX_NAME") or attrib.get("CompleteIndexName") or "").strip()
                if v:
                    detail.complete_index_name = v

            if not detail.map_name:
                v = (attrib.get("MAP_NAME") or attrib.get("MapName") or "").strip()
                if v:
                    detail.map_name = v

            # controls: 하위 CONTROL들의 STATIC_CAPTION 중심
            for c in el.findall(".//CONTROL"):
                a = c.attrib or {}

                cap = (a.get("STATIC_CAPTION") or a.get("StaticCaption") or "").strip()
                if cap:
                    detail.controls.append(Control(caption=cap))

                # ComboBox면 COMBO_ITEM도 도움됨 (예: "상위순/하위순")
                if (a.get("TYPE") or "").strip().lower() in ("combobox", "combo"):
                    combo_item = (a.get("COMBO_ITEM") or "").strip()
                    if combo_item:
                        detail.controls.append(Control(caption=combo_item))

        total = len(out)
        have_complete = sum(1 for d in out.values() if (d.complete_index_name or "").strip())
        have_mapname = sum(1 for d in out.values() if (d.map_name or "").strip())
        print(f"[MAP][DEBUG] file={os.path.basename(path)} codes={total} complete={have_complete} mapname={have_mapname}")
        print(f"[MAP][DEBUG] code sample: {list(out.keys())[:10]}")

        return out


    def get_detail(self, file_name: str, code: str) -> Optional[MapDetail]:
        file_name = file_name.lower()
        if file_name not in self._map_cache:
            p = self.map_path(file_name)
            if not os.path.exists(p):
                print(f"[MAP][MISS] file_name='{file_name}' -> not found: {p}")
                return None
            print(f"[MAP] load file: {os.path.basename(p)}")
            self._map_cache[file_name] = self._load_map_file(p)

        code_n = code.strip().upper().replace("-", "_").replace(".", "_").replace(" ", "")
        code_n = re.sub(r"[^A-Z0-9_]", "", code_n)
        code_n = re.sub(r"_+", "_", code_n)

        d = self._map_cache[file_name].get(code_n)
        if d is None:
            print(f"[MAP][MISS] code='{code}' (norm='{code_n}') not found in file='{file_name}'")
        return d


# =========================
# 5) 컴파일(첫 실행만 XML 읽기)
# =========================
def build_indicator_text(idx: MapIndex, item: CondItem, use_map: bool = False) -> str:
    parts: List[str] = []

    if item.name and item.name.strip():
        parts.append(item.name.strip())
    else:
        parts.append(item.code.strip())

    if item.path and item.path.strip():
        parts.append(item.path.strip())

    if use_map:
        detail = idx.get_detail(item.file_name, item.code)
        if detail:
            if detail.complete_index_name:
                parts.append(detail.complete_index_name)
            if detail.map_name:
                parts.append(detail.map_name)
            for c in detail.controls:
                if c.caption:
                    parts.append(c.caption)

    uniq: List[str] = []
    seen = set()
    for p in parts:
        p = (p or "").strip()
        if not p or p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    return " | ".join(uniq)


def _file_sig(path: str) -> List:
    st = os.stat(path)
    # JSON 저장/로드 안정성을 위해 list로 고정 + 초단위 mtime 사용
    return [os.path.basename(path).lower(), int(st.st_mtime), int(st.st_size)]


def build_sources_fingerprint(base_dir: str) -> Dict:
    tree_path = os.path.join(base_dir, TREECOMMON_NAME)
    map_dir = os.path.join(base_dir, MAP_DIR_NAME)

    sigs: List[List] = []
    sigs.append(_file_sig(tree_path))

    if os.path.isdir(map_dir):
        for fn in sorted(os.listdir(map_dir)):
            if fn.lower().endswith(".xml"):
                sigs.append(_file_sig(os.path.join(map_dir, fn)))

    return {
        "treecommon": os.path.basename(tree_path).lower(),
        "map_dir": os.path.basename(map_dir).lower(),
        "sigs": sigs,
    }


def compile_from_xml(base_dir: str):
    """
    XML(treecommon + map/*.xml) -> compiled_items.pkl / compiled_texts.pkl / compiled_manifest.json
    """
    print("[COMPILE] start (XML -> compiled)")
    idx = MapIndex(base_dir)

    map_dir = os.path.join(base_dir, MAP_DIR_NAME)
    if os.path.isdir(map_dir):
        files = [fn for fn in os.listdir(map_dir) if fn.lower().endswith(".xml")]
        print(f"[COMPILE][DEBUG] map_dir={map_dir} xml_files={len(files)}")
        for fn in sorted(files)[:10]:
            print(f"[COMPILE][DEBUG] map_file_sample: {fn}")
    else:
        print(f"[COMPILE][DEBUG] map_dir not found: {map_dir}")

    compiled_items: List[CompiledItem] = []
    compiled_texts: List[str] = []

    for it in idx.cond_items:
        detail = idx.get_detail(it.file_name, it.code)

        controls: List[str] = []
        has_numeric_control = False
        if detail:
            for c in detail.controls:
                cap = (c.caption or "").strip()
                if not cap:
                    continue
                controls.append(cap)
                if re.search(r"(\d+|이상|이하|초과|미만|원|%|배|일|주|월|년)", cap):
                    has_numeric_control = True

        # display_name 우선순위: complete_index_name > map_name > treecommon name > code
        display_name = ""
        if detail:
            if detail.complete_index_name and detail.complete_index_name.strip():
                display_name = detail.complete_index_name.strip()
            elif detail.map_name and detail.map_name.strip():
                display_name = detail.map_name.strip()

        if not display_name:
            base = (it.name or "").strip()
            display_name = base if base else it.code.strip()

        # search_text: 이름 중심 + 필요한 정보만(controls는 상위 12개)
        parts: List[str] = []

        if display_name:
            parts.append(display_name)

        if it.name and it.name.strip() and it.name.strip() != display_name:
            parts.append(it.name.strip())

        if detail:
            if detail.complete_index_name and detail.complete_index_name.strip():
                parts.append(detail.complete_index_name.strip())
            if detail.map_name and detail.map_name.strip():
                parts.append(detail.map_name.strip())

        if controls:
            parts.extend(controls[:12])

        parts.append(it.code)
        parts.append(it.file_name)

        uniq: List[str] = []
        seen = set()
        for p in parts:
            p = (p or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            uniq.append(p)

        search_text = " | ".join(uniq)
        if not search_text.strip():
            search_text = build_indicator_text(idx, it, use_map=True)
            if not search_text.strip():
                search_text = f"{display_name} | {it.code} | {it.file_name}".strip()

        citem = CompiledItem(
            code=it.code,
            file_name=it.file_name,
            name=it.name or "",
            path=it.path or "",
            display_name=display_name,
            search_text=search_text,
            controls=controls,
            has_numeric_control=has_numeric_control,
        )

        if len(compiled_items) < 5:
            print(
                "[COMPILE][DEBUG] "
                f"code={it.code} file={it.file_name} "
                f"tree_name='{it.name}' "
                f"display='{display_name}' "
                f"search_head='{search_text.split('|', 1)[0].strip()}'"
            )

        compiled_items.append(citem)
        compiled_texts.append(search_text)

    # dict로 저장(호환성/안정성)
    compiled_items_dict = [
        {
            "code": x.code,
            "file_name": x.file_name,
            "name": x.name,
            "path": x.path,
            "display_name": x.display_name,
            "search_text": x.search_text,
            "controls": x.controls,
            "has_numeric_control": x.has_numeric_control,
        }
        for x in compiled_items
    ]

    with open(COMPILED_ITEMS_PATH, "wb") as f:
        pickle.dump(compiled_items_dict, f)

    with open(COMPILED_TEXTS_PATH, "wb") as f:
        pickle.dump(compiled_texts, f)

    manifest = {
        "created_ms": now_ms(),
        "sources": build_sources_fingerprint(base_dir),
        "count": len(compiled_items_dict),
    }
    save_json(COMPILED_MANIFEST_PATH, manifest)

    print(f"[COMPILE] done items={len(compiled_items_dict)}")
    print(f"[COMPILE] saved: {COMPILED_ITEMS_PATH}")
    print(f"[COMPILE] saved: {COMPILED_TEXTS_PATH}")
    print(f"[COMPILE] saved: {COMPILED_MANIFEST_PATH}")


# =========================
# 6) 임베딩 검색기(compiled 기반)
# =========================
class EmbeddingRetriever:
    def __init__(self, model_name: str = EMB_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.items: List[CondItem] = []
        self.texts: List[str] = []
        self.emb: Optional[np.ndarray] = None

        # (file_name, code) -> meta
        self.compiled_lookup: Dict[Tuple[str, str], Dict] = {}

        # code -> [meta...]
        self.compiled_lookup_by_code: Dict[str, List[Dict]] = {}

        self._q_cache: Dict[str, np.ndarray] = {}
        self._q_cache_max = 256


    def build_or_load(self, idx=None):
        src_fp = build_sources_fingerprint(BASE_DIR)
        old_manifest = load_json_if_exists(COMPILED_MANIFEST_PATH)

        need_compile = False
        if old_manifest is None:
            print("[COMPILE][CHECK] manifest: NONE -> need_compile=True")
            need_compile = True
        else:
            same = (old_manifest.get("sources") == src_fp)
            print(f"[COMPILE][CHECK] manifest sources same={same}")
            if not same:
                need_compile = True

        if need_compile:
            compile_from_xml(BASE_DIR)

        if not (os.path.exists(COMPILED_ITEMS_PATH) and os.path.exists(COMPILED_TEXTS_PATH)):
            raise RuntimeError("[COMPILED] missing compiled files")

        with open(COMPILED_ITEMS_PATH, "rb") as f:
            compiled_items = pickle.load(f)

        with open(COMPILED_TEXTS_PATH, "rb") as f:
            self.texts = pickle.load(f)

        # lookup (중복 code 덮어쓰기 방지)
        # 1) 정식 키: (file_name, code)
        self.compiled_lookup = {(x["file_name"], x["code"]): x for x in compiled_items}

        # 2) 보조 키: code-only (여러 개일 수 있으므로 list)
        self.compiled_lookup_by_code = {}
        for x in compiled_items:
            self.compiled_lookup_by_code.setdefault(x["code"], []).append(x)

        self.items = [
            CondItem(
                code=x["code"],
                name=(x.get("display_name") or x.get("name") or x["code"]),
                file_name=x["file_name"],
                path=(x.get("path") or ""),
            )
            for x in compiled_items
        ]

        if len(self.items) == 0:
            raise RuntimeError("[COMPILED] compiled_items=0")

        meta = {
            "compiled_sources": src_fp,
            "model": EMB_MODEL_NAME,
            "count": len(self.items),
        }

        old_meta = load_json_if_exists(EMB_META_PATH)
        cache_ok = (
            old_meta is not None
            and old_meta.get("model") == meta.get("model")
            and old_meta.get("count") == meta.get("count")
            and old_meta.get("compiled_sources") == meta.get("compiled_sources")
            and os.path.exists(EMB_ITEMS_PATH)
            and os.path.exists(EMB_VECTORS_PATH)
        )

        if cache_ok:
            print("[EMB] cache found -> loading (mmap)")
            with open(EMB_ITEMS_PATH, "rb") as f:
                self.items = pickle.load(f)
            self.emb = np.load(EMB_VECTORS_PATH, mmap_mode="r")

            if self.emb.ndim != 2 or self.emb.shape[0] != len(self.items) or self.emb.shape[1] == 0:
                print("[EMB] broken cache detected -> rebuilding")
                self.emb = None
            else:
                return

        print("[EMB] cache not found -> building embeddings")
        emb = self.model.encode(
            self.texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.emb = np.asarray(emb, dtype=np.float32)

        with open(EMB_ITEMS_PATH, "wb") as f:
            pickle.dump(self.items, f)
        np.save(EMB_VECTORS_PATH, self.emb)
        save_json(EMB_META_PATH, meta)

        print(f"[EMB] cache saved: {EMB_ITEMS_PATH}, {EMB_VECTORS_PATH}")

    def search(self, query: str, topk: int = 200) -> List[Tuple[CondItem, float]]:
        if self.emb is None or len(self.items) != self.emb.shape[0]:
            raise RuntimeError("[EMB] embeddings not loaded")

        qkey = query.strip()
        q = self._q_cache.get(qkey)
        if q is None:
            q_emb = self.model.encode([qkey], normalize_embeddings=True)
            q = np.asarray(q_emb[0], dtype=np.float32)

            # 간단 LRU(같은 효과): max 넘으면 비움
            if len(self._q_cache) >= self._q_cache_max:
                self._q_cache.clear()
            self._q_cache[qkey] = q

        sims = self.emb @ q

        n = len(self.items)
        if topk >= n:
            idxs = np.argsort(-sims)
        else:
            idxs = np.argpartition(-sims, topk)[:topk]
            idxs = idxs[np.argsort(-sims[idxs])]

        return [(self.items[i], float(sims[i])) for i in idxs]


# =========================
# 7) 학습(리뷰)
# =========================
def load_learn() -> Dict:
    if not os.path.exists(LEARN_PATH):
        return {"token_code": {}, "code_bias": {}}
    try:
        with open(LEARN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"token_code": {}, "code_bias": {}}


def save_learn(obj: Dict):
    with open(LEARN_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def learned_boost(learn: Dict, query: str, code: str) -> float:
    token_code = learn.get("token_code", {})
    code_bias = learn.get("code_bias", {})
    score = float(code_bias.get(code, 0.0))

    for t in tokenize_ko(query):
        td = token_code.get(t)
        if not td:
            continue
        score += float(td.get(code, 0.0))
    return score


def apply_learning(learn: Dict, query: str, hits: List[Tuple[CondItem, float]]) -> List[Tuple[CondItem, float]]:
    if not hits:
        return hits
    out = []
    for it, s in hits:
        b = learned_boost(learn, query, it.code)
        out.append((it, s + LEARN_ALPHA * b))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def review_and_learn(query: str, hits: List[Tuple[CondItem, float]], start: int, end: int):
    if not hits:
        print("[LEARN] no hits")
        return

    sub = hits[start:end]
    if not sub:
        print("[LEARN] empty page")
        return

    print("\n[LEARN] 이 페이지에서 '좋은 결과' 번호를 쉼표로 입력 (예: 1,3) / 엔터=취소")
    s = input(">> ").strip()
    if not s:
        print("[LEARN] cancelled")
        return

    picks = []
    for x in s.split(","):
        x = x.strip()
        if x.isdigit():
            picks.append(int(x))
    picks = [p for p in picks if 1 <= p <= len(sub)]
    if not picks:
        print("[LEARN] invalid picks")
        return

    learn = load_learn()
    token_code = learn.setdefault("token_code", {})
    code_bias = learn.setdefault("code_bias", {})

    q_tokens = tokenize_ko(query)
    if not q_tokens:
        q_tokens = ["__all__"]

    for p in picks:
        it, _score = sub[p - 1]
        code_bias[it.code] = float(code_bias.get(it.code, 0.0)) + 1.0

        for t in q_tokens:
            td = token_code.setdefault(t, {})
            td[it.code] = float(td.get(it.code, 0.0)) + 1.0

    save_learn(learn)
    print(f"[LEARN] saved -> {LEARN_PATH}")


# =========================
# 8) B: 숫자 조건 파싱 + 하이브리드 boost
# =========================
@dataclass
class NumericConstraint:
    raw: str
    value: float
    op: str  # "ge", "gt", "le", "lt", "eq"


def parse_korean_number(s: str) -> Optional[float]:
    t = s.strip().lower().replace(",", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(천|만|억)?", t)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2)
    mult = 1.0
    if unit == "천":
        mult = 1_000.0
    elif unit == "만":
        mult = 10_000.0
    elif unit == "억":
        mult = 100_000_000.0
    return num * mult


def parse_numeric_constraint(query: str) -> Optional[NumericConstraint]:
    q = query.strip().lower().replace(" ", "")

    patterns = [
        (r"(>=)(\d+(?:\.\d+)?(?:천|만|억)?)", "ge"),
        (r"(<=)(\d+(?:\.\d+)?(?:천|만|억)?)", "le"),
        (r"(>)(\d+(?:\.\d+)?(?:천|만|억)?)", "gt"),
        (r"(<)(\d+(?:\.\d+)?(?:천|만|억)?)", "lt"),
    ]
    for pat, op in patterns:
        m = re.search(pat, q)
        if m:
            v = parse_korean_number(m.group(2))
            if v is None:
                continue
            return NumericConstraint(raw=m.group(0), value=v, op=op)

    m = re.search(r"(\d+(?:\.\d+)?(?:천|만|억)?)(이상|이하|초과|미만)", q)
    if m:
        v = parse_korean_number(m.group(1))
        if v is None:
            return None
        word = m.group(2)
        if word == "이상":
            op = "ge"
        elif word == "이하":
            op = "le"
        elif word == "초과":
            op = "gt"
        else:
            op = "lt"
        return NumericConstraint(raw=m.group(0), value=v, op=op)

    return None


def apply_numeric_boost(
    query: str,
    hits: List[Tuple[CondItem, float]],
    retriever: "EmbeddingRetriever",
) -> List[Tuple[CondItem, float]]:
    nc = parse_numeric_constraint(query)
    if nc is None:
        return hits

    out: List[Tuple[CondItem, float]] = []

    for it, s in hits:
        meta = retriever.compiled_lookup.get((it.file_name, it.code))

        if meta is not None:
            has_numeric = bool(meta.get("has_numeric_control"))
        else:
            metas = retriever.compiled_lookup_by_code.get(it.code, [])
            has_numeric = any(m.get("has_numeric_control") for m in metas)

        if has_numeric:
            s = s + 0.04

        out.append((it, s))

    out.sort(key=lambda x: x[1], reverse=True)
    return out

def apply_rule_boost(
    query: str,
    hits: List[Tuple[CondItem, float]],
    retriever: "EmbeddingRetriever",
) -> List[Tuple[CondItem, float]]:
    """
    임베딩 결과에 룰 기반 토큰 매칭 점수를 가산한다.
    - display_name / controls / code(file_name 제외)에 따라 가중치 다르게
    """
    if not hits:
        return hits

    q_tokens = build_query_token_set(query)
    if not q_tokens:
        return hits

    out: List[Tuple[CondItem, float]] = []

    for it, s in hits:
        meta = retriever.compiled_lookup.get((it.file_name, it.code))
        if meta is None:
            # fallback: code-only 첫번째 (있으면)
            metas = retriever.compiled_lookup_by_code.get(it.code, [])
            meta = metas[0] if metas else None

        display_name = ""
        controls: List[str] = []
        if meta:
            display_name = (meta.get("display_name") or meta.get("name") or "").strip()
            controls = meta.get("controls") or []

        score_add = 0.0

        # 1) display_name 토큰 매칭 (강)
        if display_name:
            d_tokens = set(tokenize_rules(display_name))
            m = len(q_tokens & d_tokens)
            if m > 0:
                score_add += 0.030 * m

        # 2) controls 토큰 매칭 (중)
        if controls:
            # controls는 문구가 짧으니 전체를 합쳐서 한번에 매칭
            c_text = " ".join([c for c in controls if c])
            c_tokens = set(tokenize_rules(c_text))
            m = len(q_tokens & c_tokens)
            if m > 0:
                score_add += 0.012 * m

        # 3) code 직접 언급 시 (약)  ex) "B1_7"
        if it.code and it.code.lower() in query.lower():
            score_add += 0.020

        out.append((it, s + score_add))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


# =========================
# 9) AND/OR 파싱
# =========================
def _normalize_logic_tokens(s: str) -> str:
    s = s.strip()

    s = re.sub(r"\bOR\b", " 또는 ", s, flags=re.IGNORECASE)
    s = s.replace("|", " 또는 ").replace("||", " 또는 ")

    s = re.sub(r"\bAND\b", " 그리고 ", s, flags=re.IGNORECASE)
    s = s.replace("+", " 그리고 ").replace("&", " 그리고 ").replace(",", " 그리고 ").replace("및", " 그리고 ")

    s = re.sub(r"\s+", " ", s).strip()
    return s


def looks_like_combo(query: str) -> bool:
    q = _normalize_logic_tokens(query)
    return (" 또는 " in q) or (" 그리고 " in q)


def parse_combo_query(query: str) -> List[List[str]]:
    q = _normalize_logic_tokens(query)
    or_parts = [p.strip() for p in q.split(" 또는 ") if p.strip()]
    groups = []
    for part in or_parts:
        and_parts = [a.strip() for a in part.split(" 그리고 ") if a.strip()]
        if and_parts:
            groups.append(and_parts)
    return groups

def intersect_and_rank(
    clause_hits: List[List[Tuple[CondItem, float]]]
) -> List[Tuple[CondItem, float, int]]:
    """
    AND clause들의 후보 리스트를 받아서,
    모든 clause에 공통으로 등장하는 (file_name, code)만 남긴 뒤
    점수 합(score_sum) 기준으로 정렬한다.

    clause_hits 예시:
        [
            [(CondItem, score), ...],  # AND-1 결과
            [(CondItem, score), ...],  # AND-2 결과
            ...
        ]

    return:
        [(CondItem, score_sum, hit_count), ...]
        여기서 hit_count == len(clause_hits) 인 것만 반환
    """
    if not clause_hits:
        return []

    clause_cnt = len(clause_hits)
    acc: Dict[Tuple[str, str], Dict] = {}

    for hits in clause_hits:
        seen_in_clause = set()
        for it, score in hits:
            key = (it.file_name, it.code)

            # 같은 clause 안에서 동일 key 중복 집계 방지
            if key in seen_in_clause:
                continue
            seen_in_clause.add(key)

            if key not in acc:
                acc[key] = {
                    "item": it,
                    "score_sum": 0.0,
                    "hit_count": 0,
                }

            acc[key]["score_sum"] += float(score)
            acc[key]["hit_count"] += 1

    out: List[Tuple[CondItem, float, int]] = []
    for v in acc.values():
        if v["hit_count"] == clause_cnt:
            out.append((v["item"], v["score_sum"], v["hit_count"]))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


def run_combo_search(retriever, query, per_clause_topk=15, learn=None):
    if learn is None:
        learn = {"token_code": {}, "code_bias": {}}
        
    groups = parse_combo_query(query)

    print("==== 조합 조건 파싱 결과 ====")
    for gi, and_clauses in enumerate(groups, start=1):
        print(f"- OR 그룹 {gi}: " + " AND ".join(and_clauses))

    print("\n==== clause별 지표 후보 ====")

    group_results = []
    if learn is None:
        learn = {"token_code": {}, "code_bias": {}}

    for gi, and_clauses in enumerate(groups, start=1):
        print(f"\n[OR 그룹 {gi}]")
        clause_results = []

        # AND 개수 많을수록 교집합이 어려워지니 후보를 조금 더 넓힘
        dyn_topk = per_clause_topk
        if len(and_clauses) >= 3:
            dyn_topk = max(dyn_topk, 35)
        elif len(and_clauses) == 2:
            dyn_topk = max(dyn_topk, 25)

        for ci, clause in enumerate(and_clauses, start=1):
            print(f"\n  (AND-{ci}) clause: {clause}")

            hits = retriever.search(clause, topk=dyn_topk)
            hits = apply_learning(learn, clause, hits)
            hits = apply_numeric_boost(clause, hits, retriever)
            hits = apply_rule_boost(clause, hits, retriever)

            clause_results.append((clause, hits))

            if not hits:
                print("    - (매칭 없음)")
                continue

            for rank, (it, score) in enumerate(hits, start=1):
                print(f"    [{rank}] {it.name} (score={score:.4f}) code={it.code} file={it.file_name}")

        group_results.append(clause_results)
        
    # -----------------------------
    # AND 교집합 랭킹 출력
    # -----------------------------
    print("\n==== AND 교집합 랭킹 결과 (OR 그룹별) ====")

    for gi, clause_results in enumerate(group_results, start=1):
        print(f"\n[OR 그룹 {gi}]")

        clause_hits = []
        for _clause_text, hits in clause_results:
            clause_hits.append(hits)

        ranked = intersect_and_rank(clause_hits)

        if not ranked:
            print("  - AND 조건을 모두 만족하는 지표 없음")
            continue

        for rank, (it, score_sum, hit_count) in enumerate(ranked[:10], start=1):
            print(
                f"  [{rank}] {it.name} "
                f"(score_sum={score_sum:.4f}, clauses={hit_count}) "
                f"code={it.code} file={it.file_name}"
            )

    return group_results


# =========================
# 10) 출력(더보기)
# =========================
def print_page(hits: List[Tuple[CondItem, float]], page: int, page_size: int) -> Tuple[int, int]:
    start = page * page_size
    end = min(len(hits), start + page_size)

    if start >= len(hits):
        print("\n[PAGE] 더 이상 결과가 없습니다.")
        return (start, start)

    print(f"\n==== 결과 page {page + 1} ({start + 1}-{end}/{len(hits)}) ====")
    for i in range(start, end):
        it, score = hits[i]
        print(
            f"[{i - start + 1}] {it.name} "
            f"(score={score:.4f}) "
            f"code={it.code} "
            f"file={it.file_name}"
        )

    return (start, end)

class CondSearchEngine:
    """
    FastAPI 서버에서 startup 시 1회 로딩하고,
    request마다 search만 수행하도록 하는 상주 엔진 래퍼.
    """
    def __init__(self, base_dir: str = BASE_DIR, model_name: str = EMB_MODEL_NAME):
        set_base_dir(base_dir)
        self.model_name = model_name
        self.retriever = EmbeddingRetriever(model_name=model_name)
        self._loaded = False

        self._learn_cache = None
        self._learn_mtime = -1

    def _get_learn(self):
        """
        learn.json을 mtime 기반으로 캐시.
        파일이 바뀌지 않으면 디스크/JSON 파싱을 반복하지 않는다.
        """
        try:
            st = os.stat(LEARN_PATH)
            mtime = int(st.st_mtime)
        except FileNotFoundError:
            self._learn_cache = {"token_code": {}, "code_bias": {}}
            self._learn_mtime = -1
            return self._learn_cache

        if self._learn_cache is None or mtime != self._learn_mtime:
            self._learn_cache = load_learn()
            self._learn_mtime = mtime
        return self._learn_cache
    
    def load(self):
        if self._loaded:
            return
        t0 = time.time()
        self.retriever.build_or_load(None)
        t1 = time.time()
        print(f"[ENGINE] loaded init+build_or_load={t1 - t0:.3f}s")
        self._loaded = True

    def search(self, query: str, topk: int = 200, per_clause_topk: int = 15) -> dict:
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() at startup first.")

        query = (query or "").strip()
        if not query:
            return {"hits": []}

        # 조합 검색(AND/OR)
        if looks_like_combo(query):
            groups = parse_combo_query(query)
            learn = load_learn()

            group_results = []
            and_ranked_results = []

            for and_clauses in groups:
                clause_hits = []

                dyn_topk = per_clause_topk
                if len(and_clauses) >= 3:
                    dyn_topk = max(dyn_topk, 35)
                elif len(and_clauses) == 2:
                    dyn_topk = max(dyn_topk, 25)

                for clause in and_clauses:
                    hits = self.retriever.search(clause, topk=dyn_topk)
                    hits = apply_learning(learn, clause, hits)
                    hits = apply_numeric_boost(clause, hits, self.retriever)
                    hits = apply_rule_boost(clause, hits, self.retriever)
                    clause_hits.append(hits)

                ranked = intersect_and_rank(clause_hits)
                and_ranked_results.append([
                    {
                        "name": it.name,
                        "code": it.code,
                        "file_name": it.file_name,
                        "score_sum": float(score_sum),
                        "hit_count": int(hit_count),
                    }
                    for (it, score_sum, hit_count) in ranked[:50]
                ])

                group_results.append(and_clauses)

            return {
                "mode": "combo",
                "groups": group_results,
                "and_ranked": and_ranked_results,
            }

        # 단일 검색
        t0 = time.time()
        hits = self.retriever.search(query, topk=topk)
        t1 = time.time()
        learn = self._get_learn()
        hits = apply_learning(learn, query, hits)
        t2 = time.time()
        hits = apply_numeric_boost(query, hits, self.retriever)
        t3 = time.time()
        hits = apply_rule_boost(query, hits, self.retriever)
        t4 = time.time()
        print(f"[TIME] search={t1-t0:.3f}s learn={t2-t1:.3f}s numeric={t3-t2:.3f}s rule={t4-t3:.3f}s total={t4-t0:.3f}s")

        return {
            "mode": "single",
            "hits": [
                {
                    "name": it.name,
                    "code": it.code,
                    "file_name": it.file_name,
                    "score": float(score),
                }
                for (it, score) in hits[:topk]
            ]
        }

# =========================
# 11) 메인 run()
# =========================
def run(query: str, page_size: int = 10):
    t_run0 = time.time()
    print("[RUN] start")
    print(f"[RUN] query={query}")

    retriever = EmbeddingRetriever()
    t_model0 = time.time()
    retriever.build_or_load(None)
    t_model1 = time.time()
    print(f"[TIME] init+build_or_load={t_model1 - t_run0:.3f}s")

    # 조합 검색
    if looks_like_combo(query):
        learn = self._get_learn()
        group_results = run_combo_search(
            retriever,
            query,
            per_clause_topk=15,
            learn=learn
        )
        print("\n명령: [review] clause 학습 | [q] 종료")
        cmd = input(">> ").strip().lower()
        if cmd in ("review", "r"):
            print("입력 형식: <OR그룹번호> <AND번호> <선택번호들(쉼표)>")
            print('예) 1 2 3   => OR그룹1, AND-2 clause에서 후보 3번을 정답으로')
            s = input(">> ").strip()
            try:
                g, a, picks = s.split(maxsplit=2)
                g = int(g) - 1
                a = int(a) - 1
                chosen = [int(x.strip()) for x in picks.split(",") if x.strip().isdigit()]
            except Exception:
                print("[LEARN] invalid input")
                return

            if not (0 <= g < len(group_results)) or not (0 <= a < len(group_results[g])):
                print("[LEARN] out of range")
                return

            clause_text, hits = group_results[g][a]
            if not hits:
                print("[LEARN] no hits for that clause")
                return

            # clause 후보들(전체)을 대상으로 학습
            review_and_learn(clause_text, hits, 0, len(hits))

        return

    # 단일 검색 + 더보기 + 학습
    print("[SEARCH] start")
    t0 = time.time()
    hits = retriever.search(query, topk=200)
    t1 = time.time()
    hits = apply_learning(load_learn(), query, hits)
    t2 = time.time()
    hits = apply_numeric_boost(query, hits, retriever)
    t3 = time.time()
    hits = apply_rule_boost(query, hits, retriever)
    t4 = time.time()
    print(f"[TIME] search={t1-t0:.3f}s learn={t2-t1:.3f}s numeric={t3-t2:.3f}s rule={t4-t3:.3f}s total={t4-t0:.3f}s")

    if not hits:
        print("- (매칭 없음)")
        return

    page = 0
    while True:
        start, end = print_page(hits, page, page_size)

        print("\n명령: [more] 다음결과  |  [review] 이 페이지로 학습  |  [q] 종료")
        cmd = input(">> ").strip().lower()

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd == "more":
            page += 1
        elif cmd == "review":
            review_and_learn(query, hits, start, end)
            hits = apply_learning(load_learn(), query, hits)
            hits = apply_numeric_boost(query, hits, retriever.compiled_lookup)
        else:
            print("알 수 없는 명령입니다. more / review / q 중 하나를 입력하세요.")


# =========================
# 12) 엔트리포인트
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('사용법: python Untitled-1.py "자연어 조건"')
        input("엔터를 누르면 종료됩니다...")
        raise SystemExit(1)

    q = " ".join(sys.argv[1:])
    try:
        run(q, page_size=10)
    except Exception as e:
        print("\n[ERROR]", repr(e))
        raise
    finally:
        input("\n엔터를 누르면 종료됩니다...")
