import numpy as np
from collections import defaultdict


class SegmParams:
    """
    仅用于 SEGM 的 COCO 评估参数（移植自 pycocotools.Params 的 segm 子集）。

    与 bbox 的 Params 保持一致：
    - IoU 阈值 [0.50:0.05:0.95]
    - 召回阈值 101 点 [0:0.01:1]
    - 最大检测数 [1, 10, 100]
    - 面积分桶（像素平方）
    """

    def __init__(self):
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.0, 101)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0**2, 1e10], [0**2, 32**2], [32**2, 96**2], [96**2, 1e10]]
        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1
        self.imgIds = []
        self.catIds = []


def rle_encode(mask: np.ndarray) -> dict:
    """
    将二值 mask 编码为 COCO 风格的 RLE（counts 为 Python int 列表）。

    约定：
    - 输入 mask 为 shape [H, W]，值域 {0,1} 或 bool
    - 使用 Fortran order (column-major) 展平，与 COCO/pycocotools 对齐
    - counts 以 0-run 起始（如果第一个像素为 1，则第一个 count 为 0）
    """
    if mask.ndim != 2:
        raise ValueError(f"rle_encode expects 2D mask, got shape={mask.shape}")
    h, w = mask.shape
    m = (mask.astype(np.uint8) > 0).reshape(-1, order="F")

    counts = []
    prev = 0
    run_len = 0
    for p in m.tolist():
        if p == prev:
            run_len += 1
        else:
            counts.append(int(run_len))
            run_len = 1
            prev = p
    counts.append(int(run_len))

    return {"size": [int(h), int(w)], "counts": counts}


def rle_area(rle: dict) -> int:
    """计算 RLE 的前景面积（像素数）。"""
    counts = rle.get("counts")
    if not counts:
        return 0
    # counts 从 0-run 开始，1-run 位于奇数索引
    return int(sum(int(x) for x in counts[1::2]))


def rle_iou_single(dt: dict, gt: dict, iscrowd: int = 0) -> float:
    """
    计算单对 RLE 的 IoU。

    crowd 规则（对齐 COCO bbox 评估器的实现方式）：
    - 若 gt.iscrowd==1，则分母使用 area(dt)（即 inter / area(dt)）
    - 否则为标准 IoU：inter / (area(dt)+area(gt)-inter)
    """
    if dt["size"] != gt["size"]:
        raise ValueError(f"RLE size mismatch: dt={dt['size']} gt={gt['size']}")

    dt_counts = dt.get("counts") or []
    gt_counts = gt.get("counts") or []

    if not dt_counts or not gt_counts:
        return 0.0

    # 双指针遍历 runs，计算 inter 与 union
    i = j = 0
    di = int(dt_counts[0])
    gj = int(gt_counts[0])
    dv = 0  # dt current value (0/1), counts 从 0-run 开始
    gv = 0

    inter = 0
    union = 0

    while True:
        n = di if di < gj else gj
        if dv == 1 and gv == 1:
            inter += n
            union += n
        elif dv == 1 or gv == 1:
            union += n

        di -= n
        gj -= n

        if di == 0:
            i += 1
            if i >= len(dt_counts):
                break
            di = int(dt_counts[i])
            dv ^= 1
        if gj == 0:
            j += 1
            if j >= len(gt_counts):
                break
            gj = int(gt_counts[j])
            gv ^= 1

    if iscrowd:
        denom = rle_area(dt)
    else:
        denom = rle_area(dt) + rle_area(gt) - inter
    denom = max(float(denom), 1e-10)
    return float(inter) / denom


class COCOevalSegmMM:
    """
    本地移植版 COCO SEGM 评估器：不依赖 pycocotools，仅做实例分割(segm)评估。

    输入 gts/dts 为 COCO 风格的字典列表（子集）：
    - gt 必需字段：image_id, category_id, segmentation(RLE), iscrowd(optional), ignore(optional), area(optional)
    - dt 必需字段：image_id, category_id, segmentation(RLE), score, area(optional)

    评估逻辑对齐 bbox 版 COCOevalBBoxMM：
    - 面积分桶：基于 GT area；未匹配且预测 area 不在当前分桶的预测记为忽略（不计 FP）
    - 每图 Top-K：按 score 裁剪至 maxDets[-1]（默认 100）
    - crowd 规则：gt.iscrowd==1 时 IoU = inter / area(dt)
    - accumulate 中使用 mergesort 保持稳定排序
    """

    def __init__(self):
        self.params = SegmParams()
        self.evalImgs = defaultdict(list)
        self.eval = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.ious = {}
        self.stats = None
        self.stats_as_dict = {}

    def set_data(self, gts, dts, imgIds, catIds):
        p = self.params
        p.imgIds = sorted(list(set(imgIds)))
        p.catIds = sorted(list(set(catIds)))

        gid = 1
        for g in gts:
            g = dict(g)
            if "id" not in g:
                g["id"] = gid
                gid += 1
            g["iscrowd"] = int(g.get("iscrowd", 0))
            g["ignore"] = int(g.get("ignore", 0)) or g["iscrowd"]
            assert "image_id" in g and "category_id" in g and "segmentation" in g
            if "area" not in g:
                g["area"] = float(rle_area(g["segmentation"]))
            self._gts[(g["image_id"], g["category_id"])].append(g)

        did = 1
        for d in dts:
            d = dict(d)
            if "id" not in d:
                d["id"] = did
                did += 1
            assert "image_id" in d and "category_id" in d and "segmentation" in d and "score" in d
            if "area" not in d:
                d["area"] = float(rle_area(d["segmentation"]))
            self._dts[(d["image_id"], d["category_id"])].append(d)

    def evaluate(self):
        p = self.params
        catIds = p.catIds if p.useCats else [-1]
        self.ious = {(imgId, catId): self._compute_iou(imgId, catId) for imgId in p.imgIds for catId in catIds}

        maxDet = p.maxDets[-1]
        self.evalImgs = [
            self._evaluate_img(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]

    def accumulate(self):
        p = self.params
        if not self.evalImgs:
            raise RuntimeError("Please run evaluate() first")

        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        catIds = p.catIds if p.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, p.areaRng))
        setM = set(p.maxDets)
        setI = set(p.imgIds)

        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]

        I0 = len(p.imgIds)
        A0 = len(p.areaRng)

        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if not E:
                        continue

                    dtScores = np.concatenate([e["dtScores"][:maxDet] for e in E])
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e["dtMatches"][:, :maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, :maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])

                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue

                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1, dtype=float)
                    fp_sum = np.cumsum(fps, axis=1, dtype=float)

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (tp + fp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        recall[t, k, a, m] = rc[-1] if nd else 0.0

                        pr_list = pr.tolist()
                        for i in range(nd - 1, 0, -1):
                            if pr_list[i] > pr_list[i - 1]:
                                pr_list[i - 1] = pr_list[i]

                        inds_rc = np.searchsorted(rc, p.recThrs, side="left")
                        for ri, pi in enumerate(inds_rc):
                            if pi < nd:
                                q[ri] = pr_list[pi]
                                ss[ri] = dtScoresSorted[pi]
                            else:
                                q[ri] = 0.0
                                ss[ri] = 0.0

                        precision[t, :, k, a, m] = q
                        scores[t, :, k, a, m] = ss

        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }

    def summarize(self):
        if not self.eval:
            raise RuntimeError("Please run accumulate() first")
        p = self.params
        precision = self.eval["precision"]
        recall = self.eval["recall"]

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            aind = [i for i, a in enumerate(p.areaRngLbl) if a == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                s = precision[:, :, :, aind, mind]
                if iouThr is not None:
                    t = np.where(np.isclose(p.iouThrs, iouThr))[0]
                    s = precision[t, :, :, aind, mind]
                s = s[s > -1]
            else:
                s = recall[:, :, aind, mind]
                if iouThr is not None:
                    t = np.where(np.isclose(p.iouThrs, iouThr))[0]
                    s = recall[t, :, aind, mind]
                s = s[s > -1]
            if s.size == 0:
                return -1.0
            return float(np.mean(s))

        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=0.5)
        stats[2] = _summarize(1, iouThr=0.75)
        stats[3] = _summarize(1, areaRng="small")
        stats[4] = _summarize(1, areaRng="medium")
        stats[5] = _summarize(1, areaRng="large")
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
        stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])

        self.stats = stats
        self.stats_as_dict = {
            "AP": stats[0],
            "AP50": stats[1],
            "AP75": stats[2],
            "APsmall": stats[3],
            "APmedium": stats[4],
            "APlarge": stats[5],
            "AR1": stats[6],
            "AR10": stats[7],
            "AR100": stats[8],
            "ARsmall": stats[9],
            "ARmedium": stats[10],
            "ARlarge": stats[11],
        }
        return self.stats_as_dict

    def compute_per_class_metrics(self):
        if not self.eval:
            return {}
        p = self.params
        precision = self.eval["precision"]  # [T,R,K,A,M]
        if precision.ndim != 5:
            return {}

        aind = [i for i, a in enumerate(p.areaRngLbl) if a == "all"]
        mind = [len(p.maxDets) - 1]

        def _thr_idx(val):
            w = np.where(np.isclose(p.iouThrs, val))[0]
            return int(w[0]) if w.size else None

        idx50 = _thr_idx(0.5)
        idx75 = _thr_idx(0.75)

        per_class = {}
        for k, catId in enumerate(p.catIds if p.useCats else [-1]):
            s = precision[:, :, k, aind, mind]
            s = np.squeeze(s, axis=(3, 4)) if s.ndim == 5 else np.squeeze(s)
            s_valid = s[s > -1]
            ap = float(np.mean(s_valid)) if s_valid.size else 0.0

            if idx50 is not None:
                s50 = s[idx50, :]
                s50v = s50[s50 > -1]
                ap50 = float(np.mean(s50v)) if s50v.size else 0.0
            else:
                ap50 = 0.0

            if idx75 is not None:
                s75 = s[idx75, :]
                s75v = s75[s75 > -1]
                ap75 = float(np.mean(s75v)) if s75v.size else 0.0
            else:
                ap75 = 0.0

            per_class[int(catId)] = {"AP": ap, "AP50": ap50, "AP75": ap75}

        return per_class

    def _compute_iou(self, imgId, catId):
        p = self.params
        gts = self._gts[(imgId, catId)] if p.useCats else [g for c in p.catIds for g in self._gts[(imgId, c)]]
        dts = self._dts[(imgId, catId)] if p.useCats else [d for c in p.catIds for d in self._dts[(imgId, c)]]
        if len(gts) == 0 and len(dts) == 0:
            return []

        inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > self.params.maxDets[-1]:
            dts = dts[: self.params.maxDets[-1]]

        if len(gts) == 0 or len(dts) == 0:
            return []

        iscrowd = np.array([int(gt.get("iscrowd", 0)) for gt in gts], dtype=np.int32)
        D = len(dts)
        G = len(gts)
        iou = np.zeros((D, G), dtype=np.float64)
        for dind, dt in enumerate(dts):
            for gind, gt in enumerate(gts):
                iou[dind, gind] = rle_iou_single(dt["segmentation"], gt["segmentation"], int(iscrowd[gind]))
        return iou

    def _evaluate_img(self, imgId, catId, aRng, maxDet):
        p = self.params
        if p.useCats:
            gt = list(self._gts[(imgId, catId)])
            dt = list(self._dts[(imgId, catId)])
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[(imgId, cId)]]
            dt = [_ for cId in p.catIds for _ in self._dts[(imgId, cId)]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            g["_ignore"] = 1 if (g.get("ignore", 0) or (g["area"] < aRng[0] or g["area"] > aRng[1])) else 0

        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[:maxDet]]
        iscrowd = np.array([int(o.get("iscrowd", 0)) for o in gt], dtype=np.int32)

        ious = self.ious.get((imgId, catId), [])
        if len(ious) > 0:
            ious = ious[:, gtind]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt], dtype=np.uint8)
        dtIg = np.zeros((T, D))

        if G > 0 and D > 0 and len(ious) != 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou_thr = min(t, 1 - 1e-10)
                    m = -1
                    for gind, g in enumerate(gt):
                        if gtm[tind, gind] > 0 and iscrowd[gind] == 0:
                            continue
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        if ious[dind, gind] < iou_thr:
                            continue
                        iou_thr = ious[dind, gind]
                        m = gind
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]

        if D > 0:
            dt_area = np.array([float(d.get("area", 0.0)) for d in dt], dtype=np.float64)
            outside = (dt_area < aRng[0]) | (dt_area > aRng[1])
            outside = outside.reshape(1, D)
            dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(outside, T, axis=0)))

        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }
