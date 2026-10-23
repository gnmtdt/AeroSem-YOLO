import numpy as np
from collections import defaultdict


class BBoxParams:
    """
    仅用于 BBOX 的 COCO 评估参数（移植自 pycocotools.Params 的 bbox 子集）。
    """

    def __init__(self):
        # IoU 阈值 [0.50:0.05:0.95]
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        # 召回阈值 101 个点 [0:0.01:1]
        self.recThrs = np.linspace(0.0, 1.0, 101)
        # 最大检测数 [1, 10, 100]
        self.maxDets = [1, 10, 100]
        # 面积分桶（像素平方）
        self.areaRng = [[0**2, 1e10], [0**2, 32**2], [32**2, 96**2], [96**2, 1e10]]
        self.areaRngLbl = ["all", "small", "medium", "large"]
        # 是否按类别
        self.useCats = 1
        # 评估图像和类别在 set_data 时填充
        self.imgIds = []
        self.catIds = []


class COCOevalBBoxMM:
    """
    本地移植版的 COCO BBox 评估器：不依赖 pycocotools，逻辑对齐其 evaluate/accumulate/summarize。

    仅实现 bbox 评估：
    - 面积分桶：基于 GT 面积；未匹配且预测面积不在当前分桶的预测记为忽略（不计 FP）。
    - 每图 Top-K：在 evaluate 阶段按分数裁剪至 maxDets[-1]（默认 100）。
    - crowd 规则：若 gt.iscrowd==1，则 IoU = inter / area(dt)，否则为标准 IoU。
    - 稳定排序：accumulate 中使用 mergesort。
    """

    def __init__(self):
        self.params = BBoxParams()
        self.evalImgs = defaultdict(list)  # [K x A x I]
        self.eval = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.ious = {}
        self.stats = None
        self.stats_as_dict = {}

    # ------------------------------
    # 公共 API
    # ------------------------------
    def set_data(self, gts, dts, imgIds, catIds):
        """设置数据并完成必要的预处理。gts/dts 为字典列表（COCO 风格）。"""
        p = self.params
        p.imgIds = sorted(list(set(imgIds)))
        p.catIds = sorted(list(set(catIds)))

        # 归一化字段，构建 _gts/_dts
        gid = 1
        for g in gts:
            g = dict(g)  # 浅拷贝
            if 'id' not in g:
                g['id'] = gid
                gid += 1
            g['iscrowd'] = int(g.get('iscrowd', 0))
            g['ignore'] = int(g.get('ignore', 0)) or g['iscrowd']
            # 必要字段：image_id/category_id/bbox/area
            assert 'image_id' in g and 'category_id' in g and 'bbox' in g
            if 'area' not in g:
                x, y, w, h = g['bbox']
                g['area'] = max(0.0, float(w)) * max(0.0, float(h))
            self._gts[(g['image_id'], g['category_id'])].append(g)

        did = 1
        for d in dts:
            d = dict(d)
            if 'id' not in d:
                d['id'] = did
                did += 1
            assert 'image_id' in d and 'category_id' in d and 'bbox' in d and 'score' in d
            if 'area' not in d:
                x, y, w, h = d['bbox']
                d['area'] = max(0.0, float(w)) * max(0.0, float(h))
            self._dts[(d['image_id'], d['category_id'])].append(d)

    def evaluate(self):
        """逐图像/逐类/逐面积桶评估，生成 evalImgs。"""
        p = self.params
        catIds = p.catIds if p.useCats else [-1]
        # 预先计算 IoU 矩阵
        self.ious = {(imgId, catId): self._compute_iou(imgId, catId)
                     for imgId in p.imgIds for catId in catIds}

        maxDet = p.maxDets[-1]
        self.evalImgs = [self._evaluate_img(imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in p.areaRng
                         for imgId in p.imgIds]

    def accumulate(self):
        """汇总各图像/类别/面积与 TopK 的精度/召回，生成与 pycocotools 一致的张量。"""
        p = self.params
        if not self.evalImgs:
            raise RuntimeError('Please run evaluate() first')

        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # 建立索引
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
                    if len(E) == 0:
                        continue

                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
                    # 稳定排序（与 pycocotools 一致）
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
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
                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # 单调性修正
                        pr_list = pr.tolist()
                        for i in range(nd - 1, 0, -1):
                            if pr_list[i] > pr_list[i - 1]:
                                pr_list[i - 1] = pr_list[i]

                        inds_rc = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds_rc):
                                if pi < nd:
                                    q[ri] = pr_list[pi]
                                    ss[ri] = dtScoresSorted[pi]
                                else:
                                    q[ri] = 0.0
                                    ss[ri] = 0.0
                        except Exception:
                            pass
                        precision[t, :, k, a, m] = q
                        scores[t, :, k, a, m] = ss

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }

    def summarize(self):
        """生成 12 项指标并保存到 self.stats / self.stats_as_dict。"""
        if not self.eval:
            raise RuntimeError('Please run accumulate() first')
        p = self.params
        precision = self.eval['precision']  # [T,R,K,A,M]
        recall = self.eval['recall']        # [T,K,A,M]

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
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
        # AP
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=0.5)
        stats[2] = _summarize(1, iouThr=0.75)
        stats[3] = _summarize(1, areaRng='small')
        stats[4] = _summarize(1, areaRng='medium')
        stats[5] = _summarize(1, areaRng='large')
        # AR
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

        self.stats = stats
        self.stats_as_dict = {
            'AP': stats[0], 'AP50': stats[1], 'AP75': stats[2],
            'APsmall': stats[3], 'APmedium': stats[4], 'APlarge': stats[5],
            'AR1': stats[6], 'AR10': stats[7], 'AR100': stats[8],
            'ARsmall': stats[9], 'ARmedium': stats[10], 'ARlarge': stats[11],
        }

        # 扩展：按尺寸的 AP50 / AP75（非标准COCO摘要，但便于分析；不破坏原 12 项）
        def _area_iou_ap(area_lbl: str, iou_thr: float):
            aind = [i for i, a in enumerate(p.areaRngLbl) if a == area_lbl]
            mind = [len(p.maxDets) - 1]  # 使用最大 maxDets（通常为100）
            t = np.where(np.isclose(p.iouThrs, iou_thr))[0]
            if t.size == 0:
                return -1.0
            s = precision[t, :, :, aind, mind]
            # squeeze 维度，得到 [K,R]
            s = np.squeeze(s)
            s = s[s > -1]
            if s.size == 0:
                return -1.0
            return float(np.mean(s))

        for area_lbl, key_prefix in [("small", "APsmall"), ("medium", "APmedium"), ("large", "APlarge")]:
            ap50 = _area_iou_ap(area_lbl, 0.5)
            ap75 = _area_iou_ap(area_lbl, 0.75)
            self.stats_as_dict[f'{key_prefix}50'] = ap50
            self.stats_as_dict[f'{key_prefix}75'] = ap75
        return self.stats_as_dict

    def compute_per_class_metrics(self):
        """
        计算每类 AP、AP50、AP75（area=all, maxDets=max）。返回 dict: {catId: {AP, AP50, AP75}}。
        """
        if not self.eval:
            return {}
        p = self.params
        precision = self.eval['precision']  # [T,R,K,A,M]
        if precision.ndim != 5:
            return {}
        aind = [i for i, a in enumerate(p.areaRngLbl) if a == 'all']
        mind = [len(p.maxDets) - 1]
        # IoU index
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

            per_class[int(catId)] = {'AP': ap, 'AP50': ap50, 'AP75': ap75}

        return per_class

    # ------------------------------
    # 内部：IoU 与单图像评估
    # ------------------------------
    def _compute_iou(self, imgId, catId):
        p = self.params
        gts = self._gts[(imgId, catId)] if p.useCats else [g for c in p.catIds for g in self._gts[(imgId, c)]]
        dts = self._dts[(imgId, catId)] if p.useCats else [d for c in p.catIds for d in self._dts[(imgId, c)]]
        if len(gts) == 0 and len(dts) == 0:
            return []
        # dt 按 score 降序，截到 maxDets[-1]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > self.params.maxDets[-1]:
            dts = dts[: self.params.maxDets[-1]]

        if len(gts) == 0 or len(dts) == 0:
            return []

        # 转为 xyxy
        def xywh_to_xyxy(bb):
            x, y, w, h = bb
            return [x, y, x + w, y + h]

        g = np.array([xywh_to_xyxy(gt['bbox']) for gt in gts], dtype=np.float64)  # [G,4]
        d = np.array([xywh_to_xyxy(dt['bbox']) for dt in dts], dtype=np.float64)  # [D,4]
        iscrowd = np.array([int(gt.get('iscrowd', 0)) for gt in gts], dtype=np.int32)  # [G]

        D = d.shape[0]
        G = g.shape[0]
        if D == 0 or G == 0:
            return np.zeros((0, 0), dtype=np.float64)

        # 交集
        x1 = np.maximum(d[:, None, 0], g[None, :, 0])
        y1 = np.maximum(d[:, None, 1], g[None, :, 1])
        x2 = np.minimum(d[:, None, 2], g[None, :, 2])
        y2 = np.minimum(d[:, None, 3], g[None, :, 3])
        inter_w = np.clip(x2 - x1, 0, None)
        inter_h = np.clip(y2 - y1, 0, None)
        inter = inter_w * inter_h  # [D,G]

        # 面积
        area_d = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])  # [D]
        area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])  # [G]

        # union，考虑 crowd：对 iscrowd==1 的列，分母改为 area(dt)
        union = area_d[:, None] + area_g[None, :] - inter
        if np.any(iscrowd == 1):
            crowd_cols = np.where(iscrowd == 1)[0]
            if crowd_cols.size:
                union[:, crowd_cols] = area_d[:, None]

        iou = np.divide(inter, np.maximum(union, 1e-10))
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

        # 标记 GT 忽略（面积桶 + iscrowd）
        for g in gt:
            g['_ignore'] = 1 if (g.get('ignore', 0) or (g['area'] < aRng[0] or g['area'] > aRng[1])) else 0

        # 排序：gt 先非忽略，dt 按分数并裁剪
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[:maxDet]]
        iscrowd = np.array([int(o.get('iscrowd', 0)) for o in gt], dtype=np.int32)

        # IoUs（按 gt 排序重排列）
        ious = self.ious.get((imgId, catId), [])
        if len(ious) > 0:
            ious = ious[:, gtind]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt], dtype=np.uint8)
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
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']

        # 未匹配的 dt 若面积不在当前桶，记忽略（不算 FP）
        if D > 0:
            dt_area = np.array([max(0.0, d['bbox'][2]) * max(0.0, d['bbox'][3]) for d in dt], dtype=np.float64)
            outside = (dt_area < aRng[0]) | (dt_area > aRng[1])
            outside = outside.reshape(1, D)
            dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(outside, T, axis=0)))

        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }
