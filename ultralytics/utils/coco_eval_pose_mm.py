import numpy as np
from collections import defaultdict

# COCO 官方 17 点人体关键点 sigma 值
COCO_KPT_SIGMAS = np.array([
    0.026,  # nose
    0.025,  # left_eye
    0.025,  # right_eye
    0.035,  # left_ear
    0.035,  # right_ear
    0.079,  # left_shoulder
    0.079,  # right_shoulder
    0.072,  # left_elbow
    0.072,  # right_elbow
    0.062,  # left_wrist
    0.062,  # right_wrist
    0.107,  # left_hip
    0.107,  # right_hip
    0.087,  # left_knee
    0.087,  # right_knee
    0.089,  # left_ankle
    0.089,  # right_ankle
])


class PoseParams:
    """COCO Pose (OKS) 评估参数。"""

    def __init__(self, kpt_sigmas=None):
        # OKS 阈值 [0.50:0.05:0.95]
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        # 召回阈值 101 个点
        self.recThrs = np.linspace(0.0, 1.0, 101)
        # 最大检测数
        self.maxDets = [1, 10, 100]
        # 面积分桶（像素平方）
        self.areaRng = [[0**2, 1e10], [0**2, 32**2], [32**2, 96**2], [96**2, 1e10]]
        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1
        self.imgIds = []
        self.catIds = []
        # 关键点 sigma（默认 COCO 17 点）
        self.kpt_sigmas = kpt_sigmas if kpt_sigmas is not None else COCO_KPT_SIGMAS


class COCOevalPoseMM:
    """
    纯 Python COCO Pose (OKS) 评估器，零外部依赖。

    接口对齐 COCOevalBBoxMM / COCOevalSegmMM：
      set_data() -> evaluate() -> accumulate() -> summarize() + compute_per_class_metrics()

    核心差异：用 OKS (Object Keypoint Similarity) 替代 IoU 进行匹配。
    OKS 公式：
      OKS(d,g) = sum_{i: v_i>0} exp(-d_i^2 / (2 * s^2 * k_i^2)) / count(v_i>0)
    其中 d_i 为第 i 个关键点欧氏距离，s=sqrt(area) 为对象尺度，k_i 为 sigma_i。
    """

    def __init__(self, kpt_sigmas=None):
        self.params = PoseParams(kpt_sigmas=kpt_sigmas)
        self.evalImgs = defaultdict(list)
        self.eval = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.ious = {}  # 存储 OKS 值（复用 ious 字段名以保持接口一致）
        self.stats = None
        self.stats_as_dict = {}

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def set_data(self, gts, dts, imgIds, catIds):
        """
        设置评估数据。

        Args:
            gts: GT 列表，每项需包含 {image_id, category_id, keypoints, bbox, area}
                 keypoints: [x1,y1,v1, x2,y2,v2, ...] 平铺格式（v: 0=未标注, 1=遮挡, 2=可见）
                 bbox: [x,y,w,h]  area: 对象面积（用于 OKS 缩放和尺寸过滤）
            dts: 预测列表，每项需包含 {image_id, category_id, keypoints, bbox, score, area}
            imgIds: 图像 ID 列表
            catIds: 类别 ID 列表
        """
        p = self.params
        p.imgIds = sorted(list(set(imgIds)))
        p.catIds = sorted(list(set(catIds)))

        gid = 1
        for g in gts:
            g = dict(g)
            if 'id' not in g:
                g['id'] = gid
                gid += 1
            g['iscrowd'] = int(g.get('iscrowd', 0))
            g['ignore'] = int(g.get('ignore', 0)) or g['iscrowd']
            assert 'image_id' in g and 'category_id' in g and 'keypoints' in g
            # 自动计算 area
            if 'area' not in g:
                if 'bbox' in g:
                    x, y, w, h = g['bbox']
                    g['area'] = max(0.0, float(w)) * max(0.0, float(h))
                else:
                    g['area'] = 0.0
            # 计算可见关键点数
            if 'num_keypoints' not in g:
                kpts = np.array(g['keypoints']).reshape(-1, 3)
                g['num_keypoints'] = int(np.sum(kpts[:, 2] > 0))
            # 无可见关键点的 GT 标记为忽略（COCO keypoint 标准）
            if g['num_keypoints'] == 0:
                g['ignore'] = 1
            self._gts[(g['image_id'], g['category_id'])].append(g)

        did = 1
        for d in dts:
            d = dict(d)
            if 'id' not in d:
                d['id'] = did
                did += 1
            assert 'image_id' in d and 'category_id' in d and 'keypoints' in d and 'score' in d
            if 'area' not in d:
                if 'bbox' in d:
                    x, y, w, h = d['bbox']
                    d['area'] = max(0.0, float(w)) * max(0.0, float(h))
                else:
                    d['area'] = 0.0
            self._dts[(d['image_id'], d['category_id'])].append(d)

    def evaluate(self):
        """逐图像/逐类/逐面积桶评估，使用 OKS 替代 IoU。"""
        p = self.params
        catIds = p.catIds if p.useCats else [-1]
        self.ious = {
            (imgId, catId): self._compute_oks(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        maxDet = p.maxDets[-1]
        self.evalImgs = [
            self._evaluate_img(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]

    def accumulate(self):
        """汇总各图像/类别/面积的精度/召回（逻辑与 COCOevalBBoxMM 一致）。"""
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
        """生成 12 项 COCO 指标（与 COCOevalBBoxMM 格式一致）。"""
        if not self.eval:
            raise RuntimeError('Please run accumulate() first')
        p = self.params
        precision = self.eval['precision']
        recall = self.eval['recall']

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
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=0.5)
        stats[2] = _summarize(1, iouThr=0.75)
        stats[3] = _summarize(1, areaRng='small')
        stats[4] = _summarize(1, areaRng='medium')
        stats[5] = _summarize(1, areaRng='large')
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

        # 扩展：按尺寸的 AP50/AP75
        def _area_iou_ap(area_lbl, iou_thr):
            aind = [i for i, a in enumerate(p.areaRngLbl) if a == area_lbl]
            mind = [len(p.maxDets) - 1]
            t = np.where(np.isclose(p.iouThrs, iou_thr))[0]
            if t.size == 0:
                return -1.0
            s = precision[t, :, :, aind, mind]
            s = np.squeeze(s)
            s = s[s > -1]
            if s.size == 0:
                return -1.0
            return float(np.mean(s))

        for area_lbl, key_prefix in [("small", "APsmall"), ("medium", "APmedium"), ("large", "APlarge")]:
            self.stats_as_dict[f'{key_prefix}50'] = _area_iou_ap(area_lbl, 0.5)
            self.stats_as_dict[f'{key_prefix}75'] = _area_iou_ap(area_lbl, 0.75)
        return self.stats_as_dict

    def compute_per_class_metrics(self):
        """计算每类 AP/AP50/AP75（area=all, maxDets=max）。"""
        if not self.eval:
            return {}
        p = self.params
        precision = self.eval['precision']
        if precision.ndim != 5:
            return {}
        aind = [i for i, a in enumerate(p.areaRngLbl) if a == 'all']
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

            per_class[int(catId)] = {'AP': ap, 'AP50': ap50, 'AP75': ap75}

        return per_class

    # ------------------------------------------------------------------
    # 内部：OKS 计算与单图像评估
    # ------------------------------------------------------------------
    def _compute_oks(self, imgId, catId):
        """
        计算 OKS (Object Keypoint Similarity) 矩阵。

        对齐 pycocotools.cocoeval.COCOeval.computeOks 的逻辑：
        - 有可见关键点 (k1>0): OKS = mean(exp(-d^2 / (2*s^2*k^2))) over visible keypoints
        - 无可见关键点 (k1==0): 使用关键点到 GT bbox 边界的距离

        Returns:
            np.ndarray: OKS 矩阵 [D, G]，或空列表
        """
        p = self.params
        gts = self._gts[(imgId, catId)] if p.useCats else [g for c in p.catIds for g in self._gts[(imgId, c)]]
        dts = self._dts[(imgId, catId)] if p.useCats else [d for c in p.catIds for d in self._dts[(imgId, c)]]

        if len(gts) == 0 and len(dts) == 0:
            return []

        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[:p.maxDets[-1]]

        if len(gts) == 0 or len(dts) == 0:
            return []

        sigmas = p.kpt_sigmas
        vars_ = (sigmas * 2) ** 2
        k = len(sigmas)

        D = len(dts)
        G = len(gts)
        oks = np.zeros((D, G), dtype=np.float64)

        for j, gt in enumerate(gts):
            g_kpts = np.array(gt['keypoints'], dtype=np.float64).reshape(-1, 3)
            xg = g_kpts[:, 0]
            yg = g_kpts[:, 1]
            vg = g_kpts[:, 2]
            k1 = int(np.sum(vg > 0))
            area = gt['area']

            # 无可见关键点时使用 bbox 边界距离（用于 crowd 标注）
            if k1 == 0 and 'bbox' in gt:
                bb = gt['bbox']
                x0 = bb[0] - bb[2]
                x1 = bb[0] + bb[2] * 2
                y0 = bb[1] - bb[3]
                y1 = bb[1] + bb[3] * 2
            else:
                x0 = x1 = y0 = y1 = 0  # 不会使用

            for i, dt in enumerate(dts):
                d_kpts = np.array(dt['keypoints'], dtype=np.float64).reshape(-1, 3)
                xd = d_kpts[:, 0]
                yd = d_kpts[:, 1]

                if k1 > 0:
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # 到 bbox 边界的距离
                    z = np.zeros(k)
                    dx = np.maximum(z, x0 - xd) + np.maximum(z, xd - x1)
                    dy = np.maximum(z, y0 - yd) + np.maximum(z, yd - y1)

                e = (dx**2 + dy**2) / (vars_ * (area + np.spacing(1)) * 2)
                if k1 > 0:
                    e = e[vg > 0]
                oks[i, j] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] > 0 else 0.0

        return oks

    def _evaluate_img(self, imgId, catId, aRng, maxDet):
        """单图像/单类别/单面积桶评估（使用 OKS 进行匹配）。"""
        p = self.params
        if p.useCats:
            gt = list(self._gts[(imgId, catId)])
            dt = list(self._dts[(imgId, catId)])
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[(imgId, cId)]]
            dt = [_ for cId in p.catIds for _ in self._dts[(imgId, cId)]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        # 标记 GT 忽略（面积桶 + ignore/iscrowd + 无关键点）
        for g in gt:
            g['_ignore'] = 1 if (g.get('ignore', 0) or (g['area'] < aRng[0] or g['area'] > aRng[1])) else 0

        # 排序：gt 先非忽略，dt 按分数并裁剪
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[:maxDet]]
        iscrowd = np.array([int(o.get('iscrowd', 0)) for o in gt], dtype=np.int32)

        # OKS 矩阵（按 gt 排序重排列）
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

        # 未匹配的 dt 若面积不在当前桶，记忽略
        if D > 0:
            dt_area = np.array([d.get('area', 0.0) for d in dt], dtype=np.float64)
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
