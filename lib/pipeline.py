#coding:UTF-8

"""
organization of candidates and optimized results
"""

from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from .rc import *

class OPTResult:
    def __init__(self, best_n, root_dirs, cell_idxs, generate_ITER_FUNCs, iter_pss, dt=DT):
        self.dt = dt
        self.best_n = best_n
        self.root_dirs = root_dirs
        self.cell_idxs = cell_idxs
        self.generate_ITER_FUNCs = generate_ITER_FUNCs
        self.iter_pss = iter_pss
        self.all_subfolders = self.get_all_subfolders()
        self.all_results = self.get_all_results()
        self.best_results = self.get_best_results()
        self.models = self.get_models()

    def get_all_subfolders(self):
        all_subfolders = {}
        for idx in self.cell_idxs:
            all_subfolders[idx] = set()
        for root_dir in self.root_dirs:
            for root, dirs, files in os.walk(root_dir):
                is_target = False
                for dir in dirs:
                    if "_res" == dir[-4:]:
                        is_target = True
                if is_target:
                    try:
                        cell_idx = int(os.path.split(root)[-1])
                        if cell_idx not in all_subfolders:
                            continue
                        all_subfolders[cell_idx].add(root)
                    except:
                        continue
        for cell_idx in all_subfolders:
            all_subfolders[cell_idx] = sorted(list(all_subfolders[cell_idx]))
        return all_subfolders

    def get_iter_models_func(self, mode=None):
        def func():
            for cell_idx in self.cell_idxs:
                for generate_ITER_FUNC, iter_ps in zip(self.generate_ITER_FUNCs, self.iter_pss):
                    for iter_func in generate_ITER_FUNC(cell_idx, **iter_ps):
                        for item in iter_func():
                            MODEL, x0, ext_ps, label = tuple(item)
                            model = MODEL(x0,**ext_ps)
                            type_name = "%s_%s"%(model.name, label)
                            if mode == 'best':
                                if type_name not in self.best_results[cell_idx]:
                                    continue
                            yield cell_idx, MODEL, x0, ext_ps, label, type_name
        return func

    def get_all_results(self):
        all_results = {}
        for cell_idx, MODEL, x0, ext_ps, label, type_name in self.get_iter_models_func()():
            if cell_idx not in all_results:
                all_results[cell_idx] = {}
            if type_name not in all_results[cell_idx]:
                all_results[cell_idx][type_name] = {}

            for root_dir in self.all_subfolders[cell_idx]:
                if len(glob(os.path.join(root_dir, os.path.join("*_res", "%s.txt"%type_name)))) == 0:
                    continue
                res = []
                for res_folder in sorted(glob(os.path.join(root_dir, "*_res"))):
                    data = np.loadtxt(os.path.join(res_folder, "%s.txt"%type_name))
                    if len(data.shape) == 1:
                        data = data.reshape((-1, 1))
                    res.append(data)
                best_indices = np.argsort(res[-1][:, 0])[:self.best_n]
                res = [i[best_indices] for i in res]
                all_results[cell_idx][type_name][root_dir] = res
        return all_results

    def get_best_results(self):
        best_results = {}    
        for cell_idx in self.cell_idxs:
            best_results[cell_idx] = {}
            for type_name in self.all_results[cell_idx]:
                if len(self.all_results[cell_idx][type_name]) == 0:
                    continue
                back_res = list(self.all_results[cell_idx][type_name].values())
                back_lens = np.array([len(i[0]) for i in back_res])
                all_values = np.concatenate(tuple(i[-1][:, 0] for i in back_res)).flatten()
                best_results[cell_idx][type_name] = np.zeros((min(self.best_n, len(all_values)), back_res[0][0].shape[1]))
                for i_best, best_index in enumerate(np.argsort(all_values)[:self.best_n]):
                    i_dir = 0
                    while back_lens[i_dir] <= best_index:
                        best_index -= back_lens[i_dir]
                        i_dir += 1                    
                    best_results[cell_idx][type_name][i_best] = back_res[i_dir][-1][best_index]
        return best_results

    def get_models(self):
        models = {}
        for cell_idx, MODEL, x0, ext_ps, label, type_name in self.get_iter_models_func(mode="best")():
            if cell_idx not in models:
                models[cell_idx] = {}
            ms = []
            ok = []
            for i in range(self.best_results[cell_idx][type_name].shape[0]):
                try:
                    ps = self.best_results[cell_idx][type_name][i, 1:]
                    model = MODEL(ps, dt=self.dt, **ext_ps)
                    ms.append(model)
                    ok.append(i)
                except Exception:
                    continue
            self.best_results[cell_idx][type_name] = self.best_results[cell_idx][type_name][ok]
            models[cell_idx][type_name] = {"PARAMS": self.best_results[cell_idx][type_name][:, 1:],
                    "LOSS": self.best_results[cell_idx][type_name][:, 0],
                    "MODEL": ms}
        return models

