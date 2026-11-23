"""LPCMCI (Gerhardus and Runge 2020) as tigramite 5.2 runs it for ETIA
(increment 6): ``LPCMCI(dataframe, cond_ind_test).run_lpcmci(tau_max=L,
pc_alpha=alpha)``.

The methods of ``_LPCMCI`` are derived from tigramite 5.2 (``tigramite/lpcmci.py``,
Jakob Runge and Andreas Gerhardus, GNU GPL v3.0) with an automatic
transformation (``ast``): the docstrings, the print calls and the verbosity
branches are removed, the logic is unchanged. The CI tests run on the
lagged matrix of ``pcmci.LaggedData`` with the batched tests of ETIA-JAX
(``_RunTest``), which gives the same values as tigramite's ``run_test``.
This file is therefore under the GNU GPL v3.0.
"""
from copy import deepcopy
from itertools import product, combinations

import numpy as np

from .pcmci import LaggedData, _Tester, TigResult


class _RunTest:
    """tigramite's cond_ind_test.run_test(X, Y, Z, tau_max, alpha_or_thres)
    on ETIA-JAX's tester (one variable in X and in Y)."""

    def __init__(self, tester):
        self.t = tester

    def run_test(self, X, Y, Z=None, tau_max=0, cut_off="2xtau_max", alpha_or_thres=None):
        return self.t.run(tuple(X[0]), tuple(Y[0]), [tuple(z) for z in (Z or [])], alpha_or_thres)


class _LPCMCI:
    def __init__(self, data, tester):
        self.N = data.N
        self.cond_ind_test = _RunTest(tester)
        self.var_names = list(range(data.N))
        self.verbosity = 0

    def _dict_to_matrix(self, val_dict, tau_max, n_vars, default=1):
        matrix = np.ones((n_vars, n_vars, tau_max + 1))
        matrix *= default
        for j in val_dict.keys():
            for link in val_dict[j].keys():
                k, tau = link
                if tau == 0:
                    matrix[k, j, 0] = matrix[j, k, 0] = val_dict[j][link]
                else:
                    matrix[k, j, abs(tau)] = val_dict[j][link]
        return matrix

    def run_lpcmci(self, link_assumptions=None, tau_min=0, tau_max=1, pc_alpha=0.05, n_preliminary_iterations=1, max_cond_px=0, max_p_global=np.inf, max_p_non_ancestral=np.inf, max_q_global=np.inf, max_pds_set=np.inf, prelim_with_collider_rules=True, parents_of_lagged=True, prelim_only=False, break_once_separated=True, no_non_ancestral_phase=False, use_a_pds_t_for_majority=True, orient_contemp=1, update_middle_marks=True, prelim_rules=1, fix_all_edges_before_final_orientation=True, auto_first=True, remember_only_parents=True, no_apr=0):
        self._initialize(link_assumptions, tau_min, tau_max, pc_alpha, n_preliminary_iterations, max_cond_px, max_p_global, max_p_non_ancestral, max_q_global, max_pds_set, prelim_with_collider_rules, parents_of_lagged, prelim_only, break_once_separated, no_non_ancestral_phase, use_a_pds_t_for_majority, orient_contemp, update_middle_marks, prelim_rules, fix_all_edges_before_final_orientation, auto_first, remember_only_parents, no_apr)
        for i in range(self.n_preliminary_iterations):
            self._run_ancestral_removal_phase(prelim=True)
            if i == self.n_preliminary_iterations - 1 and self.prelim_only:
                break
            def_ancs = self.def_ancs
            if self.remember_only_parents:
                smaller_def_ancs = dict()
                for j in range(self.N):
                    smaller_def_ancs[j] = {(i, lag_i) for i, lag_i in def_ancs[j] if self._get_link((i, lag_i), (j, 0)) != ''}
                def_ancs = smaller_def_ancs
            self._initialize_run_memory()
            self._apply_new_ancestral_information(None, def_ancs)
        if not self.prelim_only:
            self._run_ancestral_removal_phase()
        if not self.prelim_only and (not self.no_non_ancestral_phase):
            self._run_non_ancestral_removal_phase()
        if self.fix_all_edges_before_final_orientation:
            self._fix_all_edges()
            self._run_orientation_phase(rule_list=self._rules_all, only_lagged=False)
        self._fix_all_edges()
        self.graph = self._dict2graph()
        self.pval_max_matrix = self._dict_to_matrix(self.pval_max, self.tau_max, self.N, default=0)
        self.val_min_matrix = self._dict_to_matrix(self.pval_max_val, self.tau_max, self.N, default=0)
        self.cardinality_matrix = self._dict_to_matrix(self.pval_max_card, self.tau_max, self.N, default=0)
        return_dict = {'graph': self.graph, 'p_matrix': self.pval_max_matrix, 'val_matrix': self.val_min_matrix}
        return return_dict

    def _initialize(self, link_assumptions, tau_min, tau_max, pc_alpha, n_preliminary_iterations, max_cond_px, max_p_global, max_p_non_ancestral, max_q_global, max_pds_set, prelim_with_collider_rules, parents_of_lagged, prelim_only, break_once_separated, no_non_ancestral_phase, use_a_pds_t_for_majority, orient_contemp, update_middle_marks, prelim_rules, fix_all_edges_before_final_orientation, auto_first, remember_only_parents, no_apr):
        self.link_assumptions = link_assumptions
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.n_preliminary_iterations = n_preliminary_iterations
        self.max_cond_px = max_cond_px
        self.max_p_global = max_p_global
        self.max_p_non_ancestral = max_p_non_ancestral
        self.max_q_global = max_q_global
        self.max_pds_set = max_pds_set
        self.prelim_with_collider_rules = prelim_with_collider_rules
        self.parents_of_lagged = parents_of_lagged
        self.prelim_only = prelim_only
        self.break_once_separated = break_once_separated
        self.no_non_ancestral_phase = no_non_ancestral_phase
        self.use_a_pds_t_for_majority = use_a_pds_t_for_majority
        self.orient_contemp = orient_contemp
        self.update_middle_marks = update_middle_marks
        self.prelim_rules = prelim_rules
        self.fix_all_edges_before_final_orientation = fix_all_edges_before_final_orientation
        self.auto_first = auto_first
        self.remember_only_parents = remember_only_parents
        self.no_apr = no_apr
        if isinstance(pc_alpha, (list, tuple, np.ndarray)):
            raise ValueError('pc_alpha must be single float in LPCMCI.')
        if pc_alpha < 0.0 or pc_alpha > 1:
            raise ValueError('Choose 0 <= pc_alpha <= 1')
        self._check_tau_min_tau_max()
        if self.link_assumptions is not None:
            self._check_link_assumptions()
        self._rules_prelim_final = [['APR'], ['ER-08'], ['ER-02'], ['ER-01'], ['ER-09'], ['ER-10']]
        self._rules_prelim = [['APR'], ['ER-08'], ['ER-02'], ['ER-01']] if self.prelim_rules == 0 else self._rules_prelim_final
        self._rules_all = [['APR'], ['ER-08'], ['ER-02'], ['ER-01'], ['ER-00-d'], ['ER-00-c'], ['ER-03'], ['R-04'], ['ER-09'], ['ER-10'], ['ER-00-b'], ['ER-00-a']]
        self._initialize_run_memory()
        return True

    def _check_tau_min_tau_max(self):
        if not 0 <= self.tau_min <= self.tau_max:
            raise ValueError('tau_min = {}, '.format(self.tau_min) + 'tau_max = {}, '.format(self.tau_max) + 'but 0 <= tau_min <= tau_max required.')

    def _check_link_assumptions(self):
        ancs_mat_contemp = np.zeros((self.N, self.N), dtype='int32')
        ancs_mat = np.zeros((self.N * (self.tau_max + 1), self.N * (self.tau_max + 1)), dtype='int32')
        for j, links_j in self.link_assumptions.items():
            if not 0 <= j <= self.N - 1:
                raise ValueError(f"The argument 'link_assumption' must be a dictionary whose keys are in {{0, 1, ..., N-1}}, where N is the number of component time series. Here, N = {self.N}.")
            for (i, lag_i), link_ij in links_j.items():
                if i == j and lag_i == 0:
                    raise ValueError(f"The dictionary 'link_assumptions[{j}] must not have the key ({j}, 0), because this refers to a self-link.")
                if not 0 <= i <= self.N - 1 or not -self.tau_max <= lag_i <= -self.tau_min:
                    raise ValueError(f"All values of 'link_assumptions' must be dictionaries whose keys are of the form (i, lag_i), where i in {{0, 1, ..., N-1}} with N the number of component time series and lag_i in {{-tau_max, ..., -tau_min}} with tau_max the maximum considered time lag and tau_min the minimum assumed time lag. Here, N = {self.N} and tau_max = {self.tau_max} and tau_min = {self.tau_min}.")
                if link_ij == '':
                    if lag_i == 0:
                        if self.link_assumptions.get(i) is None or self.link_assumptions[i].get((j, 0)) is None or self.link_assumptions[i][j, 0] != '':
                            raise ValueError(f"The lag zero links specified by 'link_assumptions' must be symmetric: Because'link_assumptions'[{j}][({i}, {0})] = '',  there must also be 'link_assumptions'[{i}][({j}, {0})] = ''.")
                    continue
                if len(link_ij) != 3:
                    if lag_i < 0:
                        raise ValueError(f"Invalid link: 'link_assumptions'[{j}][({i}, {lag_i})] = {link_ij}. Allowed are: '-?>', '-->', '<?>', '<->', 'o?>', 'o->'.")
                    else:
                        raise ValueError(f"Invalid link: 'link_assumptions'[{j}][({i}, {lag_i})] = {link_ij}. Allowed are: '-?>', '-->', '<?>', '<->', 'o?>', 'o->', '<?-', '<--', '<?o', '<--', 'o-o', 'o?o'.")
                if link_ij[0] == '-':
                    if link_ij[2] != '>':
                        raise ValueError(f"Invalid link: 'link_assumptions'[{j}][({i}, {lag_i})] = {link_ij}. The first character '-', which says that ({i}, {lag_i}) is an ancestor (cause) of ({j}, 0). Hence, ({j}, 0) is a non-ancestor (non-cause) of ({i}, {lag_i}) and the third character must be '>'.")
                    if lag_i == 0:
                        ancs_mat_contemp[i, j] = 1
                    for Delta_t in range(0, self.tau_max + 1 - abs(lag_i)):
                        ancs_mat[self.N * (abs(lag_i) + Delta_t) + i, self.N * Delta_t + j] = 1
                elif link_ij[0] in ['<', 'o']:
                    if lag_i < 0:
                        if link_ij[2] != '>':
                            raise ValueError(f"Invalid link: 'link_assumptions'[{j}][({i}, {lag_i})] = {link_ij}. Since {lag_i} < 0, ({j}, 0) cannot be an ancestor (cause) of ({i}, {lag_i}). Hence, the third character must be '>'.")
                    else:
                        if link_ij[2] not in ['-', '>', 'o']:
                            raise ValueError(f"Invalid link: 'link_assumptions'[{j}][({i}, {0})] = {link_ij}. The third character must be one of the following: 1) '-', which says that ({j}, 0) is an ancestor (cause) of ({i}, {0}). 2) '>', which says that ({j}, 0) is a non-ancestor (non-cause) of ({i}, {0}). 3) 'o', which says that it is unknown whether or not ({j}, {0}) is an ancestor (cause) of ({i}, {0}).")
                        if link_ij[2] == '-':
                            if link_ij[0] != '<':
                                raise ValueError(f"Invalid link: 'link_assumptions'[{j}][({i}, {0})] = {link_ij}. The third character is '-', which says that ({j}, {0}) is an ancestor (cause) of ({i}, 0). Hence, ({i}, 0) is a non-ancestor (non-cause) of ({j}, {0}) and the first character must be '<'.")
                            ancs_mat_contemp[j, i] = 1
                            for Delta_t in range(0, self.tau_max + 1):
                                ancs_mat[self.N * Delta_t + j, self.N * Delta_t + i] = 1
                else:
                    raise ValueError(f"Invalid link: 'link_assumptions'[{j}][({i}, {lag_i})] = {link_ij}. The first character must be one of the following: 1) '-', which says that ({i}, {lag_i}) is an ancestor (cause) of ({j}, 0). 2) '<', which says that ({i}, {lag_i}) is a non-ancestor (non-cause) of ({j}, 0). 3) 'o', which says that itis unknown whether or not ({i}, {lag_i}) is an ancestor (cause) of ({j}, {0}).")
                if link_ij[1] not in ['-', '?']:
                    raise ValueError(f"Invalid link: 'link_assumptions'[{j}][({i}, {lag_i})] = {link_ij}. The second character must be one of the following: 1) '-', which says that the link ({i}, {lag_i}) {link_ij} ({j}, 0) is definitely part of the graph. 2) '?', which says that link might be but does not need to be part of the graph.")
                if lag_i == 0:
                    if self.link_assumptions.get(i) is None or self.link_assumptions[i].get((j, 0)) is None or self.link_assumptions[i][j, 0] != self._reverse_link(link_ij):
                        raise ValueError(f"The lag zero links specified by 'link_assumptions' must be symmetric: Because 'link_assumptions'[{j}][({i}, {0})] = '{link_ij}' there must also be 'link_assumptions'[{i}][({j}, {0})] = '{self._reverse_link(link_ij)}'.")
        ancs_mat_contemp_to_N = np.linalg.matrix_power(ancs_mat_contemp, self.N)
        if np.sum(ancs_mat_contemp_to_N) != 0:
            raise ValueError("According to 'link_assumptions', there is a contemporaneous causal cycle. Causal cycles are not allowed.")
        ancs_mat_summed = np.linalg.inv(np.eye(ancs_mat.shape[0], dtype='int32') - ancs_mat)
        for j, links_j in self.link_assumptions.items():
            for (i, lag_i), link_ij in links_j.items():
                if link_ij != '' and link_ij[0] == '<' and (ancs_mat_summed[self.N * abs(lag_i) + i, j] != 0):
                    raise ValueError(f"Inconsistency in 'link_assumptions': Since 'link_assumptions'[{j}][({i}, {lag_i})] = {link_ij}, variable ({i}, {lag_i}) is a non-ancestor (non-cause) of ({j}, 0). At the same time, however, 'link_assumptions' specifies a directed path (causal path) from ({i}, {lag_i}) to ({j}, 0).")
        for j in range(self.N):
            if self.link_assumptions.get(j) is None:
                self.link_assumptions[j] = {(i, -tau_i): '' for i, tau_i in product(range(self.N), range(self.tau_min, self.tau_max + 1)) if tau_i > 0 or i != j}
            else:
                for i, tau_i in product(range(self.N), range(self.tau_min, self.tau_max + 1)):
                    if tau_i > 0 or i != j:
                        if self.link_assumptions[j].get((i, -tau_i)) is None:
                            self.link_assumptions[j][i, -tau_i] = ''

    def _initialize_run_memory(self):
        self.graph_dict = {}
        for j in range(self.N):
            self.graph_dict[j] = {(i, 0): 'o?o' for i in range(self.N) if j != i}
            if self.max_cond_px == 0 and self.update_middle_marks:
                self.graph_dict[j].update({(i, -tau): 'oL>' for i in range(self.N) for tau in range(1, self.tau_max + 1)})
            else:
                self.graph_dict[j].update({(i, -tau): 'o?>' for i in range(self.N) for tau in range(1, self.tau_max + 1)})
        self.sepsets = {j: {(i, -tau): set() for i in range(self.N) for tau in range(self.tau_max + 1) if tau > 0 or i < j} for j in range(self.N)}
        self.def_ancs = {j: set() for j in range(self.N)}
        self.def_non_ancs = {j: set() for j in range(self.N)}
        self.ambiguous_ancestorships = {j: set() for j in range(self.N)}
        self.pval_max = {j: {(i, -tau): -np.inf for i in range(self.N) for tau in range(self.tau_max + 1) if tau > 0 or i < j} for j in range(self.N)}
        self.pval_max_val = {j: {(i, -tau): np.inf for i in range(self.N) for tau in range(self.tau_max + 1) if tau > 0 or i < j} for j in range(self.N)}
        self.pval_max_card = {j: {(i, -tau): -np.inf for i in range(self.N) for tau in range(self.tau_max + 1) if tau > 0 or i < j} for j in range(self.N)}
        self._na_pds_t = {(j, -tau_j): {} for j in range(self.N) for tau_j in range(self.tau_max + 1)}
        self.max_na_search_set_found = -1
        self.max_na_pds_set_found = -1
        self._apply_tau_min_restriction()
        if self.link_assumptions is not None:
            self._apply_link_assumptions()
        return True

    def _apply_tau_min_restriction(self):
        for i, j, tau in product(range(self.N), range(self.N), range(0, self.tau_min)):
            if tau > 0 or j != i:
                self.graph_dict[j][i, -tau] = ''
            if tau > 0 or i < j:
                self.pval_max[j][i, -tau] = np.inf
                self.pval_max_val[j][i, -tau] = -np.inf
                self.pval_max_card[j][i, -tau] = np.inf

    def _apply_link_assumptions(self):
        for j, links_j in self.link_assumptions.items():
            for (i, lag_i), link in self.link_assumptions[j].items():
                if link != '' and link[1] == '?' and (lag_i < 0) and (self.max_cond_px == 0) and self.update_middle_marks:
                    self.graph_dict[j][i, lag_i] = link[0] + 'L' + link[2]
                else:
                    self.graph_dict[j][i, lag_i] = link
                if link == '' and (lag_i < 0 or i < j):
                    self.pval_max[j][i, lag_i] = np.inf
                    self.pval_max_val[j][i, lag_i] = -np.inf
                    self.pval_max_card[j][i, lag_i] = np.inf

    def _run_ancestral_removal_phase(self, prelim=False):
        p_pc = 0
        while_broken = False
        while True:
            if p_pc > self.max_p_global:
                while_broken = True
                break
            has_converged = True
            any_removal = False
            if self.auto_first:
                link_list = [product(range(self.N), range(-self.tau_max, 0))]
                link_list = link_list + [product(range(self.N), range(self.N), range(-lag, -lag + 1)) for lag in range(0, self.tau_max + 1)]
            else:
                link_list = [product(range(self.N), range(self.N), range(-lag, -lag + 1)) for lag in range(0, self.tau_max + 1)]
            for links in link_list:
                to_remove = {j: {} for j in range(self.N)}
                for pair in links:
                    if len(pair) == 2:
                        X = (pair[0], pair[1])
                        Y = (pair[0], 0)
                    else:
                        X = (pair[0], pair[2])
                        Y = (pair[1], 0)
                        if self.auto_first and X[0] == Y[0]:
                            continue
                    if X[1] == 0 and X[0] == Y[0]:
                        continue
                    if self._is_smaller(Y, X):
                        continue
                    link = self._get_link(X, Y)
                    if link == '':
                        continue
                    if link[1] == '-':
                        continue
                    test_Y = True if link[1] not in ['R', '!'] else False
                    test_X = True if link[1] not in ['L', '!'] and (X[1] == 0 or (self.max_cond_px > 0 and self.max_cond_px >= p_pc)) else False
                    if test_Y:
                        S_default_YX, S_search_YX = self._get_default_and_search_sets(Y, X, 'ancestral')
                    if test_X:
                        S_default_XY, S_search_XY = self._get_default_and_search_sets(X, Y, 'ancestral')
                    any_middle_mark_update = False
                    if test_Y:
                        if len(S_search_YX) < p_pc:
                            self._apply_middle_mark(X, Y, 'R')
                        else:
                            has_converged = False
                    if test_X:
                        if len(S_search_XY) < p_pc:
                            self._apply_middle_mark(X, Y, 'L')
                        else:
                            has_converged = False
                    if self.break_once_separated or not np.isinf(self.max_q_global):
                        if test_Y:
                            S_search_YX = self._sort_search_set(S_search_YX, Y)
                        if test_X:
                            S_search_XY = self._sort_search_set(S_search_XY, X)
                    if test_Y:
                        q_count = 0
                        for S_pc in combinations(S_search_YX, p_pc):
                            q_count = q_count + 1
                            if q_count > self.max_q_global:
                                break
                            Z = set(S_pc)
                            Z = Z.union(S_default_YX)
                            val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                            self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))
                            if not dependent:
                                to_remove[Y[0]][X] = True
                                self._save_sepset(X, Y, (frozenset(Z), 'wm'))
                                if self.break_once_separated:
                                    break
                    if test_X:
                        q_count = 0
                        for S_pc in combinations(S_search_XY, p_pc):
                            q_count = q_count + 1
                            if q_count > self.max_q_global:
                                break
                            Z = set(S_pc)
                            Z = Z.union(S_default_XY)
                            val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                            self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))
                            if not dependent:
                                to_remove[Y[0]][X] = True
                                self._save_sepset(X, Y, (frozenset(Z), 'wm'))
                                if self.break_once_separated:
                                    break
                for j in range(self.N):
                    for i, lag_i in to_remove[j].keys():
                        any_removal = True
                        self._write_link((i, lag_i), (j, 0), '', verbosity=self.verbosity)
            if any_removal:
                only_lagged = False if self.orient_contemp == 2 else True
                any_update = self._run_orientation_phase(rule_list=self._rules_prelim, only_lagged=only_lagged)
                if any_update:
                    if self.max_cond_px == 0 and self.update_middle_marks:
                        self._update_middle_marks()
                    p_pc = 0
                else:
                    p_pc = p_pc + 1
            elif has_converged:
                break
            else:
                p_pc = p_pc + 1
        for j in range(self.N):
            for i, lag_i in self.graph_dict[j].keys():
                X = (i, lag_i)
                Y = (j, 0)
                if self._is_smaller(Y, X):
                    continue
                link = self._get_link((i, lag_i), (j, 0))
                if len(link) > 0:
                    if not while_broken:
                        assert link[1] != '?'
                        assert link[1] != 'L'
                        assert link[1] != 'R' or (lag_i < 0 and (self.max_cond_px > 0 or not self.update_middle_marks)) or self.no_apr != 0
                    if link[1] not in ['-', '!']:
                        self._write_link((i, lag_i), (j, 0), link[0] + '!' + link[2])
        if not prelim or self.prelim_with_collider_rules:
            if not prelim:
                self.no_apr = self.no_apr - 1
            any_update = self._run_orientation_phase(rule_list=self._rules_all, only_lagged=False)
            if self.max_cond_px == 0 and self.update_middle_marks and any_update:
                self._update_middle_marks()
        else:
            only_lagged = False if self.orient_contemp >= 1 else True
            any_update = self._run_orientation_phase(rule_list=self._rules_prelim_final, only_lagged=only_lagged)
            if self.max_cond_px == 0 and self.update_middle_marks and any_update:
                self._update_middle_marks()
        return True

    def _run_non_ancestral_removal_phase(self):
        self._update_middle_marks()
        self._initialize_full_graph()
        p_pc = 0
        while True:
            if p_pc > self.max_p_global or p_pc > self.max_p_non_ancestral:
                break
            has_converged = True
            any_removal = False
            if self.auto_first:
                link_list = [product(range(self.N), range(-self.tau_max, 0))]
                link_list = link_list + [product(range(self.N), range(self.N), range(-lag, -lag + 1)) for lag in range(0, self.tau_max + 1)]
            else:
                link_list = [product(range(self.N), range(self.N), range(-lag, -lag + 1)) for lag in range(0, self.tau_max + 1)]
            for links in link_list:
                to_remove = {j: {} for j in range(self.N)}
                for pair in links:
                    if len(pair) == 2:
                        X = (pair[0], pair[1])
                        Y = (pair[0], 0)
                    else:
                        X = (pair[0], pair[2])
                        Y = (pair[1], 0)
                        if self.auto_first and X[0] == Y[0]:
                            continue
                    if X[1] == 0 and X[0] == Y[0]:
                        continue
                    if self._is_smaller(Y, X):
                        continue
                    link = self._get_link(X, Y)
                    if link == '':
                        continue
                    if link[1] == '-':
                        continue
                    test_X = True if X[1] == 0 or (self.max_cond_px > 0 and self.max_cond_px >= p_pc) else False
                    S_default_YX, S_search_YX = self._get_default_and_search_sets(Y, X, 'non-ancestral')
                    self.max_na_search_set_found = max(self.max_na_search_set_found, len(S_search_YX))
                    if test_X:
                        S_default_XY, S_search_XY = self._get_default_and_search_sets(X, Y, 'non-ancestral')
                        self.max_na_search_set_found = max(self.max_na_search_set_found, len(S_search_XY))
                    if len(S_search_YX) > self.max_pds_set or (test_X and len(S_search_XY) > self.max_pds_set):
                        continue
                    if len(S_search_YX) < p_pc or (test_X and len(S_search_XY) < p_pc):
                        self._write_link(X, Y, link[0] + '-' + link[2], verbosity=self.verbosity)
                        continue
                    else:
                        has_converged = False
                    if self.break_once_separated or not np.isinf(self.max_q_global):
                        S_search_YX = self._sort_search_set(S_search_YX, Y)
                        if test_X:
                            S_search_XY = self._sort_search_set(S_search_XY, X)
                    q_count = 0
                    for S_pc in combinations(S_search_YX, p_pc):
                        q_count = q_count + 1
                        if q_count > self.max_q_global:
                            break
                        Z = set(S_pc)
                        Z = Z.union(S_default_YX)
                        val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                        self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))
                        if not dependent:
                            to_remove[Y[0]][X] = True
                            self._save_sepset(X, Y, (frozenset(Z), 'wm'))
                            if self.break_once_separated:
                                break
                    if test_X:
                        q_count = 0
                        for S_pc in combinations(S_search_XY, p_pc):
                            q_count = q_count + 1
                            if q_count > self.max_q_global:
                                break
                            Z = set(S_pc)
                            Z = Z.union(S_default_XY)
                            val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                            self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))
                            if not dependent:
                                to_remove[Y[0]][X] = True
                                self._save_sepset(X, Y, (frozenset(Z), 'wm'))
                                if self.break_once_separated:
                                    break
                any_removal_this = False
                for j in range(self.N):
                    for i, lag_i in to_remove[j].keys():
                        any_removal = True
                        any_removal_this = True
                        self._write_link((i, lag_i), (j, 0), '', verbosity=self.verbosity)
                if any_removal_this:
                    self._initialize_full_graph()
                    self._na_pds_t = {(j, -tau_j): {} for j in range(self.N) for tau_j in range(self.tau_max + 1)}
            if any_removal:
                any_update = self._run_orientation_phase(rule_list=self._rules_all, only_lagged=False)
                if any_update:
                    self._initialize_full_graph()
                    self._na_pds_t = {(j, -tau_j): {} for j in range(self.N) for tau_j in range(self.tau_max + 1)}
                    p_pc = 0
                else:
                    p_pc = p_pc + 1
            elif has_converged:
                break
            else:
                p_pc = p_pc + 1
        self._run_orientation_phase(rule_list=self._rules_all, only_lagged=False)
        return True

    def _run_orientation_phase(self, rule_list, only_lagged=False):
        restarted_once = False
        idx = 0
        while idx <= len(rule_list) - 1:
            if idx == 0:
                self._initialize_full_graph()
            restart = False
            current_rules = rule_list[idx]
            to_orient = []
            for rule in current_rules:
                orientations = self._apply_rule(rule, only_lagged)
                to_orient.extend(orientations)
            links_to_remove = set()
            links_to_fix = set()
            new_ancs = {j: set() for j in range(self.N)}
            new_non_ancs = {j: set() for j in range(self.N)}
            for (i, j, lag_i), new_link in to_orient:
                old_link = self._get_link((i, lag_i), (j, 0))
                if new_link == '' and len(old_link) > 0:
                    links_to_remove.add((i, j, lag_i))
                    continue
                assert not (lag_i > 0 and new_link[2] == '-')
                if new_link[1] == '-' and old_link[1] != '-':
                    links_to_fix.add((i, j, lag_i))
                if new_link[0] == '-' and old_link[0] != '-':
                    new_ancs[j].add((i, lag_i))
                elif new_link[0] == '<' and old_link[0] != '<':
                    new_non_ancs[j].add((i, lag_i))
                if lag_i == 0:
                    if new_link[2] == '-' and old_link[2] != '-':
                        new_ancs[i].add((j, 0))
                    elif new_link[2] == '>' and old_link[2] != '>':
                        new_non_ancs[i].add((j, 0))
            ambiguous_links = links_to_fix.intersection(links_to_remove)
            links_to_fix = links_to_fix.difference(ambiguous_links)
            links_to_remove = links_to_remove.difference(ambiguous_links)
            for i, j, lag_i in links_to_remove:
                self._write_link((i, lag_i), (j, 0), '', verbosity=self.verbosity)
                restart = True
            for i, j, lag_i in links_to_fix:
                old_link = self._get_link((i, lag_i), (j, 0))
                new_link = old_link[0] + '-' + old_link[2]
                self._write_link((i, lag_i), (j, 0), new_link, verbosity=self.verbosity)
                restart = True
            for i, j, lag_i in ambiguous_links:
                old_link = self._get_link((i, lag_i), (j, 0))
                new_link = old_link[0] + 'x' + old_link[2]
                self._write_link((i, lag_i), (j, 0), new_link, verbosity=self.verbosity)
            restart = restart or self._apply_new_ancestral_information(new_non_ancs, new_ancs)
            if len(links_to_remove) > 0:
                new_ancs = {j: set() for j in range(self.N)}
                new_non_ancs = {j: set() for j in range(self.N)}
                for i, j, lag_i in links_to_remove:
                    X = (i, lag_i)
                    Y = (j, 0)
                    ancs_XY = self._get_ancs([X, Y]).difference({X, Y})
                    old_sepsets_all = {Z for Z, _ in self._get_sepsets(X, Y)}
                    min_size = min({len(Z) for Z in old_sepsets_all})
                    old_sepsets_smallest = {Z for Z in old_sepsets_all if len(Z) == min_size}
                    self._delete_sepsets(X, Y)
                    self._make_sepset_weakly_minimal(X, Y, old_sepsets_smallest, ancs_XY)
                    new_sepsets = self._get_sepsets(X, Y)
            if restart:
                idx = 0
                restarted_once = True
            else:
                idx = idx + 1
        return restarted_once

    def _get_default_and_search_sets(self, A, B, phase):
        if phase == 'ancestral':
            S_raw = self._get_a_pds_t(A, B)
            S_default = self._get_parents(A, B).difference({A, B})
            S_search = S_raw.difference(S_default)
        elif phase == 'non-ancestral':
            S_raw = self._get_na_pds_t(A, B)
            self.max_na_pds_set_found = max(self.max_na_pds_set_found, len(S_raw))
            S_default = S_raw.intersection(self._get_ancs([A, B]))
            S_default = S_default.union(self._get_parents(A, B))
            S_default = S_default.difference({A, B})
            S_search = S_raw.difference(S_default)
        return (S_default, S_search)

    def _apply_new_ancestral_information(self, new_non_ancs, new_ancs):
        add_to_def_non_ancs = {j: set() for j in range(self.N)}
        add_to_def_ancs = {j: set() for j in range(self.N)}
        add_to_ambiguous_ancestorships = {j: set() for j in range(self.N)}
        put_head_or_tail = False
        if new_non_ancs is None:
            new_non_ancs = {j: set() for j in range(self.N)}
        if new_ancs is None:
            new_ancs = {j: set() for j in range(self.N)}
        for j in range(self.N):
            for i, lag_i in new_ancs[j]:
                if lag_i == 0:
                    new_non_ancs[i].add((j, 0))
        for j in range(self.N):
            for i, lag_i in new_non_ancs[j]:
                if (i, lag_i) in self.ambiguous_ancestorships[j]:
                    pass
                elif (i, lag_i) in self.def_ancs[j]:
                    add_to_ambiguous_ancestorships[j].add((i, lag_i))
                elif (i, lag_i) in new_ancs[j]:
                    add_to_ambiguous_ancestorships[j].add((i, lag_i))
                else:
                    add_to_def_non_ancs[j].add((i, lag_i))
        for j in range(self.N):
            for i, lag_i in new_ancs[j]:
                if (i, lag_i) in self.ambiguous_ancestorships[j]:
                    pass
                elif lag_i == 0 and (j, 0) in self.ambiguous_ancestorships[i]:
                    pass
                elif (i, lag_i) in self.def_non_ancs[j]:
                    add_to_ambiguous_ancestorships[j].add((i, lag_i))
                elif (i, lag_i) in new_non_ancs[j]:
                    add_to_ambiguous_ancestorships[j].add((i, lag_i))
                else:
                    add_to_def_ancs[j].add((i, lag_i))
        for j in range(self.N):
            for i, lag_i in add_to_ambiguous_ancestorships[j]:
                old_link = self._get_link((i, lag_i), (j, 0))
                if len(old_link) > 0 and old_link[0] != 'x':
                    new_link = 'x' + old_link[1] + old_link[2]
                    self._write_link((i, lag_i), (j, 0), new_link, verbosity=self.verbosity)
                self.def_ancs[j].discard((i, lag_i))
                self.def_non_ancs[j].discard((i, lag_i))
                if lag_i == 0:
                    self.def_ancs[i].discard((j, 0))
                self.ambiguous_ancestorships[j].add((i, lag_i))
        for j in range(self.N):
            for i, lag_i in add_to_def_non_ancs[j]:
                old_link = self._get_link((i, lag_i), (j, 0))
                if len(old_link) > 0 and old_link[0] != '<':
                    new_link = '<' + old_link[1] + old_link[2]
                    self._write_link((i, lag_i), (j, 0), new_link, verbosity=self.verbosity)
                    put_head_or_tail = True
                self.def_non_ancs[j].add((i, lag_i))
            for i, lag_i in add_to_def_ancs[j]:
                old_link = self._get_link((i, lag_i), (j, 0))
                if len(old_link) > 0 and (old_link[0] != '-' or old_link[2] != '>'):
                    new_link = '-' + old_link[1] + '>'
                    self._write_link((i, lag_i), (j, 0), new_link, verbosity=self.verbosity)
                    put_head_or_tail = True
                self.def_ancs[j].add((i, lag_i))
                if lag_i == 0:
                    self.def_non_ancs[i].add((j, 0))
        return put_head_or_tail

    def _apply_rule(self, rule, only_lagged):
        if rule == 'APR':
            return self._apply_APR(only_lagged)
        elif rule == 'ER-00-a':
            return self._apply_ER00a(only_lagged)
        elif rule == 'ER-00-b':
            return self._apply_ER00b(only_lagged)
        elif rule == 'ER-00-c':
            return self._apply_ER00c(only_lagged)
        elif rule == 'ER-00-d':
            return self._apply_ER00d(only_lagged)
        elif rule == 'ER-01':
            return self._apply_ER01(only_lagged)
        elif rule == 'ER-02':
            return self._apply_ER02(only_lagged)
        elif rule == 'ER-03':
            return self._apply_ER03(only_lagged)
        elif rule == 'R-04':
            return self._apply_R04(only_lagged)
        elif rule == 'ER-08':
            return self._apply_ER08(only_lagged)
        elif rule == 'ER-09':
            return self._apply_ER09(only_lagged)
        elif rule == 'ER-10':
            return self._apply_ER10(only_lagged)

    def _get_na_pds_t(self, A, B):
        var_A, lag_A = A
        var_B, lag_B = B
        assert lag_A == 0 or lag_B == 0
        memo = self._na_pds_t[A].get(B)
        if memo is not None:
            return memo
        na_pds_t_1 = {(var, lag + lag_A) for (var, lag), link in self.graph_dict[var_A].items() if len(link) > 0 and (var, lag + lag_A) != B and (lag + lag_A >= -self.tau_max) and (lag + lag_A <= 0) and (link[0] != '<') and ((var_B, lag_B - (lag + lag_A)) not in self.def_ancs[var])}
        C1_list = set()
        for (var, lag), link in self.graph_full_dict[var_A].items():
            node = (var, lag + lag_A)
            if (var, lag + lag_A) == B:
                continue
            if lag + lag_A < -self.tau_max:
                continue
            if lag + lag_A > lag_B:
                continue
            if link[0] == '-':
                continue
            if link[2] == '-':
                continue
            if (var, lag + lag_A - lag_B) in self.def_non_ancs[var_B]:
                continue
            C1_list.add(node)
        visited = set()
        start_from = {(C1, A) for C1 in C1_list}
        while start_from:
            new_start_from = set()
            new_do_not_visit = set()
            for current_node, previous_node in start_from:
                visited.add((current_node, previous_node))
                for var, lag in self.graph_full_dict[current_node[0]]:
                    next_node = (var, lag + current_node[1])
                    if next_node[1] < -self.tau_max:
                        continue
                    if next_node[1] > 0:
                        continue
                    if (next_node, current_node) in visited:
                        continue
                    if next_node == previous_node:
                        continue
                    if next_node == B:
                        continue
                    if next_node == A:
                        continue
                    link_l = self._get_link(next_node, current_node)
                    link_r = self._get_link(previous_node, current_node)
                    if link_l[2] == '-' or link_r[2] == '-':
                        continue
                    if self._get_link(next_node, previous_node) == '' and (link_l[2] == 'o' or link_r[2] == 'o'):
                        continue
                    if (var_A, lag_A - next_node[1]) in self.def_ancs[next_node[0]] or (var_B, lag_B - next_node[1]) in self.def_ancs[next_node[0]]:
                        continue
                    if (next_node[1] - lag_A > 0 or (next_node[0], next_node[1] - lag_A) in self.def_non_ancs[var_A]) and (next_node[1] - lag_B > 0 or (next_node[0], next_node[1] - lag_B) in self.def_non_ancs[var_B]):
                        continue
                    new_start_from.add((next_node, current_node))
            start_from = new_start_from
        na_pds_t_2 = {node for node, _ in visited}
        self._na_pds_t[A][B] = na_pds_t_1.union(na_pds_t_2).difference({A, B})
        return self._na_pds_t[A][B]

    def _make_sepset_weakly_minimal(self, X, Y, Z_list, ancs):
        assert len({len(Z) for Z in Z_list}) == 1
        any_weakly_minimal = False
        for Z in Z_list:
            if len(Z) <= 1 or Z.issubset(ancs):
                self._save_sepset(X, Y, (frozenset(Z), 'wm'))
                any_weakly_minimal = True
        if any_weakly_minimal:
            return None
        sepsets_next_call = set()
        for Z in Z_list:
            removable = Z.difference(ancs)
            new_sepsets = []
            val_values = []
            for A in removable:
                Z_A = [node for node in Z if node != A]
                val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=Z_A, tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                self._update_pval_val_card_dicts(X, Y, pval, val, len(Z_A))
                if not dependent:
                    new_sepsets.append(frozenset(Z_A))
                    val_values.append(val)
            if len(new_sepsets) == 0:
                self._save_sepset(X, Y, (frozenset(Z), 'wm'))
                any_weakly_minimal = True
            if not any_weakly_minimal:
                new_sepsets = [node for _, node in sorted(zip(val_values, new_sepsets), reverse=True)]
                i = -1
                while i <= len(val_values) - 2 and val_values[i + 1] == val_values[0]:
                    sepsets_next_call.add(new_sepsets[i])
                    i = i + 1
                assert i >= 0
        if not any_weakly_minimal:
            self._make_sepset_weakly_minimal(X, Y, sepsets_next_call, ancs)
        else:
            return None

    def _B_not_in_SepSet_AC(self, A, B, C):
        if C[1] < A[1] or (C[1] == A[1] and C[0] < A[0]):
            return self._B_not_in_SepSet_AC(C, B, A)
        all_sepsets = set()
        if not self.use_a_pds_t_for_majority:
            adj_A = self._get_non_future_adj([A]).difference({A, C})
            adj_C = self._get_non_future_adj([C]).difference({A, C})
        else:
            adj_A = self._get_a_pds_t(A, C).difference({A, C})
            adj_C = self._get_a_pds_t(C, A).difference({A, C})
        Z_add = self._get_parents(A, C).difference({A, C})
        search_A = adj_A.difference(Z_add)
        search_C = adj_C.difference(Z_add)
        if not np.isinf(self.max_q_global):
            search_A = self._sort_search_set(search_A, A)
            search_C = self._sort_search_set(search_C, C)
        if A[1] < C[1]:
            max_p_A = min([len(search_A), self.max_cond_px, self.max_p_global]) + 1
        else:
            max_p_A = min([len(search_A), self.max_p_global]) + 1
        search_A = [(var, lag - C[1]) for var, lag in search_A]
        search_C = [(var, lag - C[1]) for var, lag in search_C]
        Z_add = {(var, lag - C[1]) for var, lag in Z_add}
        X = (A[0], A[1] - C[1])
        Y = (C[0], 0)
        for p in range(max_p_A):
            q_count = 0
            for Z_raw in combinations(search_A, p):
                q_count = q_count + 1
                if q_count > self.max_q_global:
                    break
                Z = {node for node in Z_raw if node != X and node != Y}
                Z = Z.union(Z_add)
                val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))
                if not dependent:
                    all_sepsets.add(frozenset(Z))
        for p in range(min(len(search_C), self.max_p_global) + 1):
            q_count = 0
            for Z_raw in combinations(search_C, p):
                q_count = q_count + 1
                if q_count > self.max_q_global:
                    break
                Z = {node for node in Z_raw if node != X and node != Y}
                Z = Z.union(Z_add)
                val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))
                if not dependent:
                    all_sepsets.add(frozenset(Z))
        all_sepsets = all_sepsets.union({Z for Z, _ in self._get_sepsets(X, Y)})
        n_sepsets = len(all_sepsets)
        n_sepsets_with_B = len([1 for Z in all_sepsets if (B[0], B[1] - C[1]) in Z])
        return True if 2 * n_sepsets_with_B < n_sepsets else False

    def _B_in_SepSet_AC(self, A, B, C):
        if C[1] < A[1] or (C[1] == A[1] and C[0] < A[0]):
            return self._B_in_SepSet_AC(C, B, A)
        link_AB = self._get_link(A, B)
        link_CB = self._get_link(C, B)
        if link_AB == '' or link_CB == '' or link_AB[1] != '-' or (link_CB[1] != '-'):
            all_sepsets = {Z for Z, _ in self._get_sepsets(A, C)}
            n_sepsets = len(all_sepsets)
            n_sepsets_with_B = len([1 for Z in all_sepsets if B in Z])
            return True if 2 * n_sepsets_with_B > n_sepsets else False
        else:
            all_sepsets = set()
            if not self.use_a_pds_t_for_majority:
                adj_A = self._get_non_future_adj([A]).difference({A, C})
                adj_C = self._get_non_future_adj([C]).difference({A, C})
            else:
                adj_A = self._get_a_pds_t(A, C).difference({A, C})
                adj_C = self._get_a_pds_t(C, A).difference({A, C})
            Z_add = self._get_parents(A, C).difference({A, C})
            search_A = adj_A.difference(Z_add)
            search_C = adj_C.difference(Z_add)
            if not np.isinf(self.max_q_global):
                search_A = self._sort_search_set(search_A, A)
                search_C = self._sort_search_set(search_C, C)
            if A[1] < C[1]:
                max_p_A = min([len(search_A), self.max_cond_px, self.max_p_global]) + 1
            else:
                max_p_A = min([len(search_A), self.max_p_global]) + 1
            search_A = [(var, lag - C[1]) for var, lag in search_A]
            search_C = [(var, lag - C[1]) for var, lag in search_C]
            Z_add = {(var, lag - C[1]) for var, lag in Z_add}
            X = (A[0], A[1] - C[1])
            Y = (C[0], 0)
            for p in range(max_p_A):
                q_count = 0
                for Z_raw in combinations(search_A, p):
                    q_count = q_count + 1
                    if q_count > self.max_q_global:
                        break
                    Z = {node for node in Z_raw if node != X and node != Y}
                    Z = Z.union(Z_add)
                    val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                    self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))
                    if not dependent:
                        all_sepsets.add(frozenset(Z))
            for p in range(min(len(search_C), self.max_p_global) + 1):
                q_count = 0
                for Z_raw in combinations(search_C, p):
                    q_count = q_count + 1
                    if q_count > self.max_q_global:
                        break
                    Z = {node for node in Z_raw if node != X and node != Y}
                    Z = Z.union(Z_add)
                    val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                    self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))
                    if not dependent:
                        all_sepsets.add(frozenset(Z))
            all_sepsets = all_sepsets.union({Z for Z, _ in self._get_sepsets(X, Y)})
            n_sepsets = len(all_sepsets)
            n_sepsets_with_B = len([1 for Z in all_sepsets if (B[0], B[1] - C[1]) in Z])
            return True if 2 * n_sepsets_with_B > n_sepsets else False

    def _get_parents(self, A, B):
        if self.parents_of_lagged or A[1] == B[1]:
            out = {(var, lag + A[1]) for (var, lag), link in self.graph_dict[A[0]].items() if len(link) > 0 and link[0] == '-' and (lag + A[1] >= -self.tau_max)}
            return out.union({(var, lag + B[1]) for (var, lag), link in self.graph_dict[B[0]].items() if len(link) > 0 and link[0] == '-' and (lag + B[1] >= -self.tau_max)})
        elif A[1] < B[1]:
            return {(var, lag + B[1]) for (var, lag), link in self.graph_dict[B[0]].items() if len(link) > 0 and link[0] == '-' and (lag + B[1] >= -self.tau_max)}
        else:
            return {(var, lag + A[1]) for (var, lag), link in self.graph_dict[A[0]].items() if len(link) > 0 and link[0] == '-' and (lag + A[1] >= -self.tau_max)}

    def _apply_middle_mark(self, X, Y, char):
        old_link = self._get_link(X, Y)
        if old_link[1] == '?':
            new_link = old_link[0] + char + old_link[2]
        elif old_link[1] == 'L' and char == 'R' or (old_link[1] == 'R' and char == 'L'):
            new_link = old_link[0] + '!' + old_link[2]
        else:
            assert False
        self._write_link(X, Y, new_link, verbosity=self.verbosity)
        return True

    def _update_middle_marks(self):
        for j in range(self.N):
            for (i, lag_i), link in self.graph_dict[j].items():
                if link == '':
                    continue
                X = (i, lag_i)
                Y = (j, 0)
                link_XY = self._get_link(X, Y)
                smaller_XY = self._is_smaller(X, Y)
                if link_XY[2] == '>':
                    if link_XY[1] == '?':
                        if smaller_XY:
                            new_link = link_XY[0] + 'L>'
                        else:
                            new_link = link_XY[0] + 'R>'
                        self._write_link(X, Y, new_link, verbosity=self.verbosity)
                    elif link_XY[1] == 'R' and smaller_XY or (link_XY[1] == 'L' and (not smaller_XY)):
                        new_link = link_XY[0] + '!>'
                        self._write_link(X, Y, new_link, verbosity=self.verbosity)
                link_YX = self._get_link(Y, X)
                smaller_YX = self._is_smaller(Y, X)
                if link_YX[2] == '>':
                    if link_YX[1] == '?':
                        if smaller_YX:
                            new_link = link_YX[0] + 'L>'
                        else:
                            new_link = link_YX[0] + 'R>'
                        self._write_link(Y, X, new_link, verbosity=self.verbosity)
                    elif link_YX[1] == 'R' and smaller_YX or (link_YX[1] == 'L' and (not smaller_YX)):
                        new_link = link_YX[0] + '!>'
                        self._write_link(Y, X, new_link, verbosity=self.verbosity)

    def _is_smaller(self, X, Y):
        return X[1] < Y[1] or (X[1] == Y[1] and X[0] < Y[0])

    def _get_a_pds_t(self, A, B):
        var_A, lag_A = A
        return {(var, lag + lag_A) for (var, lag), link in self.graph_dict[var_A].items() if len(link) > 0 and (var, lag + lag_A) != B and (lag + lag_A >= -self.tau_max) and (link[0] != '<')}

    def _get_ancs(self, node_list):
        out = set()
        for A in node_list:
            var_A, lag_A = A
            out = out.union({(var, lag + lag_A) for var, lag in self.def_ancs[var_A] if lag + lag_A >= -self.tau_max})
        return out

    def _get_non_ancs(self, node_list):
        out = set()
        for A in node_list:
            var_A, lag_A = A
            out = out.union({(var, lag + lag_A) for var, lag in self.def_non_ancs[var_A] if lag + lag_A >= -self.tau_max})
        return out

    def _fix_all_edges(self):
        for j in range(self.N):
            for i, lag_i in self.graph_dict[j].keys():
                link = self._get_link((i, lag_i), (j, 0))
                if len(link) > 0:
                    new_link = link[0] + '-' + link[2]
                    self.graph_dict[j][i, lag_i] = new_link

    def _apply_APR(self, only_lagged):
        out = []
        if self.no_apr > 0:
            return out
        for j in range(self.N):
            for i, lag_i in self.graph_dict[j]:
                A = (i, lag_i)
                B = (j, 0)
                if only_lagged and lag_i == 0:
                    continue
                link_AB = self._get_link(A, B)
                if self._match_link(pattern='-!>', link=link_AB) or (self._match_link(pattern='-R>', link=link_AB) and self._is_smaller(A, B)) or (self._match_link(pattern='-L>', link=link_AB) and self._is_smaller(B, A)):
                    out.append(self._get_pair_key_and_new_link(A, B, '-->'))
        return out

    def _apply_ER01(self, only_lagged):
        out = []
        all_appropriate_triples = self._find_triples(pattern_ij='**>', pattern_jk='o*+', pattern_ik='')
        for A, B, C in all_appropriate_triples:
            if only_lagged and B[1] == C[1]:
                continue
            if self._B_in_SepSet_AC(A, B, C):
                link_BC = self._get_link(B, C)
                new_link_BC = '-' + link_BC[1] + '>'
                out.append(self._get_pair_key_and_new_link(B, C, new_link_BC))
        return out

    def _apply_ER02(self, only_lagged):
        out = []
        all_appropriate_triples = set(self._find_triples(pattern_ij='-*>', pattern_jk='**>', pattern_ik='+*o'))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='**>', pattern_jk='-*>', pattern_ik='+*o')))
        for A, B, C in all_appropriate_triples:
            if only_lagged and A[1] == C[1]:
                continue
            link_AC = self._get_link(A, C)
            new_link_AC = link_AC[0] + link_AC[1] + '>'
            out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))
        return out

    def _apply_ER03(self, only_lagged):
        out = []
        all_appropriate_quadruples = self._find_quadruples(pattern_ij='**>', pattern_jk='<**', pattern_ik='', pattern_il='+*o', pattern_jl='o*+', pattern_kl='+*o')
        for A, B, C, D in all_appropriate_quadruples:
            if only_lagged and B[1] == D[1]:
                continue
            if self._B_in_SepSet_AC(A, D, C):
                link_DB = self._get_link(D, B)
                new_link_DB = link_DB[0] + link_DB[1] + '>'
                out.append(self._get_pair_key_and_new_link(D, B, new_link_DB))
        return out

    def _apply_R04(self, only_lagged):
        out = []
        all_appropriate_triples = self._find_triples(pattern_ij='<-*', pattern_jk='o-+', pattern_ik='-->')
        for triple in all_appropriate_triples:
            W, V, Y = triple
            if only_lagged and (V[1] == Y[1] and W[1] == V[1]):
                continue
            link_WV = self._get_link(W, V)
            discriminating_paths = self._get_R4_discriminating_paths(triple, max_length=np.inf)
            for path in discriminating_paths:
                X_1 = path[-1]
                if self._B_in_SepSet_AC(X_1, V, Y):
                    out.append(self._get_pair_key_and_new_link(V, Y, '-->'))
                elif link_WV != '<-x' and self._B_not_in_SepSet_AC(X_1, V, Y):
                    out.append(self._get_pair_key_and_new_link(V, Y, '<->'))
                    if link_WV != '<->':
                        out.append(self._get_pair_key_and_new_link(W, V, '<->'))
        return out

    def _apply_ER08(self, only_lagged):
        out = []
        all_appropriate_triples = self._find_triples(pattern_ij='-*>', pattern_jk='-*>', pattern_ik='o*+')
        for A, B, C in all_appropriate_triples:
            if only_lagged and A[1] == C[1]:
                continue
            link_AC = self._get_link(A, C)
            new_link_AC = '-' + link_AC[1] + '>'
            out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))
        return out

    def _apply_ER09(self, only_lagged):
        out = []
        all_appropriate_triples = set(self._find_triples(pattern_ij='o*o', pattern_jk='o*>', pattern_ik=''))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='<*o', pattern_jk='o*>', pattern_ik='')))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='<*-', pattern_jk='o*>', pattern_ik='')))
        for B_1, A, C in all_appropriate_triples:
            if only_lagged and A[1] == C[1]:
                continue
            if not self._B_in_SepSet_AC(B_1, A, C):
                continue
            link_AC = self._get_link(A, C)
            new_link_AC = '-' + link_AC[1] + '>'
            pair_key, new_link = self._get_pair_key_and_new_link(A, C, new_link_AC)
            first_link = self._get_link(A, B_1)
            if self._match_link(pattern='o*o', link=first_link):
                initial_allowed_patterns = ['-*>', 'o*>', 'o*o']
            elif self._match_link(pattern='o*>', link=first_link) or self._match_link(pattern='-*>', link=first_link):
                initial_allowed_patterns = ['-*>']
            uncovered_pd_paths = self._get_potentially_directed_uncovered_paths(B_1, C, initial_allowed_patterns)
            for upd_path in uncovered_pd_paths:
                if len(upd_path) < 3 or A in upd_path or self._get_link(A, upd_path[1]) != '':
                    continue
                if first_link[2] == '>':
                    out.append((pair_key, new_link))
                    break
                if not self._B_in_SepSet_AC(A, B_1, upd_path[1]):
                    continue
                path_qualifies = True
                for i in range(len(upd_path) - 2):
                    left_link = self._get_link(upd_path[i], upd_path[i + 1])
                    if left_link[2] == '>':
                        break
                    if not self._B_in_SepSet_AC(upd_path[i], upd_path[i + 1], upd_path[i + 2]):
                        path_qualifies = False
                        break
                if path_qualifies:
                    out.append((pair_key, new_link))
                    break
        return out

    def _apply_ER10(self, only_lagged):
        out = []
        all_appropriate_triples = set(self._find_triples(pattern_ij='o*>', pattern_jk='<*-', pattern_ik=''))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='o*>', pattern_jk='<*-', pattern_ik='***')))
        triple_sorting_dict = {}
        for A, C, P_C in all_appropriate_triples:
            if triple_sorting_dict.get((A, C)) is None:
                triple_sorting_dict[A, C] = [P_C]
            else:
                triple_sorting_dict[A, C].append(P_C)
        for A, C in triple_sorting_dict.keys():
            if only_lagged and A[1] == C[1]:
                continue
            relevant_paths = []
            for P_C in triple_sorting_dict[A, C]:
                for upd_path in self._get_potentially_directed_uncovered_paths(A, P_C, ['-*>', 'o*>', 'o*o']):
                    if len(upd_path) < 3 or C in upd_path or self._get_link(upd_path[-2], C) != '':
                        continue
                    upd_path.append(C)
                    path_qualifies = True
                    for i in range(len(upd_path) - 2):
                        left_link = self._get_link(upd_path[i], upd_path[i + 1])
                        if left_link[2] == '>':
                            break
                        if not self._B_in_SepSet_AC(upd_path[i], upd_path[i + 1], upd_path[i + 2]):
                            path_qualifies = False
                            break
                    if path_qualifies:
                        relevant_paths.append(upd_path)
            second_nodes = list({path[1] for path in relevant_paths})
            for i, j in product(range(len(second_nodes)), range(len(second_nodes))):
                if i < j and self._get_link(second_nodes[i], second_nodes[j]) == '' and self._B_in_SepSet_AC(second_nodes[i], A, second_nodes[j]):
                    link_AC = self._get_link(A, C)
                    new_link_AC = '-' + link_AC[1] + '>'
                    out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))
                    break
        return out

    def _apply_ER00a(self, only_lagged):
        out = []
        all_appropriate_triples = self._find_triples(pattern_ij='***', pattern_jk='***', pattern_ik='')
        for A, B, C in all_appropriate_triples:
            i, lag_i = A
            j, lag_j = B
            k, lag_k = C
            if only_lagged and (A[1] == B[1] or B[1] == C[1]):
                continue
            sepsets = self._get_sepsets(A, C)
            sepsets = {Z for Z, status in sepsets if status == 'wm'}
            remove_AB = False
            link_AB = self._get_link(A, B)
            if link_AB[1] not in ['-', 'x']:
                Z_add = self._get_parents(A, B).difference({A, B})
                if lag_i <= lag_j:
                    X = (i, lag_i - lag_j)
                    Y = (j, 0)
                    delta_lag = lag_j
                else:
                    X = (j, lag_j - lag_i)
                    Y = (i, 0)
                    delta_lag = lag_i
                for Z in sepsets:
                    Z_test = Z.union(Z_add).difference({A, B})
                    Z_test = {(var, lag - delta_lag) for var, lag in Z_test if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}
                    Z_add2 = {(var, lag - delta_lag) for var, lag in Z_add.difference({A, B}) if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}
                    val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z_test), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                    self._update_pval_val_card_dicts(X, Y, pval, val, len(Z_test))
                    if not dependent:
                        remove_AB = True
                        self._save_sepset(X, Y, (frozenset(Z_test), 'nwm'))
                if remove_AB:
                    pair_key, new_link = self._get_pair_key_and_new_link(A, B, '')
                    out.append((pair_key, new_link))
            remove_CB = False
            link_CB = self._get_link(C, B)
            if link_CB[1] not in ['-', 'x']:
                Z_add = self._get_parents(C, B).difference({C, B})
                if lag_k <= lag_j:
                    X = (k, lag_k - lag_j)
                    Y = (j, 0)
                    delta_lag = lag_j
                else:
                    X = (j, lag_j - lag_k)
                    Y = (k, 0)
                    delta_lag = lag_k
                for Z in sepsets:
                    Z_test = Z.union(Z_add).difference({C, B})
                    Z_test = {(var, lag - delta_lag) for var, lag in Z_test if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}
                    Z_add2 = {(var, lag - delta_lag) for var, lag in Z_add.difference({A, B}) if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}
                    val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z_test), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                    self._update_pval_val_card_dicts(X, Y, pval, val, len(Z_test))
                    if not dependent:
                        remove_CB = True
                        self._save_sepset(X, Y, (frozenset(Z_test), 'nwm'))
                if remove_CB:
                    pair_key, new_link = self._get_pair_key_and_new_link(C, B, '')
                    out.append((pair_key, new_link))
            if remove_AB or remove_CB or link_AB[2] in ['-', 'x'] or (link_CB[2] in ['-', 'x']) or (link_AB[1] == 'x') or (link_CB[1] == 'x') or (link_AB[2] == '>' and link_CB[2] == '>'):
                continue
            if self._B_not_in_SepSet_AC(A, B, C):
                if link_AB[2] != '>':
                    new_link_AB = link_AB[0] + link_AB[1] + '>'
                    out.append(self._get_pair_key_and_new_link(A, B, new_link_AB))
                new_link_CB = link_CB[0] + link_CB[1] + '>'
                if link_CB[2] != '>':
                    out.append(self._get_pair_key_and_new_link(C, B, new_link_CB))
        return out

    def _apply_ER00b(self, only_lagged):
        out = []
        triples_1 = self._find_triples(pattern_ij='**>', pattern_jk='o!+', pattern_ik='')
        triples_2 = [trip for trip in self._find_triples(pattern_ij='**>', pattern_jk='oR+', pattern_ik='') if self._is_smaller(trip[1], trip[2])]
        triples_3 = [trip for trip in self._find_triples(pattern_ij='**>', pattern_jk='oL+', pattern_ik='') if self._is_smaller(trip[2], trip[1])]
        all_appropriate_triples = set(triples_1).union(set(triples_2), set(triples_3))
        for A, B, C in all_appropriate_triples:
            i, lag_i = A
            j, lag_j = B
            k, lag_k = C
            if only_lagged and A[1] == B[1]:
                continue
            sepsets = self._get_sepsets(A, C)
            sepsets = {Z for Z, status in sepsets if status == 'wm'}
            remove_AB = False
            link_AB = self._get_link(A, B)
            if link_AB[1] not in ['-', 'x']:
                Z_add = self._get_parents(A, B).difference({A, B})
                if lag_i <= lag_j:
                    X = (i, lag_i - lag_j)
                    Y = (j, 0)
                    delta_lag = lag_j
                else:
                    X = (j, lag_j - lag_i)
                    Y = (i, 0)
                    delta_lag = lag_i
                for Z in sepsets:
                    Z_test = Z.union(Z_add).difference({A, B})
                    Z_test = {(var, lag - delta_lag) for var, lag in Z_test if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}
                    Z_add2 = {(var, lag - delta_lag) for var, lag in Z_add.difference({A, B}) if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}
                    val, pval, dependent = self.cond_ind_test.run_test(X=[X], Y=[Y], Z=list(Z_test), tau_max=self.tau_max, alpha_or_thres=self.pc_alpha)
                    self._update_pval_val_card_dicts(X, Y, pval, val, len(Z_test))
                    if not dependent:
                        remove_AB = True
                        self._save_sepset(X, Y, (frozenset(Z_test), 'nwm'))
                if remove_AB:
                    pair_key, new_link = self._get_pair_key_and_new_link(A, B, '')
                    out.append((pair_key, new_link))
            if only_lagged and B[1] == C[1]:
                continue
            if remove_AB or link_AB[1] == 'x':
                continue
            if self._B_not_in_SepSet_AC(A, B, C):
                link_CB = self._get_link(C, B)
                new_link_CB = link_CB[0] + link_CB[1] + '>'
                out.append(self._get_pair_key_and_new_link(C, B, new_link_CB))
        return out

    def _apply_ER00c(self, only_lagged):
        out = []
        triples_1 = self._find_triples(pattern_ij='*-*', pattern_jk='o!+', pattern_ik='')
        triples_2 = [trip for trip in self._find_triples(pattern_ij='*-*', pattern_jk='oR+', pattern_ik='') if self._is_smaller(trip[1], trip[2])]
        triples_3 = [trip for trip in self._find_triples(pattern_ij='*-*', pattern_jk='oL+', pattern_ik='') if self._is_smaller(trip[2], trip[1])]
        all_appropriate_triples = set(triples_1).union(set(triples_2), set(triples_3))
        for A, B, C in all_appropriate_triples:
            if only_lagged and B[1] == C[1]:
                continue
            if self._B_not_in_SepSet_AC(A, B, C):
                link_CB = self._get_link(C, B)
                new_link_CB = link_CB[0] + link_CB[1] + '>'
                out.append(self._get_pair_key_and_new_link(C, B, new_link_CB))
        return out

    def _apply_ER00d(self, only_lagged):
        out = []
        triples_1 = self._find_triples(pattern_ij='*-o', pattern_jk='o-*', pattern_ik='')
        triples_2 = self._find_triples(pattern_ij='*->', pattern_jk='o-*', pattern_ik='')
        all_appropriate_triples = set(triples_1).union(set(triples_2))
        for A, B, C in all_appropriate_triples:
            if only_lagged and (A[1] == B[1] and B[1] == C[1]):
                continue
            if self._B_not_in_SepSet_AC(A, B, C):
                if not only_lagged or B[1] != C[1]:
                    link_CB = self._get_link(C, B)
                    new_link_CB = link_CB[0] + link_CB[1] + '>'
                    out.append(self._get_pair_key_and_new_link(C, B, new_link_CB))
                link_AB = self._get_link(A, B)
                if (not only_lagged or A[1] != B[1]) and link_AB[2] == 'o':
                    new_link_AB = link_AB[0] + link_AB[1] + '>'
                    out.append(self._get_pair_key_and_new_link(A, B, new_link_AB))
        return out

    def _get_link(self, A, B):
        var_A, lag_A = A
        var_B, lag_B = B
        if abs(lag_A - lag_B) > self.tau_max:
            return ''
        elif lag_A <= lag_B:
            return self.graph_dict[var_B][var_A, lag_A - lag_B]
        else:
            return self._reverse_link(self.graph_dict[var_A][var_B, lag_B - lag_A])

    def _get_non_future_adj(self, node_list):
        out = set()
        for A in node_list:
            var_A, lag_A = A
            out = out.union({(var, lag + lag_A) for (var, lag), link in self.graph_dict[var_A].items() if len(link) > 0 and lag + lag_A >= -self.tau_max})
        return out

    def _update_pval_val_card_dicts(self, X, Y, pval, val, card):
        if X[1] < 0 or X[0] < Y[0]:
            if pval > self.pval_max[Y[0]][X]:
                self.pval_max[Y[0]][X] = pval
                self.pval_max_val[Y[0]][X] = val
                self.pval_max_card[Y[0]][X] = card
        elif pval > self.pval_max[X[0]][Y]:
            self.pval_max[X[0]][Y] = pval
            self.pval_max_val[X[0]][Y] = val
            self.pval_max_card[X[0]][Y] = card

    def _save_sepset(self, X, Y, Z):
        i, lag_i = X
        j, lag_j = Y
        assert lag_j == 0
        if lag_i < 0 or i < j:
            self.sepsets[j][X].add(Z)
        else:
            self.sepsets[i][Y].add(Z)

    def _reverse_link(self, link):
        if link == '':
            return ''
        if link[2] == '>':
            left_mark = '<'
        else:
            left_mark = link[2]
        if link[0] == '<':
            right_mark = '>'
        else:
            right_mark = link[0]
        return left_mark + link[1] + right_mark

    def _write_link(self, A, B, new_link, verbosity=0):
        var_A, lag_A = A
        var_B, lag_B = B
        if lag_A < lag_B:
            self.graph_dict[var_B][var_A, lag_A - lag_B] = new_link
        elif lag_A == lag_B:
            self.graph_dict[var_B][var_A, 0] = new_link
            self.graph_dict[var_A][var_B, 0] = self._reverse_link(new_link)
        else:
            self.graph_dict[var_A][var_B, lag_B - lag_A] = self._reverse_link(new_link)

    def _get_sepsets(self, A, B):
        var_A, lag_A = A
        var_B, lag_B = B

        def _shift(Z, lag_B):
            return frozenset([(var, lag + lag_B) for var, lag in Z])
        if lag_A < lag_B:
            out = {(_shift(Z, lag_B), status) for Z, status in self.sepsets[var_B][var_A, lag_A - lag_B]}
        elif lag_A > lag_B:
            out = {(_shift(Z, lag_A), status) for Z, status in self.sepsets[var_A][var_B, lag_B - lag_A]}
        else:
            out = {(_shift(Z, lag_A), status) for Z, status in self.sepsets[max(var_A, var_B)][min(var_A, var_B), 0]}
        return out

    def _initialize_full_graph(self):
        self.graph_full_dict = {j: {} for j in range(self.N)}
        for j in range(self.N):
            for (var, lag), link in self.graph_dict[j].items():
                if link != '':
                    self.graph_full_dict[j][var, lag] = link
                    if lag < 0:
                        self.graph_full_dict[var][j, -lag] = self._reverse_link(link)
        return None

    def _get_pair_key_and_new_link(self, A, B, link_AB):
        var_A, lag_A = A
        var_B, lag_B = B
        if lag_A <= lag_B:
            return ((var_A, var_B, lag_A - lag_B), link_AB)
        elif lag_A > lag_B:
            return ((var_B, var_A, lag_B - lag_A), self._reverse_link(link_AB))

    def _match_link(self, pattern, link):
        if pattern == '' or link == '':
            return True if pattern == link else False
        else:
            left_mark, middle_mark, right_mark = pattern
            if left_mark != '*':
                if left_mark == '+':
                    if link[0] not in ['<', 'o']:
                        return False
                elif link[0] != left_mark:
                    return False
            if right_mark != '*':
                if right_mark == '+':
                    if link[2] not in ['>', 'o']:
                        return False
                elif link[2] != right_mark:
                    return False
            if middle_mark != '*' and link[1] != middle_mark:
                return False
            return True

    def _dict2graph(self):
        graph = np.zeros((self.N, self.N, self.tau_max + 1), dtype='U3')
        for j in range(self.N):
            for adj in self.graph_dict[j]:
                i, lag_i = adj
                graph[i, j, abs(lag_i)] = self.graph_dict[j][adj]
        return graph

    def _find_adj(self, graph, node, patterns, exclude=None, ignore_time_bounds=True):
        i, lag_i = node
        if exclude is None:
            exclude = []
        if type(patterns) == str:
            patterns = [patterns]
        adj = []
        for k, lag_ik in zip(*np.where(graph[i, :, :])):
            matches = [self._match_link(patt, graph[i, k, lag_ik]) for patt in patterns]
            if np.any(matches):
                match = (k, lag_i + lag_ik)
                if match not in adj and (k, lag_i + lag_ik) not in exclude and (-self.tau_max <= lag_i + lag_ik <= 0 or ignore_time_bounds):
                    adj.append(match)
        for k, lag_ki in zip(*np.where(graph[:, i, :])):
            matches = [self._match_link(self._reverse_link(patt), graph[k, i, lag_ki]) for patt in patterns]
            if np.any(matches):
                match = (k, lag_i - lag_ki)
                if match not in adj and (k, lag_i - lag_ki) not in exclude and (-self.tau_max <= lag_i - lag_ki <= 0 or ignore_time_bounds):
                    adj.append(match)
        return adj

    def _is_match(self, graph, X, Y, pattern_ij):
        i, lag_i = X
        j, lag_j = Y
        tauij = lag_j - lag_i
        if abs(tauij) >= graph.shape[2]:
            return False
        return tauij >= 0 and self._match_link(pattern_ij, graph[i, j, tauij]) or (tauij < 0 and self._match_link(self._reverse_link(pattern_ij), graph[j, i, abs(tauij)]))

    def _find_triples(self, pattern_ij, pattern_jk, pattern_ik):
        graph = self._dict2graph()
        matched_triples = []
        for i in range(self.N):
            lag_i = 0
            adjacencies_i = self._find_adj(graph, (i, lag_i), pattern_ij)
            for j, lag_j in adjacencies_i:
                adjacencies_j = self._find_adj(graph, (j, lag_j), pattern_jk, exclude=[(i, lag_i)])
                for k, lag_k in adjacencies_j:
                    if self._is_match(graph, (i, lag_i), (k, lag_k), pattern_ik):
                        righmost_lag = max(lag_i, lag_j, lag_k)
                        match = ((i, lag_i - righmost_lag), (j, lag_j - righmost_lag), (k, lag_k - righmost_lag))
                        largest_lag = min(lag_i - righmost_lag, lag_j - righmost_lag, lag_k - righmost_lag)
                        if match not in matched_triples and -self.tau_max <= largest_lag <= 0:
                            matched_triples.append(match)
        return matched_triples

    def _find_quadruples(self, pattern_ij, pattern_jk, pattern_ik, pattern_il, pattern_jl, pattern_kl):
        assert pattern_il != ''
        graph = self._dict2graph()
        matched_quadruples = []
        ijk_triples = self._find_triples(pattern_ij, pattern_jk, pattern_ik)
        for triple in ijk_triples:
            (i, lag_i), (j, lag_j), (k, lag_k) = triple
            adjacencies = set(self._find_adj(graph, (i, lag_i), pattern_il, exclude=[(j, lag_j), (k, lag_k)]))
            if pattern_jl != '':
                adjacencies = adjacencies.intersection(set(self._find_adj(graph, (j, lag_j), pattern_jl, exclude=[(i, lag_i), (k, lag_k)])))
            else:
                adjacencies = set([adj for adj in adjacencies if self._is_match(graph, (j, lag_j), adj, '')])
            if pattern_kl != '':
                adjacencies = adjacencies.intersection(set(self._find_adj(graph, (k, lag_k), pattern_kl, exclude=[(i, lag_i), (j, lag_j)])))
            else:
                adjacencies = set([adj for adj in adjacencies if self._is_match(graph, (k, lag_k), adj, '')])
            for adj in adjacencies:
                l, lag_l = adj
                righmost_lag = max(lag_i, lag_j, lag_k, lag_l)
                match = ((i, lag_i - righmost_lag), (j, lag_j - righmost_lag), (k, lag_k - righmost_lag), (l, lag_l - righmost_lag))
                largest_lag = min(lag_i - righmost_lag, lag_j - righmost_lag, lag_k - righmost_lag, lag_l - righmost_lag)
                if match not in matched_quadruples and -self.tau_max <= largest_lag <= 0:
                    matched_quadruples.append(match)
        return matched_quadruples

    def _get_R4_discriminating_paths(self, triple, max_length=np.inf):

        def _search(path_taken, max_length):
            last_node = path_taken[-1]
            link_to_Y = self._get_link(last_node, path_taken[0])
            if len(path_taken) > 3 and link_to_Y == '':
                return [path_taken]
            paths = []
            if self._get_link(last_node, path_taken[-2])[0] == '<' and link_to_Y == '-->' and (len(path_taken) < max_length):
                for var, lag in self.graph_full_dict[last_node[0]].keys():
                    next_node = (var, lag + last_node[1])
                    next_link = self._get_link(next_node, last_node)
                    if next_node[1] <= 0 and next_node[1] >= -self.tau_max and (next_node not in path_taken) and self._match_link('*->', next_link):
                        paths.extend(_search(path_taken[:] + [next_node], max_length))
            return paths
        W, V, Y = triple
        return _search([Y, V, W], max_length)

    def _get_potentially_directed_uncovered_paths(self, start_node, end_node, initial_allowed_patterns):
        assert start_node != end_node

        def _search(end_node, path_taken, allowed_patterns):
            paths = []
            start_node = path_taken[-1]
            if start_node == end_node:
                paths.append(path_taken)
            else:
                for var, lag in self.graph_full_dict[start_node[0]].keys():
                    next_node = (var, lag + start_node[1])
                    if next_node[1] < -self.tau_max or next_node[1] > 0:
                        continue
                    if next_node in path_taken:
                        continue
                    if len(path_taken) >= 2 and self._get_link(path_taken[-2], next_node) != '':
                        continue
                    link = self._get_link(start_node, next_node)
                    if not any([self._match_link(pattern=pattern, link=link) for pattern in allowed_patterns]):
                        continue
                    if self._match_link(pattern='o*o', link=link):
                        new_allowed_patters = ['o*o', 'o*>', '-*>']
                    elif self._match_link(pattern='o*>', link=link) or self._match_link(pattern='-*>', link=link):
                        new_allowed_patters = ['-*>']
                    new_path_taken = path_taken[:] + [next_node]
                    paths.extend(_search(end_node, new_path_taken, new_allowed_patters))
            return paths
        paths = _search(end_node, [start_node], initial_allowed_patterns)
        return [path for path in paths if len(path) > 2]

    def _sort_search_set(self, search_set, reference_node):
        sort_by_potential_minus_infs = [self._get_pval_max_val(node, reference_node) for node in search_set]
        sort_by = [np.abs(value) if value != -np.inf else 0 for value in sort_by_potential_minus_infs]
        return [x for _, x in sorted(zip(sort_by, search_set), reverse=True)]

    def _get_pval_max_val(self, X, Y):
        if X[1] < 0 or X[0] < Y[0]:
            return self.pval_max_val[Y[0]][X]
        else:
            return self.pval_max_val[X[0]][Y]

    def _delete_sepsets(self, X, Y):
        i, lag_i = X
        j, lag_j = Y
        assert lag_j == 0
        if lag_i < 0 or i < j:
            self.sepsets[j][X] = set()
        else:
            self.sepsets[i][Y] = set()


def lpcmci(X, test="ParCor", tau_max=1, pc_alpha=0.05, kinds=None):
    """LPCMCI on the raw series X (T x N), with tigramite's defaults."""
    d = LaggedData(X, tau_max, kinds)
    tester = _Tester(d, test, pc_alpha)
    alg = _LPCMCI(d, tester)
    out = alg.run_lpcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    return TigResult(out["graph"], out["p_matrix"], out["val_matrix"], None, tester.n_tests)
