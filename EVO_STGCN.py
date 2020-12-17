import argparse
from ExperimentDataLogger import *
from Env import *
import numpy as np
from copy import *
import wandb
import geatpy as ea
import shutil


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False


class MyProblem(ea.Problem):
    def __init__(self, env, logger):
        name = 'Auto-STGCN'
        maxormins = [-1]
        Dim = 24
        # [LF, BS, ILR, OF, SIPM, TIPM, FES,PBIndex, SIPM, TIPM, FES,PBIndex, SIPM, TIPM,FES,PBIndex,SIPM, TIPM, FES,PBIndex,IS,OS,FI,MBOF]
        varTypes = np.array([1] * Dim)
        lb = [1, 1, 1, 1] + [1, 1, 1, 0] + [1, 1, 1, -1] * 3 + [1, 1, 1, 1]
        assert len(lb) == Dim
        lbin = [1] * Dim
        ub = [2, 3, 3, 3] + [4, 3, 4, 0] + [4, 3, 4, 1] + [4, 3, 4, 2] + [4, 3, 4, 3] + [2, 2, 3, 2]
        assert len(ub) == Dim
        ubin = [1] * Dim
        self.env = env
        self.logger = logger
        ea.Problem.__init__(self, name, 1, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        vars = pop.Phen
        self.logger.plus_episode()
        rewards = []
        vars = vars.astype(np.int)
        for i in range(vars.shape[0]):
            self.env.reset()
            var = vars[i, :]
            action_1 = var[0:4]
            action_2 = var[-4:]
            blocks = []
            blocks.append(var[4:8])
            blocks.append(var[8:12])
            blocks.append(var[12:16])
            blocks.append(var[16:20])
            self.env.step(action_1)
            self.env.step(action_2)
            reward = 0
            for block in blocks:
                flag = False
                if block[-1] == -1:
                    block = np.array([-1, -1, -1, -1])
                    flag = True
                _, reward, _, _ = self.env.step(block)
                if flag:
                    break
            blocks.insert(0, action_1)
            blocks.insert(1, action_2)
            self.logger(action=list(map(lambda x: x.tolist(), blocks)), reward=reward)
            rewards.append(reward)
            self.logger.flush_log()
        pop.ObjV = np.array(rewards).reshape(vars.shape[0], 1)
        max_reward = np.max(np.array(rewards))
        self.logger.append_log_file(f'ObjV:{pop.ObjV}')
        wandb.log({'reward': max_reward})
        # if not isinstance(pop.ObjV, np.ndarray):
        #     raise RuntimeError('error: 目标函数值矩阵ObjV不是numpy array')
        # elif pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or pop.ObjV.shape[1] != self.problem.M:
        #     raise RuntimeError(f'error: 目标函数值矩阵ObjV的shape错误，为{pop.ObjV.shape}, ndim应为2, shape[0]应为{pop.sizes}, shape[1]应为{self.problem.M}')


def ea_select_model(config, log_name):
    #####################
    # set up parameters  #
    ######################
    if isinstance(config['ctx'], list):
        ctx = [mx.gpu(i) for i in config['ctx']]
    elif isinstance(config['ctx'], int):
        ctx = mx.gpu(config['ctx'])
    else:
        raise Exception("config_ctx error:" + str(config['ctx']))
    logger = Logger(log_name, config, False)

    #######################
    # init Env #
    #######################
    env = GNNEnv(config, ctx, logger)

    ##############
    #  training  #
    ##############
    problem = MyProblem(env, logger)
    Encoding = 'RI'
    NIND = 50
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = 39
    myAlgorithm.logTras = 1
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 1
    [NDSet, population] = myAlgorithm.run()
    NDSet.save()
    """==================================输出结果=============================="""
    print('用时：%f 秒' % myAlgorithm.passTime)
    print('评价次数：%d 次' % myAlgorithm.evalsNum)
    print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
    if myAlgorithm.log is not None and NDSet.sizes != 0:
        print('eval', myAlgorithm.log['eval'][-1])
        print('f_opt', myAlgorithm.log['f_opt'][-1])
        print('f_max', myAlgorithm.log['f_max'][-1])
        print('f_avg', myAlgorithm.log['f_avg'][-1])
        print('f_min', myAlgorithm.log['f_min'][-1])
        print('f_std', myAlgorithm.log['f_std'][-1])
        """=========================进化过程指标追踪分析========================="""
        metricName = [['eval']]
        Metrics = np.array([myAlgorithm.log[metricName[i][0]] for i in range(len(metricName))]).T
        # 绘制指标追踪分析图
        ea.trcplot(Metrics, labels=metricName, titles=metricName)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--train_length', type=int, default=None)
    parser.add_argument('--pred_length', type=int, default=None)
    parser.add_argument('--split_ratio', type=list, default=None)
    parser.add_argument('--time_max', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--epsilon_initial', type=float, default=None)
    parser.add_argument('--epsilon_decay_step', type=int, default=None)
    parser.add_argument('--epsilon_decay_ratio', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--ctx', type=int, default=None)
    args = parser.parse_args()

    config_filename = './Config/default.json'
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())
    # override default config
    dataset = args.data.upper()
    if dataset == 'PEMS03':
        config["id_filename"] = "data/PEMS03/PEMS03.txt"
        config["num_of_vertices"] = 358
    elif dataset == 'PEMS04':
        config["id_filename"] = None
        config["num_of_vertices"] = 307
    elif dataset == 'PEMS07':
        config["id_filename"] = None
        config["num_of_vertices"] = 883
    elif dataset == 'PEMS08':
        config["id_filename"] = None
        config["num_of_vertices"] = 170
    else:
        raise Exception(f'Input data is {args.data}, only support PEMS03/04/07/08')
    config["adj_filename"] = f"data/{dataset}/{dataset}.csv"
    config["graph_signal_matrix_filename"] = f"data/{dataset}/{dataset}.npz"
    config["pearsonr_adj_filename"] = f"data/{dataset}/{dataset}_pearsonr.npz"
    arg_dict = copy(vars(args))
    for key, value in vars(args).items():
        if value is None:
            arg_dict.pop(key)
    config.update(arg_dict)

    print(json.dumps(config, sort_keys=True, indent=4))
    if is_debug():
        log_name = 'debug'
        if os.path.exists('Log/debug_retrain/'):
            shutil.rmtree('Log/debug_retrain/')
    else:
        log_name = input('log_name:\n')

    wandb.init(project="GNN2", config=config, notes=log_name)
    ea_select_model(config, log_name)
