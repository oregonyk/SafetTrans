import csv
import os

from pymoo.core.population import Population

from evaluate_nocrash_withmodel.regression_code.test_DL_SaftyDegree_collision import getCollision
from evaluate_nocrash_withmodel.regression_code.test_DL_SaftyDegree_nocollision import getNocollision
from evaluate_nocrash_withmodel.regression_code.test_DL_TET import getTET
from evaluate_nocrash_withmodel.regression_code.test_DL_TIT import getTIT
from evaluate_nocrash_withmodel.regression_code.test_DL_SaftyDegree import getSaftyDegree
from evaluate_nocrash_withmodel.retraining.regressionCollisionSaftyDegreeRetrain import reTrainCollisionSaftyDegreeModel
from evaluate_nocrash_withmodel.retraining.regressionNocollisionSaftyDegreeRetrain import \
    reTrainNocollisionSaftyDegreeModel
from evaluate_nocrash_withmodel.retraining.regressionTETRetrain import reTrainTETModel
from evaluate_nocrash_withmodel.retraining.regressionTITRetrain import reTrainTITModel
from evaluate_nocrash_withmodel.retraining.regressionSaftyDegreeRetrain import \
    reTrainSaftyDegreeModel  # TODO sd retraining
from evaluate_nocrash_withmodel.utils.data_processing import readcsv, readcsv_binary
from runners import NoCrashEvalRunner
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
import os

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def find_available_filename(folder_path):
    # 初始化模型计数器为1
    model_counter = 1

    writtenPath = f"{folder_path}{model_counter}.csv"
    # 检查文件夹中是否存在以'model'开头的文件名，如果有，递增计数器
    while os.path.exists(writtenPath):
        model_counter += 1
        writtenPath = f"{folder_path}{model_counter}.csv"
    print(writtenPath)
    return writtenPath


def getCluster_center(num_clusters, csvpath="evaluate_nocrash_withmodel/dataset/regression/collisionSeeds.csv"):
    data = pd.read_csv(csvpath)
    X = data.iloc[:, :12].values  # 假设前12列是特征数据
    column_names = data.columns[:12]
    # 选择聚类数
    # print(X[:num_clusters])
    # 使用K均值聚类算法
    # kmeans = KMeans(n_clusters=num_clusters)
    # kmeans.fit(X)
    # 获取每个簇的代表性点作为待测种子
    # cluster_centers = kmeans.cluster_centers_
    # return cluster_centers
    return X[:num_clusters]


def main(args):
    global f_r_model
    global writtenPath_model
    writtenPath_model = find_available_filename(
        "ScenarioNewAddedConfig/NoTransfer/modelWithReal_NoTransferModel")  # TODO need to connect to the real dataset
    f_r_model = open(writtenPath_model, "a+")
    global f_r_real
    global writtenPath_real
    writtenPath_real = find_available_filename(
        "ScenarioNewAddedConfig/NoTransfer/modelWithReal_NoTransferReal")  # TODO need to connect to the real dataset
    f_r_real = open(writtenPath_real, "a+")
    town = args.town
    weather = args.weather
    port = args.port
    tm_port = port + 2
    #runner = NoCrashEvalRunner(args, town, weather, port=port, tm_port=tm_port)
    runner = None
    # runner.run([4477.4, 0.118, 1.118, 0.229, 0.301, 8.607, 2297.667, 0.248, 3.5, 0.25, 32.863, 1500.0, 0.794938103241722])
    # runner.run([5800, 0.15, 2.0, 0.35, 0.5, 10.0, 2404, 0.3, 3.5, 0.25, 35.5, 1500])
    binaryGood = False
    regressionDangerGood = False
    regressionSafeGood = False
    TITGood = False
    TETGood = False
    collisionGood = False
    NocollisionGood = False
    # define whether directly use the model,if all good, just using model

    global writer_model
    writer_model = csv.writer(f_r_model)

    writer_real = csv.writer(f_r_real)
    global xWithResultCounter
    xWithResultCounter = 0
    global hadCollision
    hadCollision = 0
    # FIXME when using a new csv file to save the dataset, the following line is used for creating the table head.
    row0 = ['max_rpm', 'dampRateFullT', 'dampRateZero_CE', 'dampRateZero_CD', 'gearSwitchTime', 'clutchStrength',
            'mass', 'dragCoeff', 'tireFric', 'dampRate', 'radius', 'maxBrakeTorque', 'SaftyDegree', 'TET', 'TIT', 'acc',
            'TLCT', 'ALCJ', 'JerkAVG', 'yaw']
    # return output: result, TET, TIT, acc, TLCT, ALCJ ,JerkAVG, yaw, yaw is a list
    #the first Line
    #writer_model.writerow(row0)
    writer_real.writerow(row0)
    f_r_real.close()
    # get center to accelerate coverage
    popSize = 50
    cluster_centers = getCluster_center(popSize)
    # print(cluster_centers)

    problem = calraProblem(args, runner, binaryGood, TITGood, TETGood, collisionGood, NocollisionGood)
    crossover = SBX(prob=0.9, eta=15)
    mutation = PolynomialMutation(eta=20)

    # 创建 NSGA-II 算法实例并传递初始种群
    algorithm = NSGA2(
        pop_size=popSize,
        sampling=cluster_centers,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 100),
        # seed_population=population,
        verbose=False,
        save_history=False, )
    # xlist = [[4477.4, 0.118, 1.118, 0.229, 0.301, 8.607, 2297.667, 0.248, 3.5, 0.25, 32.863, 1500.0, 0.794938103241722], [4749.826, 0.107, 2.842, 0.378, 0.5, 10.724, 2365.108, 0.344, 1.146, 0.279, 35.5, 1591.547, 3.4684763077713665], [4749.826, 0.107, 2.842, 0.206, 0.5, 8.602, 2300.19, 0.247, 1.146, 0.25, 32.832, 1500.0, 3.427686755602993], [4477.4, 0.118, 1.118, 0.397, 0.301, 10.73, 2362.585, 0.346, 3.5, 0.279, 35.5, 1595.686, 1.1842791202538803]]

    f_r_model.close()


class calraProblem(ElementwiseProblem):
    def __init__(self, args, runner, binaryGood, TITGood, TETGood, collisionGood, nocollisionGood):
        self.runner = runner
        self.binaryGood = binaryGood
        self.TETGood = TETGood
        self.TITGood = TITGood
        self.collisionGood = collisionGood
        self.nocollisionGood = nocollisionGood

        self.TETRetrainModelPath = "evaluate_nocrash_withmodel/model/Torch/CARLA_SUN_TET_BEST.pth"
        self.TITRetrainModelPath = "evaluate_nocrash_withmodel/model/Torch/CARLA_SUN_TIT_BEST.pth"
        self.binaryRetrainModelPath = 'evaluate_nocrash_withmodel/model/Torch/CARLA_SUN_SaftyDegree_BEST.pth'
        self.collisionRetrainModlePath = "evaluate_nocrash_withmodel/model/Torch/CollisionCARLA_SUN_SaftyDegree_BEST.pth"
        self.nocollisionRetrainModlePath = "evaluate_nocrash_withmodel/model/Torch/NocollisionCARLA_SUN_SaftyDegree_BEST.pth"
        # all model are good, then no need to retrain the model else training
        self.modelTrainingSign = False
        super().__init__(n_var=12,
                         n_obj=5,  # 目标 3+TIT+TET
                         n_ieq_constr=1,  # 限制
                         # n_constr=0,
                         # phy = [5800, 0.15, 2.0, 0.35, 0.5, 10.0, 2404, 0.3, 3.5, 0.25, 35.5, 1500]
                         xl=[4200, 0.1, 1.0, 0.2, 0.3, 8.0, 2040, 0.2, 1.0, 0.2, 31.7, 1200],
                         xu=[7000, 0.2, 3.0, 0.4, 0.6, 12.0, 2700, 0.5, 3.9, 0.3, 37.0, 1650]
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        MinMaxScalerPath = "evaluate_nocrash_withmodel/regression_code/retrainedRegressionMinMaxScaler_1.pt"
        MinMaxScalerSDPath = "evaluate_nocrash_withmodel/regression_code/retrainedRegressionSaftyDegreeMinMaxScaler_1.pt"
        MinMaxScalerCollisionSDPath = "evaluate_nocrash_withmodel/regression_code/retrainedRegressionCollisionSaftyDegreeMinMaxScaler_1.pt"
        MinMaxScalerNocollisionSDPath = "evaluate_nocrash_withmodel/regression_code/retrainedRegressionNocollisionSaftyDegreeMinMaxScaler_1.pt"

        TETmodelPath = "evaluate_nocrash_withmodel/model/Torch/CARLA_SUN_TET_BEST.pth"
        TITmodelPath = "evaluate_nocrash_withmodel/model/Torch/CARLA_SUN_TIT_BEST.pth"
        binaryModelPath = 'evaluate_nocrash_withmodel/model/Torch/CARLA_SUN_SaftyDegree_BEST.pth'
        CollisionModelPath = "evaluate_nocrash_withmodel/model/Torch/CollisionCARLA_SUN_SaftyDegree_BEST.pth"
        NocollisionModelPath = "evaluate_nocrash_withmodel/model/Torch/NocollisionCARLA_SUN_SaftyDegree_BEST.pth"

        # TETRetrainModelPath =    "evaluate_nocrash_withmodel/retraining/retrainedRegressionTETTorch_2.pt"
        # TITRetrainModelPath =    "evaluate_nocrash_withmodel/retraining/retrainedRegressionTITTorch_2.pt"
        # binaryRetrainModelPath = "evaluate_nocrash_withmodel/retraining/retrainedRegressionSaftyDegreeTorch_1.pt"
        global hadCollision

        np.set_printoptions(suppress=True)
        th1 = 0.1

        unchange_count, x_after_th, maxPercent = self.unchangeCounter(x, th1)
        x = x_after_th
        x = [float('%.3f' % i) for i in x]
        x = np.array(x)
        changed = 12 - unchange_count
        # self.modelTrainingSign = False
        # model or realCarla
        self.modelTrainingSign = False
        if (self.modelTrainingSign == False):
            # running model prediction return with format:result, TET, TIT
            output = self.Prediction(x, MinMaxScalerSDPath, MinMaxScalerPath, MinMaxScalerCollisionSDPath,
                                     MinMaxScalerNocollisionSDPath, self.binaryRetrainModelPath,
                                     self.TITRetrainModelPath, self.TETRetrainModelPath, self.collisionRetrainModlePath,
                                     self.nocollisionRetrainModlePath)
            x = np.append(x, output)
            writer_model.writerow(x.tolist())
            print(output)
        out["F"] = [output[0], -output[2], -output[1], changed, maxPercent]

    def lowhigh_boundary(self):
        th_low = []
        th_high = []
        origin_phy = [5800, 0.15, 2, 0.35, 0.5, 10, 2404, 0.3, 3.5, 0.25, 35.5, 1500]
        for i in origin_phy:
            if (i > 1000):
                th_high.append(i * 1.01)
                th_low.append(i * 0.99)
            elif (i > 100):
                th_high.append(i * 1.02)
                th_low.append(i * 0.98)
            elif (i > 1):
                th_high.append(i * 1.04)
                th_low.append(i * 0.96)
            else:
                th_high.append(i * 1.08)
                th_low.append(i * 0.92)
        return th_high, th_low

    def unchangeCounter(self, x, th):
        counter = 0
        default_x = [5800, 0.15, 2, 0.35, 0.5, 10, 2404, 0.3, 3.5, 0.25, 35.5, 1500]
        max_unchange, min_unchange = self.lowhigh_boundary()
        x_after_th = x
        maxPercent = 0
        for i in range(12):
            if (x[i] < max_unchange[i] and x[i] > min_unchange[i]):
                x_after_th[i] = default_x[i]
                counter += 1
            else:
                perc = abs(x_after_th[i] - default_x[i]) / default_x[i]
                if (perc > maxPercent):
                    maxPercent = perc
        return counter, x_after_th, maxPercent

    def reTrainBinaryModel(self, modelPath, MinMaxScalerSDPath, data_path, label_name="SaftyDegree"):
        print(f"retrainBinaryWithDataPath:{data_path}")
        accuracy, path = reTrainSaftyDegreeModel(modelPath, MinMaxScalerSDPath, data_path, label_name)  # TODO
        return accuracy, path

    def reTrainTET(self, modelPath, MinMaxScalerPath, data_path, label_name="TET"):
        RMSE, path = reTrainTETModel(modelPath, MinMaxScalerPath, data_path, label_name)
        return RMSE, path

    def reTrainTIT(self, modelPath, MinMaxScalerPath, data_path, label_name="TIT"):
        # data_path = "../newDataset/Scenario2.csv"
        # label_name = "TIT"
        # modelPath = "../model/Torch/CARLA_SUN_BEST.pth"    CARLA_SUN_TIT_1
        # MinMaxScalerPath = "../model/minmax/minmax_scaler_model_best.pkl"
        RMSE, path = reTrainTITModel(modelPath, MinMaxScalerPath, data_path, label_name)
        return RMSE, path

    def reTrainCollision(self, modelPath, MinMaxScalerPath, data_path, label_name="SaftyDegree"):
        RMSE, path = reTrainCollisionSaftyDegreeModel(modelPath, MinMaxScalerPath, data_path, label_name)
        return RMSE, path

    def reTrainNocollision(self, modelPath, MinMaxScalerPath, data_path, label_name="SaftyDegree"):
        RMSE, path = reTrainNocollisionSaftyDegreeModel(modelPath, MinMaxScalerPath, data_path, label_name)
        return RMSE, path

    # def run(self, physics):
    #     results = self.runner.run(physics)
    #     return results

    def Prediction(self, x, MinMaxScalerSDPath, MinMaxScalerPath, MinMaxScalerCollisionSDPath,
                   MinMaxScalerNocollisionSDPath, binaryRetrainModelPath, TITRetrainModelPath, TETRetrainModelPath,
                   collisionRetrainModlePath, nocollisionRetrainModlePath):
        output1 = getSaftyDegree([x], binaryRetrainModelPath, MinMaxScalerSDPath)
        # print(output1)
        print(output1,end=" ")
        # if (output1 < 0.5):
        #     output4 = getCollision([x], collisionRetrainModlePath, MinMaxScalerCollisionSDPath)
        # else:
        #     output4 = getNocollision([x], nocollisionRetrainModlePath, MinMaxScalerNocollisionSDPath)

        output4a = getCollision([x], collisionRetrainModlePath, MinMaxScalerCollisionSDPath)

        output4b = getNocollision([x], nocollisionRetrainModlePath, MinMaxScalerNocollisionSDPath)
        print(output4a,output4b)
        output2 = getTET([x], TETRetrainModelPath, MinMaxScalerPath)
        output3 = getTIT([x], TITRetrainModelPath, MinMaxScalerPath)
        return [output4a, output2, output3, output1]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Agent configs
    parser.add_argument('--agent', default='autoagents/image_agent')
    parser.add_argument('--agent-config', default='config_nocrash.yaml')
    parser.add_argument('--town', default='Town01', choices=['Town01', 'Town02'])
    parser.add_argument('--weather', default='train', choices=['train', 'test'])
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')
    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--TITMinMaxScalerPath", type=str,
                        default='evaluate_nocrash_withmodel/regression_code/retrainedRegressionTITMinMaxScaler_1.pt',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--TETMinMaxScalerPath", type=str,
                        default='evaluate_nocrash_withmodel/regression_code/retrainedRegressionTETMinMaxScaler_1.pt',
                        help="Path to checkpoint used for saving statistics and resuming")

    args = parser.parse_args()
    for i in range(10):
        main(args)
    # town = args.town
    # weather = args.weather
    # port = args.port
    # tm_port = port + 2
    # runner = NoCrashEvalRunner(args, town, weather, port=port, tm_port=tm_port)
    # # pan
    # problem = CarlaProblem(args, runner)
    # phy = [5532.327075971282, 0.1511764262503684, 1.949074359107494, 0.34334726081207867, 0.49878789241714994,
    #        10.013150804231483, 2392.1375683104447, 0.3017215874217072, 2.0512287919754395, 0.24584126601219614,
    #        31.76028504012397, 1380.4365696487216]
    # phy = [5343.421404511669, 0.14246694073588326, 2.048842162969498, 0.3202717141191987, 0.43792433291358546,
    #        9.705702151310536, 2695.2102010540048, 0.31396058475976324, 2.0538906098972944, 0.2758477437686173,
    #        31.89079658699955, 1215.8487967017286]
    # phy = [4700.487514537978, 0.1561350081458622, 2.0401193355414042, 0.3500438730165327, 0.48402275567539177, 9.938917588467554, 2417.185666637578, 0.3112500435723273, 2.05071655916033, 0.28219832737191114, 31.760936442229553, 1215.5907326736597 ]
    #
    # algorithm = NSGAII(
    #     problem=problem,
    #     population_size=50,
    #     offspring_population_size=50,
    #     mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
    #     crossover=SBXCrossover(probability=0.9, distribution_index=20),
    #     termination_criterion=StoppingByEvaluations(max_evaluations=5000)
    # )
    #
    # algorithm.run()
    #
    # front = get_non_dominated_solutions(algorithm.get_result())
    #
    # # Save results to file
    # print_function_values_to_file(front, 'FUN.' + algorithm.label)
    # print_variables_to_file(front, 'VAR.' + algorithm.label)
    #
    # print(f'Algorithm: ${algorithm.get_name()}')
    # print(f'Problem: ${problem.get_name()}')
    # print(f'Computing time: ${algorithm.total_computing_time}')
    #
    # plot_front = Plot(title='Pareto Front',
    #                   axis_labels=['min maximum parameter change', 'min distance - speed', 'min changed para num'])
    # plot_front.plot(front, label='Three Objectives', filename='Pareto Front', format='png')
