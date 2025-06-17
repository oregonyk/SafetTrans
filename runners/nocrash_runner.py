import os
import csv
import ray
from copy import deepcopy
from leaderboard.nocrash_evaluator import NoCrashEvaluator

import random
import numpy as np

from jmetal.core.problem import FloatProblem, BinaryProblem, Problem
from jmetal.core.solution import FloatSolution, BinarySolution, IntegerSolution, CompositeSolution
# ---------------------------------------------------------------------------------------------------
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, \
    print_variables_to_file
from jmetal.algorithm.multiobjective.random_search import RandomSearch
from jmetal.lab.visualization import Plot
import dill as pickle


class NoCrashEvalRunner():
    def __init__(self, args, town, weather, port=1000, tm_port=1002, debug=False):
        args = deepcopy(args)

        # Inject args
        args.scenario_class = 'nocrash_eval_scenario'
        args.port = port
        args.trafficManagerPort = tm_port
        args.debug = debug
        args.record = ''

        args.town = town
        args.weather = weather

        self.runner = NoCrashEvaluator(args, StatisticsManager(args))
        self.args = args
        self.problem = CarlaProblem(self.runner, self.args)

    def run(self,phy0):
        #phy0 = [5800.000, 0.150, 2.000, 0.350, 0.500, 10.000, 2404.00, 0.300, 3.500, 0.150, 35.500, 1500.000]

        # phy0 =[]
        # # nsga 0.005 14 = 4.75/29 = 4.72 /31 = 5.1
        # phy1 = [5804.736063286919, 0.14811081207608082, 1.3443176530778997, 0.34430134527900846, 0.4656599004982905,
        #         9.953745199176446, 2626.0426013869755, 0.29240937103301934, 3.4791873379482423, 0.2485979055478391,
        #         35.462615961268725, 1228.4761491912661]
        # phy2 = [5038.2947606782855, 0.14692449581132766, 1.6676619566074649, 0.27702851787710053, 0.49402758299939126,
        #         10.070441501191365, 2641.959308059798, 0.30714567419259975, 3.2898502785721027, 0.24842197267375546,
        #         35.48985725646545, 1247.9625061604152]
        # phy2 = [5800.86904935382, 0.15032140000774796, 1.5289805658371731, 0.3537423023878549, 0.49465277680120096,
        #         9.97151566424783, 2406.3925732325047, 0.30699644157020267, 3.545824199928635, 0.2516134577625787,
        #         35.48844277939358, 1209.4158557629557]
        # weather hard
        # phy = [4860.274040637637, 0.1445245555578696, 1.628242265760577, 0.34346523167095283, 0.5077039960133325, 9.338703775973896, 2688.141612864842, 0.2946375649292883, 1.1086206561253193, 0.2529164292068681, 34.80575520686602, 1225.7652268775792 ]
        # P = [phy2, phy1, phy3, phy0 ]
        # P=[]
        # result = []
        # for i in range(1):
        #     result.append(self.runner.run(self.args,phy0))
        # return result
        # phy0 = [4821.292962776257, 0.1518226982069306, 1.6168707125389707, 0.3030727388754983, 0.483710975877087, 11.871978910216283, 2436.8135195024565, 0.4341022000136966, 1.6585788508777826, 0.25729640977679236, 32.41619372998978, 1200.8649126049736 ]
        #
        # phy0 =[4821.292962776257, 0.1518226982069306, 1.8792576626557949, 0.2329384592874013, 0.48033330175289135, 11.82614643869514, 2668.5972706722855, 0.4341022000136966, 3.726642274319383, 0.25729640977679236, 36.123339503177, 1200.8649126049736 ]
        # phy0 =[4849.253486030477, 0.15408939140360475, 1.6236918855101248, 0.3044378084803953, 0.4243981074042719, 11.566514221568005, 2658.6132023327386, 0.2925105370615385, 1.747291817516912, 0.29828738554839046 ,32.441858605107335, 1200.823245578978 ]
        # phy0 = [4654.157253147053,
        #         0.14553847603499234,
        #         2.0158969544838867,
        #         0.344866466863447,
        #         0.5175600201129448,
        #         9.880048186520021,
        #         2416.2505195629124,
        #         0.308539116754132,
        #         3.5469342912094097,
        #         0.25593418987448663,
        #         33.21925183281808,
        #         1205.4921026238856]
        # phy0 =[4768.512945499018, 0.1697922243522476, 1.899636669708449, 0.3062621346074821, 0.5895775696561507, 10.713798809082892, 2580.342123796868, 0.33255521319972803, 1.115415024099688, 0.2771708649219944, 32.0103699194326, 1632.2832068388889]
        return self.runner.run(self.args, phy0)
        #
        # return self.search()
        # return  self.RS_search()

    def search(self):
        algorithm = NSGAII(
            problem=self.problem,
            population_size=50,
            offspring_population_size=50,
            mutation=PolynomialMutation(probability=1.0 / self.problem.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=0.9, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=5000)
        )
        """
            reload pickle
        """
        #
        # pickle_file  = open('nsga.pkl', 'rb')
        # al = pickle.load(pickle_file)
        # pickle_file.close()
        # algorithm = al
        # algorithm.getProblem().set_runner(self.runner)
        # eva = None
        # eva = algorithm.get_eva()
        # print("current eva:", eva)
        try:
            # print("abc")
            algorithm.run()
            # algorithm.rerun()
        except Exception as e:
            print("catch error:", e.args)
            save_file = open('nsga.pkl', 'wb')
            pickle.dump(algorithm, save_file)
            save_file.close()
            print("SAVE pkl")
        finally:
            print("done!")

        front = get_non_dominated_solutions(algorithm.get_result())

        # Save results to file
        print_function_values_to_file(front, 'FUN.' + algorithm.label)
        print_variables_to_file(front, 'VAR.' + algorithm.label)

        print(f'Algorithm: ${algorithm.get_name()}')
        print(f'Problem: ${self.problem.get_name()}')
        print(f'Computing time: ${algorithm.total_computing_time}')

        plot_front = Plot(title='Pareto Front',
                          axis_labels=['min maximum parameter change', 'min distance - speed', 'min changed para num'])
        plot_front.plot(front, label='Three Objectives', filename='Pareto rain Front', format='png')

    def RS_search(self):
        max_evaluations = 5000
        #
        # algorithm = RandomSearch(
        #     problem=self.problem,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # )

        # algorithm.run()
        """
                   reload pickle
        """
        # #
        pickle_file  = open('rs.pkl', 'rb')
        al = pickle.load(pickle_file)
        pickle_file.close()
        algorithm = al
        algorithm.getProblem().set_runner(self.runner)
        eva = algorithm.get_eva()
        print("current eva:", eva)
        eva = None
        try:
            # print("abc")
            # algorithm.run()
            algorithm.rerun()
        except Exception as e:
            print("catch error:", e.args)
            save_file = open('rs.pkl', 'wb')
            pickle.dump(algorithm, save_file)
            save_file.close()
            print("SAVE pkl")
            eva = algorithm.get_eva()
            print("current eva:",eva)
        finally:
            print("done!")
        front = algorithm.get_result()

        from jmetal.lab.visualization import Plot

        # Save results to file
        print_function_values_to_file(front, 'FUN.' + algorithm.label)
        print_variables_to_file(front, 'VAR.' + algorithm.label)

        print(f'Algorithm: ${algorithm.get_name()}')
        print(f'Problem: ${self.problem.get_name()}')
        print(f'Computing time: ${algorithm.total_computing_time}')

        plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
        plot_front.plot(front, label='Three Objectives', filename='RS Pareto Front', format='png')


# ---------------------------------------------------------------------------------
class CarlaProblem(FloatProblem):

    def __init__(self, runner, args):
        super(CarlaProblem, self).__init__()
        self.number_of_variables = 12
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.runner = runner
        self.args = args
        self.count = 0

        self.obj_direction = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(1)', 'f(2)', 'f(3)']
        # phy = [5800, 0.15, 2.0, 0.35, 0.5, 10.0, 2404, 0.3, 3.5, 0.25, 35.5, 1500]
        self.lower_bound = [4200, 0.1, 1.0, 0.2, 0.3, 8.0, 2040, 0.2, 1.0, 0.2, 31.7, 1200]
        self.upper_bound = [7000, 0.2, 3.0, 0.4, 0.6, 12.0, 2700, 0.5, 3.9, 0.3, 37.0, 1650]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def set_runner(self, runner):
        self.runner = runner

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        Vars = solution.variables
        print(Vars)

        x0 = float('%.3f' % Vars[0])
        x1 = float('%.3f' % Vars[1])
        x2 = float('%.3f' % Vars[2])
        x3 = float('%.3f' % Vars[3])
        x4 = float('%.3f' % Vars[4])
        x5 = float('%.3f' % Vars[5])
        x6 = float('%.3f' % Vars[6])
        x7 = float('%.3f' % Vars[7])
        x8 = float('%.3f' % Vars[8])
        x9 = float('%.3f' % Vars[9])
        x10 = float('%.3f' % Vars[10])
        x11 = float('%.3f' % Vars[11])

        global f_r
        f_r = open("file_storage.txt", 'a')
        # x0 = 5800
        # x1 = 0.15
        # x2 = 2.0
        # x3 = 0.35
        # x4 = 0.5
        # x5 = 10.0
        # x6 = 2404
        # x7 = 0.3
        # x8 = 3.5
        # x9 = 0.25
        # x10 = 35.5
        # x11 = 1500

        f_r.write(str(x0))
        f_r.write(' ')
        f_r.write(str(x1))
        f_r.write(' ')
        f_r.write(str(x2))
        f_r.write(' ')
        f_r.write(str(x3))
        f_r.write(' ')
        f_r.write(str(x4))
        f_r.write(' ')
        f_r.write(str(x5))
        f_r.write(' ')
        f_r.write(str(x6))
        f_r.write(' ')
        f_r.write(str(x7))
        f_r.write(' ')
        f_r.write(str(x8))
        f_r.write(' ')
        f_r.write(str(x9))
        f_r.write(' ')
        f_r.write(str(x10))
        f_r.write(' ')
        f_r.write(str(x11))
        f_r.write(' ')

        change = []
        change_ratio = []

        # changes' precision
        d0 = float('%.3f' % (np.abs(x0 - 5800) / (7000 - 4200)))
        d1 = float('%.3f' % (np.abs(x1 - 0.15) / (0.2 - 0.1)))
        d2 = float('%.3f' % (np.abs(x2 - 2) / (3 - 1)))
        d3 = float('%.3f' % (np.abs(x3 - 0.35) / (0.4 - 0.2)))
        d4 = float('%.3f' % (np.abs(x4 - 0.5) / (0.6 - 0.3)))
        d5 = float('%.3f' % (np.abs(x5 - 10) / (12 - 8)))
        d6 = float('%.3f' % (np.abs(x6 - 2404) / (2700 - 2040)))
        d7 = float('%.3f' % (np.abs(x7 - 0.3) / (0.5 - 0.2)))
        d8 = float('%.3f' % (np.abs(x8 - 3.5) / (3.9 - 1)))
        d9 = float('%.3f' % (np.abs(x9 - 0.25) / (0.3 - 0.2)))
        d10 = float('%.3f' % (np.abs(x10 - 35.5) / (37.0 - 31.7)))
        d11 = float('%.3f' % (np.abs(x11 - 1500) / (1650 - 1200)))

        # changes' ratio
        r0 = float('%.3f' % (np.abs(x0 - 5800) / 5800))
        r1 = float('%.3f' % (np.abs(x1 - 0.15) / 0.15))
        r2 = float('%.3f' % (np.abs(x2 - 2) / 2))
        r3 = float('%.3f' % (np.abs(x3 - 0.35) / 0.35))
        r4 = float('%.3f' % (np.abs(x4 - 0.5) / 0.5))
        r5 = float('%.3f' % (np.abs(x5 - 10) / 10))
        r6 = float('%.3f' % (np.abs(x6 - 2404) / 2404))
        r7 = float('%.3f' % (np.abs(x7 - 0.3) / 0.3))
        r8 = float('%.3f' % (np.abs(x8 - 3.5) / 3.5))
        r9 = float('%.3f' % (np.abs(x9 - 0.25) / 0.25))
        r10 = float('%.3f' % (np.abs(x10 - 35.5) / 35.5))
        r11 = float('%.3f' % (np.abs(x11 - 1500) / 1500))

        change_ratio.append(r0)
        change_ratio.append(r1)
        change_ratio.append(r2)
        change_ratio.append(r3)
        change_ratio.append(r4)
        change_ratio.append(r5)
        change_ratio.append(r6)
        change_ratio.append(r7)
        change_ratio.append(r8)
        change_ratio.append(r9)
        change_ratio.append(r10)
        change_ratio.append(r11)

        change_max = max(change_ratio)

        # max_rpm
        if d0 < 0.01:

            x0 = 5800.0
        else:
            change.append(d0)

        # damping_rate_full_throttle
        if d1 < 0.08:

            x1 = 0.15
        else:
            change.append(d1)

        # damping_rate_zero_throttle_clutch_engaged
        if d2 < 0.04:

            x2 = 2.0
        else:
            change.append(d2)

        # damping_rate_zero_throttle_clutch_disengaged
        if d3 < 0.08:

            x3 = 0.35
        else:
            change.append(d3)

        # gear_switch_time
        if d4 < 0.08:

            x4 = 0.5
        else:
            change.append(d4)

        # clutch_strength
        if d5 < 0.04:

            x5 = 10.0
        else:
            change.append(d5)

        # mass
        if d6 < 0.02:

            x6 = 2404.0
        else:
            change.append(d6)

        # drag_coefficient
        if d7 < 0.08:

            x7 = 0.3
        else:
            change.append(d7)

        # tire_friction
        if d8 < 0.04:

            x8 = 3.5
        else:
            change.append(d8)

        # damping_rate
        if d9 < 0.08:

            x9 = 0.25
        else:
            change.append(d9)

        # radius
        if d10 < 0.04:

            x10 = 35.5
        else:
            change.append(d10)

        # max_brake_torque
        if d11 < 0.02:

            x11 = 1500.0
        else:
            change.append(d11)

        physics = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
        # physics = [5800.000, 0.150, 2.000, 0.350, 0.500, 10.000, 2404.00, 0.300, 3.500, 0.150, 35.500, 1500.000]
        physics = []
        if len(change) == 0:
            physics = []
        try:
            result, TET, TIT, acc = self.runner.run(self.args, physics)
        except Exception as e:
            print("error:", e.args)
            self.runner = None
            f12 = float('%.3f' % change_max)
            f13 = float('%.3f' % 6.0)
            f14 = len(change)
            f_r.write(str(f12))
            f_r.write(' ')
            f_r.write(str(f13))
            f_r.write(' ')
            f_r.write(str(f14))
            f_r.write(' ')
            f_r.write(str(TET))
            f_r.write(' ')
            f_r.write(str(TIT))
            f_r.write(' ')
            f_r.write(str(acc))
            f_r.write(' ')
            f_r.write("\n")
            f_r.close()
            solution.objectives[0] = f12  # min maximum para change
            solution.objectives[1] = f13  # distance - speed
            solution.objectives[2] = f14  # changed para num
            raise Exception(e)

        # print(result)

        f12 = float('%.3f' % change_max)
        f13 = float('%.3f' % result)
        f14 = len(change)

        f_r.write(str(f12))
        f_r.write(' ')
        f_r.write(str(f13))
        f_r.write(' ')
        f_r.write(str(f14))
        f_r.write(' ')
        f_r.write(str(TET))
        f_r.write(' ')
        f_r.write(str(TIT))
        f_r.write(' ')
        f_r.write(str(acc))
        f_r.write(' ')
        f_r.write("\n")
        f_r.close()
        solution.objectives[0] = f12  # min maximum para change
        solution.objectives[1] = f13  # distance - speed
        solution.objectives[2] = f14  # changed para num

        self.count += 1
        if self.count == 36:
            print("time to end")
            self.runner = None
            raise Exception("over")
        return solution

    def get_name(self):
        return 'CarlaProblem'


# ---------------------------------------------------------------------------------

class StatisticsManager:
    headers = [
        'town',
        'traffic',
        'weather',
        'start',
        'target',
        'route_completion',
        'lights_ran',
        'duration',
        'distance',
    ]

    def __init__(self, args):

        self.finished_tasks = {
            'Town01': {},
            'Town02': {}
        }

        logdir = args.agent_config.replace('.yaml', '.csv')

        if args.resume and os.path.exists(logdir):
            self.load(logdir)
            self.csv_file = open(logdir, 'a')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.headers)
        else:
            self.csv_file = open(logdir, 'w')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.headers)
            self.csv_writer.writeheader()

    def load(self, logdir):
        with open(logdir, 'r') as file:
            log = csv.DictReader(file)
            for row in log:
                self.finished_tasks[row['town']][(
                    int(row['traffic']),
                    int(row['weather']),
                    int(row['start']),
                    int(row['target']),
                )] = [
                    float(row['route_completion']),
                    int(row['lights_ran']),
                    float(row['duration']),
                    float(row['distance']),
                ]

    def log(self, town, traffic, weather, start, target, route_completion, lights_ran, duration, distance):
        self.csv_writer.writerow({
            'town': town,
            'traffic': traffic,
            'weather': weather,
            'start': start,
            'target': target,
            'route_completion': route_completion,
            'lights_ran': lights_ran,
            'duration': duration,
            'distance': distance,
        })

        self.csv_file.flush()

    def is_finished(self, town, route, weather, traffic):
        start, target = route
        key = (int(traffic), int(weather), int(start), int(target))
        return key in self.finished_tasks[town]
