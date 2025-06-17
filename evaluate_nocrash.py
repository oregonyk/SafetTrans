from runners import NoCrashEvalRunner

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
from jmetal.lab.visualization import Plot


def main(args):
    town = args.town
    weather = args.weather

    port = args.port
    tm_port = port + 2
    runner = NoCrashEvalRunner(args, town, weather, port=port, tm_port=tm_port)
    #runner.run()

    result = runner.run([4477.4, 0.118, 1.118, 0.229, 0.301, 8.607, 2297.667, 0.248, 3.5, 0.25, 32.863, 1500.0, 0.794938103241722])
    # return


class CarlaProblem(FloatProblem):

    def __init__(self, args, runner):
        super(CarlaProblem, self).__init__()
        self.number_of_variables = 12
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.args = args
        self.runner = runner

        self.obj_direction = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(1)', 'f(2)', 'f(3)']

        self.lower_bound = [4200, 0.1, 1.0, 0.2, 0.3, 8.0, 1940, 0.2, 1.0, 0.2, 31.7, 1200]
        self.upper_bound = [5900, 0.2, 3.0, 0.4, 0.6, 12.0, 2700, 0.5, 3.0, 0.3, 36.0, 1600]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

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
        d0 = float('%.3f' % (np.abs(x0 - 5800) / (5900 - 4200)))
        d1 = float('%.3f' % (np.abs(x1 - 0.15) / (0.2 - 0.1)))
        d2 = float('%.3f' % (np.abs(x2 - 2) / (3 - 1)))
        d3 = float('%.3f' % (np.abs(x3 - 0.35) / (0.4 - 0.2)))
        d4 = float('%.3f' % (np.abs(x4 - 0.5) / (0.6 - 0.3)))
        d5 = float('%.3f' % (np.abs(x5 - 10) / (12 - 8)))
        d6 = float('%.3f' % (np.abs(x6 - 2404) / (2700 - 1940)))
        d7 = float('%.3f' % (np.abs(x7 - 0.3) / (0.5 - 0.2)))
        d8 = float('%.3f' % (np.abs(x8 - 2) / (3 - 1)))
        d9 = float('%.3f' % (np.abs(x9 - 0.25) / (0.3 - 0.2)))
        d10 = float('%.3f' % (np.abs(x10 - 31.7) / (36.0 - 31.7)))
        d11 = float('%.3f' % (np.abs(x11 - 1500) / (1600 - 1200)))

        # changes' ratio
        r0 = float('%.3f' % (np.abs(x0 - 5800) / 5800))
        r1 = float('%.3f' % (np.abs(x1 - 0.15) / 0.15))
        r2 = float('%.3f' % (np.abs(x2 - 2) / 2))
        r3 = float('%.3f' % (np.abs(x3 - 0.35) / 0.35))
        r4 = float('%.3f' % (np.abs(x4 - 0.5) / 0.5))
        r5 = float('%.3f' % (np.abs(x5 - 10) / 10))
        r6 = float('%.3f' % (np.abs(x6 - 2404) / 2404))
        r7 = float('%.3f' % (np.abs(x7 - 0.3) / 0.3))
        r8 = float('%.3f' % (np.abs(x8 - 2) / 2))
        r9 = float('%.3f' % (np.abs(x9 - 0.25) / 0.25))
        r10 = float('%.3f' % (np.abs(x10 - 31.7) / 31.7))
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

            x8 = 2.0
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

        # result = self.runner.run(self.args, physics)
        # result = self.run(physics)
        result = self.runner.run(physics)

        print(result)

        f12 = float('%.3f' % change_max)
        f13 = float('%.3f' % result)
        f14 = len(change)

        f_r.write(str(f12))
        f_r.write(' ')
        f_r.write(str(f13))
        f_r.write(' ')
        f_r.write(str(f14))
        f_r.write(' ')
        f_r.write("\n")

        solution.objectives[0] = f12  # min maximum para change
        solution.objectives[1] = f13  # distance - speed
        solution.objectives[2] = f14  # changed para num

        return solution

    def run(self, physics):

        # town = self.args.town
        # weather = self.args.weather
        #
        # port = self.args.port
        # tm_port = port + 2
        # runner = NoCrashEvalRunner(self.args, town, weather, port=port, tm_port=tm_port)
        # result = runner.run(physics)
        # del runner
        result = self.runner.run(physics)
        print(result)
        return result

    def get_name(self):
        return 'CarlaProblem'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Agent configs
    parser.add_argument('--agent', default='autoagents/image_agent')
    parser.add_argument('--agent-config', default='config_nocrash.yaml')

    # Benchmark configs
    #parser.add_argument('--town', required=True, choices=['Town01', 'Town02'])
    #parser.add_argument('--weather', required=True, choices=['train', 'test'])
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

    args = parser.parse_args()

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
