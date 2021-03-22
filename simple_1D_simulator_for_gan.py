"""
Generate dataset by extremely Simplified 1D Simulations.
The simulations are not step-by-step, rather hard scripted.

This work is inspired by and based on the work:
    "AirDominance Through Machine Learning, A Preliminary Exploration of
    Artificial Intelligence-Assisted Mission Planning", RAND Corporation, 2020
        https://www.rand.org/pubs/research_reports/RR4311.html
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


class Fighter:
    FIGHTER_MAX_FIRING_RANGE = 40  # km

    def __init__(self):
        self.speed = 740  # km/h
        self.ingress = 0  # km
        self.max_firing_range = self.FIGHTER_MAX_FIRING_RANGE
        self.sskp = 1.0  # Assume perfect, but explicitly not used in the program


class Jammer:
    def __init__(self):
        self.jam_range = 30  # km
        self.speed = 740  # km/h
        self.ingress = 0  # km
        self.lead_distance = 0  # lead distance from fighter km (Not used in this 1D simulation)
        self.jam_effectiveness = 0.7  # Reduce ratio to the adversarial SAM range


class SAM:
    SAM_MAX_FIRING_RANGE = 40  # km
    SAM_MAX_OFFSET = 100  # km

    def __init__(self):
        self.max_firing_range = self.SAM_MAX_FIRING_RANGE
        self.max_offset = self.SAM_MAX_OFFSET  # km
        self.sskp = 1.0  # Assume perfect, but explicitly not used in the program


class DataClassBuffer:
    def __init__(self):
        self.conditions = []
        self.actions = []
        self.mission_results = []

    def setter(self, mission, fighter, jammer, sam):
        self.conditions.append([fighter.firing_range, sam.offset, sam.firing_range, sam.jammed_firing_range])
        self.actions.append([fighter.ingress, jammer.ingress])
        self.mission_results.append(mission)

    def saver(self, filename):
        conditions = np.array(self.conditions)
        actions = np.array(self.actions)
        mission_results = np.array(self.mission_results)

        np.savez_compressed(filename,
                            conditions=conditions,
                            actions=actions,
                            mission_results=mission_results)


class DataBuffer:
    def __init__(self):
        self.w1 = DataClassBuffer()
        self.w2 = DataClassBuffer()
        self.w3 = DataClassBuffer()
        self.l1 = DataClassBuffer()
        self.l2 = DataClassBuffer()

    def result_type_classifier(self, fighter, jammer, sam):
        mission = 'No classification'
        if (sam.firing_range > jammer.jam_range) and (fighter.firing_range > sam.firing_range):
            mission = 'w1'
            self.w1.setter(mission, fighter, jammer, sam)

        if jammer.jam_range > sam.firing_range > fighter.firing_range > sam.jammed_firing_range:
            mission = 'w2'
            self.w2.setter(mission, fighter, jammer, sam)

        if (jammer.jam_range > sam.firing_range) and (fighter.firing_range > sam.firing_range):
            mission = 'w3'
            self.w3.setter(mission, fighter, jammer, sam)

        if (sam.firing_range > jammer.jam_range) and (sam.firing_range > fighter.firing_range):
            mission = 'l1'
            self.l1.setter(mission, fighter, jammer, sam)

        if jammer.jam_range > sam.firing_range > sam.jammed_firing_range > fighter.firing_range:
            mission = 'l2'
            self.l2.setter(mission, fighter, jammer, sam)

        if mission == 'No classification':
            print('\nAttention!!!!!!!!!! Something is wrong in data generation!!!!!!!!!!')


def summarize_simulation_results(blue_win, blue_not_win, success_count, not_success_count):
    total_count = success_count + not_success_count
    print('\n----------------- Summary of Simulations -----------------')
    print(f'Blue win: {success_count},   Blue not win: {not_success_count},   '
          f'Total: {total_count}')
    print(f'Blue win rate: {success_count / total_count * 100:.3f} %,  '
          f'Blue not win rate: {not_success_count / total_count * 100:.3f} %')

    print(f'\n\n----------------- Summary of Blue win simulations -----------------')
    blue_win_array = analyze_simulation_results(blue_win, total_count)

    print(f'\n\n----------------- Summary of Blue not win simulations -----------------')
    blue_not_win_array = analyze_simulation_results(blue_not_win, total_count)

    print(f'\n\n----------------- Summary of whole simulations -----------------')
    whole_array = analyze_whole_simulation_results(blue_win_array, blue_not_win_array)

    return blue_win_array, blue_not_win_array, whole_array


def analyze_simulation_results(dataset, total_count):
    w1_len = len(dataset.w1.mission_results)
    w2_len = len(dataset.w2.mission_results)
    w3_len = len(dataset.w3.mission_results)
    l1_len = len(dataset.l1.mission_results)
    l2_len = len(dataset.l2.mission_results)

    total_length_of_data = w1_len + w2_len + w3_len + l1_len + l2_len

    if total_length_of_data > 0:
        print(f'Number of simulation data {total_length_of_data}')

        print(f'   case - w1: {w1_len},   ratio in the data:{w1_len / total_length_of_data * 100:.3f} %, '
              f'   ratio in the whole simulation:{w1_len / total_count * 100:.3f} %')
        print(f'   case - w2: {w2_len},   ratio in the data:{w2_len / total_length_of_data * 100:.3f} %, '
              f'   ratio in the whole simulation:{w2_len / total_count * 100:.3f} %')
        print(f'   case - w3: {w3_len},   ratio in the data:{w3_len / total_length_of_data * 100:.3f} %, '
              f'   ratio in the whole simulation:{w3_len / total_count * 100:.3f} %')
        print(f'   case - l1: {l1_len},   ratio in the data:{l1_len / total_length_of_data * 100:.3f} %, '
              f'   ratio in the whole simulation:{l1_len / total_count * 100:.3f} %')
        print(f'   case - l2: {l2_len},   ratio in the data:{l2_len / total_length_of_data * 100:.3f} %, '
              f'   ratio in the whole simulation:{l2_len / total_count * 100:.3f} %')

        win = w1_len + w2_len + w3_len
        lose = l1_len + l2_len
        print('\n')
        print(f'   * win_rate in the data: {win / total_length_of_data * 100:.3f} %')
        print(f'   * lose rate in the data: {lose / total_length_of_data * 100: .3f} %')

    else:
        print('   Unfortunately, total length of data = 0.  So, pass the calculation.')

    return np.array([w1_len, w2_len, w3_len, l1_len, l2_len])


def analyze_whole_simulation_results(blue_win_array, blue_not_win_array):
    whole_array = blue_win_array + blue_not_win_array

    w1 = whole_array[0]
    w2 = whole_array[1]
    w3 = whole_array[2]
    l1 = whole_array[3]
    l2 = whole_array[4]

    total_count = np.sum(whole_array)
    print(f'Number of total data: {total_count}')

    print(f'   Case W1:{w1},  {w1 / total_count * 100:.3f} %')
    print(f'   Case W2:{w2},  {w2 / total_count * 100:.3f} %')
    print(f'   Case W3:{w3},  {w3 / total_count * 100:.3f} %')
    print(f'   Case L1:{l1},  {l1 / total_count * 100:.3f} %')
    print(f'   Case L2:{l2},  {l2 / total_count * 100:.3f} %')
    print('\n')

    print(f'   * Case W1+W2+W3:{w1 + w2 + w3},   {(w1 + w2 + w3) / total_count * 100:.4} %')
    print(f'   * Case L1+L2   :{l1 + l2},   {(l1 + l2) / total_count * 100:.4} %')
    print('\n')

    return whole_array


def save_dataset(filename, dataset):
    dataset_name = filename + "_w1.npz"
    filename_w1 = os.path.join(filename, dataset_name)
    dataset.w1.saver(filename_w1)

    dataset_name = filename + "_w2.npz"
    filename_w2 = os.path.join(filename, dataset_name)
    dataset.w2.saver(filename_w2)

    dataset_name = filename + "_w3.npz"
    filename_w3 = os.path.join(filename, dataset_name)
    dataset.w3.saver(filename_w3)

    dataset_name = filename + "_l1.npz"
    filename_l1 = os.path.join(filename, dataset_name)
    dataset.l1.saver(filename_l1)

    dataset_name = filename + "_l2.npz"
    filename_l2 = os.path.join(filename, dataset_name)
    dataset.l2.saver(filename_l2)

    # Check saved data
    load_data(filename_w1)
    load_data(filename_w2)
    load_data(filename_w3)
    load_data(filename_l1)
    load_data(filename_l2)


def load_data(filename):
    data_1d = np.load(filename)
    print(f'file {filename}')
    print(f'   * contents: {data_1d.files}')
    # print(data_1d.files)

    print(f'   * conditions.shape: {data_1d["conditions"].shape},   '
          f'actions.shape: {data_1d["actions"].shape},   '
          f'mission_results.shape: {data_1d["mission_results"].shape}\n')


def main():
    filedir = os.path.join(os.getcwd(), DATASET_DIR)
    if not os.path.exists(filedir):
        os.mkdir(filedir)

    blue_win = DataBuffer()
    blue_not_win = DataBuffer()

    success_count = 0
    not_success_count = 0

    """ simulation loop """
    for _ in range(NUM_SIMULATIONS):
        """ Define and initialize the blue agents """
        fighter = Fighter()
        fighter.firing_range = np.random.random() * fighter.max_firing_range  # km

        jammer = Jammer()

        """ Define and initialize the red agents """
        sam = SAM()
        sam.firing_range = np.random.random() * sam.max_firing_range  # km
        sam.jammed_firing_range = jammer.jam_effectiveness * sam.firing_range

        sam_min_offset = max(sam.firing_range, fighter.firing_range, jammer.jam_range) + 1.0  # km
        sam.offset = sam_min_offset + np.random.random() * (sam.max_offset - sam_min_offset)  # km

        """ Generate the blue agents' action (mission plan), not beyond sam.offset """
        fighter_ingress = np.random.random() * sam.offset  # km
        jammer_ingress = np.random.random() * sam.offset  # km
        # jammer_lead_distance = np.random() * jammer_ingress  # km

        """ Perform simulation """
        ### update the position of fighter, jammer
        fighter.ingress = fighter_ingress  # km
        jammer.ingress = jammer_ingress  # km
        # jammer.lead_distance = jammer_lead_distance

        # Blue team win without using jammer
        blue_win_condition_1 = (fighter.ingress < sam.offset - sam.firing_range) and \
                               (jammer.ingress < sam.offset - sam.firing_range) and \
                               (fighter.firing_range > sam.offset - fighter.ingress)

        # Blue team wins with using jammer
        blue_win_condition_2 = (jammer.jam_range > sam.firing_range) and \
                               (fighter.ingress < sam.offset - sam.jammed_firing_range) and \
                               (jammer.ingress + jammer.jam_range > sam.offset) and \
                               (jammer.ingress < sam.offset - sam.jammed_firing_range) and \
                               (fighter.firing_range > sam.offset - fighter.ingress)

        """ Classify simulation results """
        if blue_win_condition_1 or blue_win_condition_2:
            success_count += 1
            blue_win.result_type_classifier(fighter, jammer, sam)
        else:
            not_success_count += 1
            blue_not_win.result_type_classifier(fighter, jammer, sam)

    """ Summarize the results and data """
    blue_win_array, blue_not_win_array, whole_array = \
        summarize_simulation_results(blue_win, blue_not_win, success_count, not_success_count)

    """ Save and load the data """
    print('\n--------------------- Blue win: saved and loaded data set fact ---------------------')
    filename = os.path.join(filedir, DATASET_NAME + '_blue_win')
    save_dataset(filename, blue_win)

    print('\n--------------------- Blue not win: saved and loaded data set fact ---------------------')
    filename = os.path.join(filedir, DATASET_NAME + '_blue_not__win')
    save_dataset(filename, blue_not_win)


if __name__ == '__main__':
    DATASET_DIR = 'dataset'

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="training", help="training/evaluation/test")
    args = parser.parse_args()

    if args.mode == 'training':
        SEED = 1000
        DATASET_NAME = 'data_for_gan_' + 'training'
        NUM_SIMULATIONS = 30000000
    elif args.mode == 'evaluation':
        SEED = 2000
        DATASET_NAME = 'data_for_gan_' + 'evaluation'
        NUM_SIMULATIONS = 100000
    elif args.mode == 'test':
        SEED = 3000
        DATASET_NAME = 'data_for_gan_' + 'test'
        NUM_SIMULATIONS = 100000
    else:
        print('Argment is wrong!!!!!')

    np.random.seed(SEED)
    main()
