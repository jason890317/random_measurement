import json
import sys
import numpy as np
from event_learning_fuc import quantum_event_finding, quantum_event_identification
from classical_shadow import classical_shadow
from tools import print_progress

def load_test_data(test_data_path):
    with open(test_data_path, 'r') as file:
        return json.load(file)

def get_povm_set_and_method_function(method, measurement_dir, d, m, rank, case):
    if method in ["special_blended", "special_random", "interweave", "blended_three", "optimizing_blended"]:
        return quantum_event_identification, np.load(f'{measurement_dir}case_{case}_d_{d}_m_{m}_r_{rank}.npy')
    elif method == "classical_shadow":
        return classical_shadow, np.load(f'{measurement_dir}case_{case}_d_{d}_m_{m}_r_{rank}.npy')
    elif method in ["random", "blended"]:
        return quantum_event_finding, np.load(f'{measurement_dir}case_{case}_d_{d}_m_{m}_r_{rank}.npy')
    else:
        raise ValueError(f"Unknown method: {method}")

def run_experiment(test_data, povm_set, excuted_function, average_time, state, roh_0, case_1_high, test_time):
    experiment_raw_data = []
    theorem_raw_data = []

    for i in range(average_time):
        function_arg = (test_data["copies"], test_data["dimension"], test_data["m"], povm_set[i], state) if test_data["method"] == "classical_shadow" else (
            test_data["copies"], test_data["dimension"], test_data["m"], test_data["gate_num_time"], povm_set[i], roh_0, test_data["case"], state, test_time, case_1_high, test_data["method"])
        result = excuted_function(*function_arg)
        print(f'result:{result}')
        print(f'd: {test_data["dimension"]}, rank: {test_data["rank"]}, method: {test_data["method"]}, m: {test_data["m"]}, coef: {test_data["gate_num_time"]}, copies: {test_data["copies"]}')
        experiment_raw_data.append(result["experiment"])
        theorem_raw_data.append(result["theorem"])

        print_progress(i + 1, average_time, bar_length=average_time)
        print("\n")

    return experiment_raw_data, theorem_raw_data

def main():

    test_data_path = "./test_script/" + sys.argv[1]
    file_path = "result_json/" + sys.argv[1]
    measurement_dir = './Haar_measurement_dir/'

    data = load_test_data(test_data_path)

    case_1_high = data["case_1_high"]
    case_1_low = data["case_1_low"]
    case_2_pro = data["case_2_pro"]
    test_time = data["test_time"]
    average_time = data["average_time"]

    for test_data in data["test_data"]:
        state = np.array([np.hstack((1, np.zeros(test_data["dimension"] - 1)))])
        roh_0 = np.outer(state, state.T.conj())

        excuted_function, povm_set = get_povm_set_and_method_function(test_data["method"], measurement_dir, test_data["dimension"], test_data["m"], test_data["rank"], test_data["case"])

        experiment_raw_data, theorem_raw_data = run_experiment(test_data, povm_set, excuted_function, average_time, state, roh_0, case_1_high, test_time)

        test_data["result"] = {
            "experiment": experiment_raw_data,
            "theorem": theorem_raw_data
        }

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    main()
