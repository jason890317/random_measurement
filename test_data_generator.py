import json
from itertools import product

def validate_parameters(d_s, m_s, gate_num_times, method_s, total_methods):
    """ Validate various input parameters to ensure they meet expected criteria. """
    for d in d_s:
        if not (2 <= d <= 64):
            raise ValueError("Dimension 'd' must be between 2 and 64.")
    for m in m_s:
        if not isinstance(m, int):
            raise TypeError("Number of measurements 'm' must be integers.")
    for gate_num_time in gate_num_times:
        if not gate_num_time > 0:
            raise ValueError("Gate number time must be greater than 0.")
    for method in method_s:
        if method not in total_methods:
            raise ValueError(f"Method {method} is not a valid method.")

def generate_test_entries(d_s, m_s, rank_s, gate_num_times, case,method_s, copies_s, special_methods):
    """ Generate individual test entries based on the provided parameters. """
    test_data = []
    for d in d_s:
        for method, m, rank in product(method_s, m_s, rank_s.get(d, [])):
            gate_num_time = 1
            copies = 1
            if method in special_methods:
                copies = copies_s
                if method in ["interweave", "special_blended"]:
                    gate_num_time = gate_num_times
            for copy in (copies if isinstance(copies, list) else [copies]):
                for time in (gate_num_time if isinstance(gate_num_time, list) else [gate_num_time]):
                    test_data.append({
                        "d": d,
                        "m": m,
                        "rank": rank,
                        "gate_num_time": time,
                        "method": method,
                        "copies": copy,
                        "case":case
                    })
    return test_data

def save_data_to_json(file_path, data):
    """ Save the provided data to a JSON file at the specified path. """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    d_s = [4]          
    m_s = [10,20,30]         
    gate_num_times = [1]
    total_methods = {"special_blended", "special_random", "interweave", "blended_three", "classical_shadow", "blended", "random"}
    method_s = ["special_blended", "classical_shadow", "blended_three", "interweave", "special_random", "random", "blended"]
    copies_s = [1, 2]
    special_methods = {"interweave", "special_blended", "blended_three", "classical_shadow"}
    rank_s = {2:[1], 4:[2], 8:[4], 16:[7,8,9], 32:[16]}
    file_path = "test_data.json"

    case_1_high = 0.9                 
    case_1_low = 0.1                 
    case_2_pro = 0.01
    test_time = 50                     
    average_time = 30          
    case = 1

    validate_parameters(d_s, m_s, gate_num_times, method_s, total_methods)
    test_data = generate_test_entries(d_s, m_s, rank_s, gate_num_times,case, method_s, copies_s, special_methods)
    
    data = {
        "test_data": test_data,
        "case_1_high": case_1_high,
        "case_1_low": case_1_low,
        "case_2_pro": case_2_pro,
        "test_time": test_time,
        "average_time": average_time,
        "method_s": method_s,
        "m_s": m_s,
        "d_s": d_s,
        "rank_s": rank_s,
        "gate_num_times": gate_num_times
    }
    
    save_data_to_json(file_path, data)

if __name__ == "__main__":
    main()
