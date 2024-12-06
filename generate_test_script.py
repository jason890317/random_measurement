import json
from itertools import product
import argparse

def validate_parameters(case, dimension, m_s, gate_num_times, method_s, total_methods, QEI_methods):
    """ Validate various input parameters to ensure they meet expected criteria. """
    
    if not (2 <= dimension <= 128):
        raise ValueError("Dimension 'dimension' must be between 2 and 64.")
    for m in m_s:
        if not isinstance(m, int):
            raise TypeError("Number of measurements 'm' must be integers.")
    for gate_num_time in gate_num_times:
        if not gate_num_time > 0:
            raise ValueError("Gate number time must be greater than 0.")
    for method in method_s:
        if method not in total_methods:
            raise ValueError(f"Method {method} is not a valid method.")
    for method in method_s:
        if method in QEI_methods and case != "special":
            raise ValueError(f"Case {case} is not valid, Case should be 1 or 2.")
    for method in method_s:
        if method not in QEI_methods and (case != "1" and case != "2"):
            raise ValueError(f"Case {case} should be 1 or 2.")
    if case != "1" and case != "2" and case != "special":
        raise ValueError(f"Case {case} should be 1, 2, or special.")

def generate_test_entries(dimension, m_s, rank_s, gate_num_times, case, method_s, copies_s, QEI_methods):
    """ Generate individual test entries based on the provided parameters. """
    
    test_data = []

    for method, m, rank in product(method_s, m_s, rank_s.get(dimension, [])):
        gate_num_time = 1
        copies = 1
        if method in QEI_methods:
            copies = copies_s
            if method in ["interweave", "special_blended", "optimizing_blended"]:
                gate_num_time = gate_num_times
        for time in (gate_num_time if isinstance(gate_num_time, list) else [gate_num_time]):
            if time == 1:
                for copy in (copies if isinstance(copies, list) else [copies]):
                    test_data.append({
                        "dimension": dimension,
                        "m": m,
                        "rank": rank,
                        "gate_num_time": time,
                        "method": method,
                        "copies": copy,
                        "case": case
                    })
            else:
                test_data.append({
                    "dimension": dimension,
                    "m": m,
                    "rank": rank,
                    "gate_num_time": time,
                    "method": method,
                    "copies": 1,
                    "case": case
                })
    return test_data

def save_data_to_json(file_path, data):
    """ Save the provided data to a JSON file at the specified path. """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def prompt_for_input(prompt_text, current_value, parse_func):
    """
    Prompt the user for input, displaying the current default value.
    If the user provides input, parse it; otherwise, use the current value or enforce input if default is None.

    Parameters:
    - prompt_text (str): The prompt message to display.
    - current_value: The current default value. If None, input is required.
    - parse_func (function): Function to parse the user input.

    Returns:
    - The parsed user input or the current value if input is blank and default is not None.
    """
    while True:
        if current_value is None:
            user_input = input(f"{prompt_text} (Required): ").strip()
            if user_input:  # Ensure input is not empty
                return parse_func(user_input)
            else:
                print("This field is required. Please provide a valid input.")
        else:
            if isinstance(current_value, list) or isinstance(current_value, dict):
                current_value_str = json.dumps(current_value)
            else:
                current_value_str = str(current_value)
            user_input = input(f"{prompt_text} (Press Enter for default: {current_value_str}): ").strip()
            if user_input:  # Use user-provided input
                return parse_func(user_input)
            else:  # Use default if provided
                return current_value

def main():
    QEI_methods = {"optimizing_blended", "interweave", "special_blended", "blended_three", "classical_shadow", "special_random"}
    total_methods = {"special_blended", "special_random", "interweave", "blended_three", "classical_shadow", "optimizing_blended", "blended", "random"}
    
    parser = argparse.ArgumentParser(description="Generate test script with specified parameters.")
    
    # Arguments with and without default values
    parser.add_argument("--dimension", type=int, nargs='+', help="Dimension of the system.")
    parser.add_argument("--m_s", type=int, nargs='+', help="List of measurements.")
    parser.add_argument("--gate_num_times", type=float, nargs='+', default=[1], help="List of gate number times.")
    parser.add_argument("--method_s", type=str, nargs='+', help="List of methods.")
    parser.add_argument("--copies_s", type=int, nargs='+', default=[1], help="List of copies.")
    parser.add_argument("--rank_s", type=json.loads, default={"16": [8]}, help="Dictionary of ranks.")
    parser.add_argument("--file_path", type=str, help="Output file name (without extension).")
    parser.add_argument("--case_1_high", type=float, default=0.9, help="Case 1 high value.")
    parser.add_argument("--case_1_low", type=float, default=0.1, help="Case 1 low value.")
    parser.add_argument("--case_2_pro", type=float, default=0.01, help="Case 2 pro value.")
    parser.add_argument("--test_time", type=int, default=100, help="Test time.")
    parser.add_argument("--average_time", type=int, default=100, help="Average time.")
    parser.add_argument("--case", type=str, help="Case identifier.")

    args = parser.parse_args()

    # Prompt user for arguments, even if defaults are set
    args.dimension = prompt_for_input("Enter dimension (space-separated)", args.dimension, int)
    args.m_s = prompt_for_input("Enter measurements (space-separated)", args.m_s, lambda x: list(map(int, x.split())))
    args.gate_num_times = prompt_for_input("Enter gate number times (space-separated)", args.gate_num_times, lambda x: list(map(float, x.split())))
    args.method_s = prompt_for_input("Enter methods (space-separated)", args.method_s, lambda x: x.split())
    
    while True:
        args.method_s = prompt_for_input("Enter methods (space-separated)", args.method_s, lambda x: x.split())
        if all(method in total_methods for method in args.method_s):
            break
        else:
            print(f"Some methods in {args.method_s} are not valid. Please enter valid methods from {total_methods}.")
    
    args.copies_s = prompt_for_input("Enter copies (space-separated)", args.copies_s, lambda x: list(map(int, x.split())))
    
    while True:
        args.rank_s = prompt_for_input(
            "Enter ranks (JSON format, e.g., {4:[2], 8:[3,4]})", 
            args.rank_s, 
            lambda x: json.loads(x)
        )
        try:
            args.rank_s = {int(k): v for k, v in args.rank_s.items()}
            print(f"Rank information: {args.rank_s}")
            if args.dimension in args.rank_s:
                break
            else:
                print(f"Rank information for the provided dimension {args.dimension} is missing in rank_s. Please provide the correct information.")
        except (ValueError, TypeError) as e:
            print(f"Invalid rank_s format: {e}. Please provide the correct information.")
    
    args.file_path = prompt_for_input("Enter output file name (no need \".json\")", args.file_path, lambda x: x)
    args.case_1_high = prompt_for_input("Enter case 1 high value", args.case_1_high, float)
    args.case_1_low = prompt_for_input("Enter case 1 low value", args.case_1_low, float)
    args.case_2_pro = prompt_for_input("Enter case 2 pro value", args.case_2_pro, float)
    args.test_time = prompt_for_input("Enter test time for case 1 and case 2", args.test_time, int)
    args.average_time = prompt_for_input("Enter average time", args.average_time, int)
    args.case = prompt_for_input("Enter case", args.case, str)

    # Debugging output for verification
    print("\nFinal arguments:")
    print(args)

    dimension = args.dimension
    m_s = args.m_s
    gate_num_times = args.gate_num_times
    method_s = args.method_s
    copies_s = args.copies_s
    rank_s = args.rank_s
    file_path = "./test_script/" + args.file_path + ".json"
    case_1_high = args.case_1_high
    case_1_low = args.case_1_low
    case_2_pro = args.case_2_pro
    test_time = args.test_time
    average_time = args.average_time
    case = args.case

    validate_parameters(case, dimension, m_s, gate_num_times, method_s, total_methods, QEI_methods)
    test_data = generate_test_entries(dimension, m_s, rank_s, gate_num_times, case, method_s, copies_s, QEI_methods)
    
    data = {
        "test_data": test_data,
        "case_1_high": case_1_high,
        "case_1_low": case_1_low,
        "case_2_pro": case_2_pro,
        "test_time": test_time,
        "average_time": average_time,
        "method_s": method_s,
        "m_s": m_s,
        "dimension": dimension,
        "rank_s": rank_s,
        "gate_num_times": gate_num_times
    }
    
    save_data_to_json(file_path, data)

if __name__ == "__main__":
    main()
