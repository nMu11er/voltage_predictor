import socket
import pandas as pd
import pickle
import sys
import time
import os
import numpy as np

# Default file paths
DEFAULT_DATA_DIR = '/home/pi/Documents/VoltageP_4_4'
DEFAULT_INPUT_FILE = 'voltage_data.xlsx'
DEFAULT_OUTPUT_FILE = 'voltage_predictions.xlsx'

# Worker Pi configuration
WORKER_PI_PORT = 65435

# Target number of prediction points
TARGET_PREDICTION_COUNT = 600

# Worker IPs
WORKER_IPS = [
    "10.24.195.121",  # Pi B
]

# Pi F and Pi G addresses
PI_F_ADDRESS = "10.24.196.48"
PI_G_ADDRESS = "10.24.198.233"

def send_data_to_worker(worker_ip, df_numeric):
    print(f"Sending data to worker Pi at {worker_ip}...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(30)
            s.connect((worker_ip, WORKER_PI_PORT))
            data_package = {
                'data': df_numeric.to_dict(orient='list'),
                'columns': df_numeric.columns.tolist()
            }
            data = pickle.dumps(data_package)
            s.send(len(data).to_bytes(8, 'big'))
            s.sendall(data)
            print(f"Sent {len(data)} bytes to worker {worker_ip}")
            response = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                response += chunk
            return pickle.loads(response)
    except Exception as e:
        print(f"Error communicating with worker Pi: {e}")
        return None

def make_local_predictions_with_workers(df_numeric, worker_ips=None):
    if worker_ips is None:
        worker_ips = WORKER_IPS
    all_worker_predictions = {}
    all_stable_columns = set()
    for worker_ip in worker_ips:
        result_package = send_data_to_worker(worker_ip, df_numeric)
        if result_package:
            all_worker_predictions[worker_ip] = result_package.get('predictions', {})
            all_stable_columns.update(result_package.get('stable_columns', []))
    combined_predictions = {}
    for col in df_numeric.columns:
        col_predictions = []
        for predictions in all_worker_predictions.values():
            if col in predictions:
                col_predictions.append(predictions[col])
        if col_predictions:
            min_len = min(len(p) for p in col_predictions)
            combined_predictions[col] = [sum(p[i] for p in col_predictions) / len(col_predictions) for i in range(min_len)]
    return {'predictions': combined_predictions, 'stable_columns': list(all_stable_columns)}

def send_data_to_pi_d(pi_d_address, df_numeric, local_results):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(60)
            s.connect((pi_d_address, WORKER_PI_PORT))
            data_package = {
                'data': df_numeric.to_dict(orient='list'),
                'columns': df_numeric.columns.tolist(),
                'local_predictions': local_results['predictions'],
                'local_stable_columns': local_results['stable_columns']
            }
            data = pickle.dumps(data_package)
            s.send(len(data).to_bytes(8, 'big'))
            s.sendall(data)
            response = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                response += chunk
            result_package = pickle.loads(response)
            return pd.DataFrame(result_package.get('predictions', {})), result_package.get('stable_columns', [])
    except Exception as e:
        print(f"Error communicating with Pi D: {e}")
        return None, []

def send_matrix_to_pi(ip_address, matrix_df):
    """Send the current full matrix to Pi F or Pi G."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ip_address, WORKER_PI_PORT))
            data = pickle.dumps(matrix_df)
            s.send(len(data).to_bytes(8, 'big'))
            s.sendall(data)
            print(f"Matrix sent to {ip_address}, shape {matrix_df.shape}")
    except Exception as e:
        print(f"Failed to send matrix to {ip_address}: {e}")

def run_iterative_prediction(pi_d_address, input_file=None, output_file=None, worker_ips=None):
    if input_file is None:
        input_file = os.path.join(DEFAULT_DATA_DIR, DEFAULT_INPUT_FILE)
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DEFAULT_DATA_DIR, f"voltage_predictions_{timestamp}.xlsx")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading dataset from {input_file}...")
    try:
        original_df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    df_numeric = original_df.select_dtypes(include=['float', 'int'])
    current_df = df_numeric.copy()
    all_predictions_a = pd.DataFrame()
    all_stable_columns = set()
    iteration = 1

    while True:
        print(f"\n--- Iteration {iteration} ---")
        print("Step 1: Making predictions with Pi A's workers...")
        local_results = make_local_predictions_with_workers(current_df, worker_ips)
        a_predictions_df = pd.DataFrame(local_results['predictions'])

        if len(a_predictions_df) > 100:
            a_predictions_df = a_predictions_df.iloc[:100]
            print(f"Limiting to first 100 predictions")

        print("Step 2: Sending to Pi D for comparison...")
        pi_d_predictions_df, pi_d_stable_columns = send_data_to_pi_d(pi_d_address, current_df, local_results)
        if pi_d_predictions_df is None or len(pi_d_predictions_df) == 0:
            print("No valid predictions from Pi D, ending")
            break

        # Compare
        compare_len = min(len(a_predictions_df), len(pi_d_predictions_df))
        a_compare = a_predictions_df.iloc[:compare_len]
        d_compare = pi_d_predictions_df.iloc[:compare_len]
        similarity_scores = []
        for col in a_compare.columns:
            if col in d_compare.columns:
                a_vals, d_vals = a_compare[col].values, d_compare[col].values
                diffs = [abs((dv - av) / av) * 100 if abs(av) >= 0.001 else 0 for av, dv in zip(a_vals, d_vals)]
                similarity_scores.append(np.mean(diffs))
        overall_similarity = np.mean(similarity_scores)
        print(f"Overall similarity: {overall_similarity:.2f}%")

        if overall_similarity <= 20.0:
            print("Predictions are similar enough, continuing...")
            if all_predictions_a.empty:
                all_predictions_a = a_predictions_df
            else:
                all_predictions_a = pd.concat([all_predictions_a, a_predictions_df], ignore_index=True)
            all_stable_columns.update(local_results['stable_columns'])
            print(f"Total predictions so far: {len(all_predictions_a)}")

            # 🛫 NEW: Send full matrix to Pi F every 100 predictions
            if len(all_predictions_a) % 100 == 0:
                full_matrix = pd.concat([df_numeric, all_predictions_a], ignore_index=True)
                send_matrix_to_pi(PI_F_ADDRESS, full_matrix)

            # 🛫 NEW: Send full matrix to Pi G every 200 predictions
            if len(all_predictions_a) % 200 == 0:
                full_matrix = pd.concat([df_numeric, all_predictions_a], ignore_index=True)
                send_matrix_to_pi(PI_G_ADDRESS, full_matrix)

            # Stop when we hit target
            if len(all_predictions_a) >= TARGET_PREDICTION_COUNT:
                print(f"Reached {TARGET_PREDICTION_COUNT} predictions!")
                all_predictions_a = all_predictions_a.iloc[:TARGET_PREDICTION_COUNT]
                break

            # Update input
            current_df = pd.concat([df_numeric, all_predictions_a], ignore_index=True)
            if len(current_df) > len(df_numeric) * 2:
                current_df = pd.concat([df_numeric, current_df.iloc[-len(a_predictions_df):]], ignore_index=True)
        else:
            print("Predictions differ too much, stopping")
            break

        iteration += 1

    # Save results
    print(f"Saving predictions to {output_file}")
    with pd.ExcelWriter(output_file) as writer:
        all_components = [original_df]
        header_row = pd.DataFrame([["Predicted Voltages"] + [""] * (len(original_df.columns) - 1)], columns=original_df.columns)
        all_components.append(header_row)
        formatted_predictions = pd.DataFrame(index=range(len(all_predictions_a)), columns=original_df.columns)
        for col in df_numeric.columns:
            if col in all_predictions_a.columns:
                formatted_predictions[col] = all_predictions_a[col]
        all_components.append(formatted_predictions)
        pd.concat(all_components, ignore_index=True).to_excel(writer, sheet_name="All Columns", index=False)

        stable_list = list(all_stable_columns)
        if stable_list:
            stable_df = original_df[stable_list]
            stable_components = [stable_df]
            stable_header = pd.DataFrame([["Predicted Voltages"] + [""] * (len(stable_list) - 1)], columns=stable_list)
            stable_components.append(stable_header)
            stable_predictions = all_predictions_a[stable_list]
            stable_components.append(stable_predictions)
            pd.concat(stable_components, ignore_index=True).to_excel(writer, sheet_name="Stable Columns", index=False)

    print("Done saving!")

if __name__ == "__main__":
    PI_D_ADDRESS = "10.24.194.11"
    if len(sys.argv) > 1:
        PI_D_ADDRESS = sys.argv[1]
    input_file = None
    output_file = None
    if len(sys.argv) > 2:
        input_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    run_iterative_prediction(PI_D_ADDRESS, input_file, output_file, WORKER_IPS)
