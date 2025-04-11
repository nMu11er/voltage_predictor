import socket
import pandas as pd
import pickle
import sys
import time
import os

# Default file paths
DEFAULT_DATA_DIR = '/home/pi/Documents/VoltageP_4_4'
DEFAULT_INPUT_FILE = 'voltage_data.xlsx'
DEFAULT_OUTPUT_FILE = 'voltage_predictions.xlsx'

# Worker Pi configuration
WORKER_PI_PORT = 65435

# Target number of prediction points
TARGET_PREDICTION_COUNT = 500

# Add your worker Pi IP addresses here
WORKER_IPS = [
    "10.24.195.121",  # Pi B
    # Add more worker IPs below as needed
    # "10.24.195.xxx",
    # "10.24.195.yyy",
]

def send_data_to_worker(worker_ip, df_numeric):
    print(f"Sending data to worker Pi at {worker_ip}...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(30)  # 30 second timeout
            s.connect((worker_ip, WORKER_PI_PORT))
            
            # Package data - send the full dataframe
            data_package = {
                'data': df_numeric.to_dict(orient='list'),
                'columns': df_numeric.columns.tolist()
            }
            
            # Send data
            data = pickle.dumps(data_package)
            s.send(len(data).to_bytes(8, 'big'))
            s.send(data)
            print(f"Sent {len(data)} bytes of data to worker Pi at {worker_ip}")
            
            # Receive predictions
            print(f"Waiting for predictions from worker Pi at {worker_ip}...")
            response = b""
            while True:
                try:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                except socket.timeout:
                    print(f"Socket timed out waiting for more data from {worker_ip}")
                    break
            
            # Deserialize the predictions
            result_package = pickle.loads(response)
            return result_package
            
    except Exception as e:
        print(f"Error communicating with worker Pi at {worker_ip}: {e}")
        return None

def make_local_predictions_with_workers(df_numeric, worker_ips=None):
    if worker_ips is None:
        worker_ips = WORKER_IPS
    
    print(f"Making predictions with Pi A's workers: {worker_ips}")
    
    # Dictionary to hold all workers' predictions
    all_worker_predictions = {}
    all_stable_columns = set()
    
    # Contact each worker in parallel (could be threaded for better performance)
    for worker_ip in worker_ips:
        result_package = send_data_to_worker(worker_ip, df_numeric)
        
        if result_package:
            worker_predictions = result_package.get('predictions', {})
            stable_columns = result_package.get('stable_columns', [])
            
            # Store worker's predictions
            all_worker_predictions[worker_ip] = worker_predictions
            all_stable_columns.update(stable_columns)
    
    # Combine predictions from all workers
    combined_predictions = {}
    for col in df_numeric.columns:
        col_predictions = []
        
        # Collect predictions for this column from all workers
        for worker_ip, predictions in all_worker_predictions.items():
            if col in predictions:
                col_predictions.append(predictions[col])
        
        # Average the predictions if available
        if col_predictions:
            # Find the shortest prediction length to avoid index errors
            min_len = min(len(pred) for pred in col_predictions)
            
            # Average all predictions
            combined_predictions[col] = []
            for i in range(min_len):
                values = [pred[i] for pred in col_predictions if i < len(pred)]
                if values:
                    combined_predictions[col].append(sum(values) / len(values))
    
    print(f"Predictions collected from {len(all_worker_predictions)} workers")
    print(f"Found {len(all_stable_columns)} stable columns")
    
    return {'predictions': combined_predictions, 'stable_columns': list(all_stable_columns)}

def send_data_to_pi_d(pi_d_address, df_numeric, local_results):
    PORT = 65435
    
    print(f"Connecting to Pi D ({pi_d_address})...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(60)  # Timeout for the entire operation
            s.connect((pi_d_address, PORT))
            
            # Package data
            data_package = {
                'data': df_numeric.to_dict(orient='list'),
                'columns': df_numeric.columns.tolist(),
                'local_predictions': local_results['predictions'],
                'local_stable_columns': local_results['stable_columns']
            }
            
            # Send data
            data = pickle.dumps(data_package)
            s.send(len(data).to_bytes(8, 'big'))
            s.send(data)
            print(f"Sent {len(data)} bytes of data to Pi D")
            
            # Receive predictions
            print("Waiting for predictions from Pi D...")
            response = b""
            while True:
                try:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                except socket.timeout:
                    print("Socket timed out waiting for more data")
                    break
            
            # Deserialize the predictions
            try:
                result_package = pickle.loads(response)
                predictions_dict = result_package.get('predictions', {})
                stable_columns = result_package.get('stable_columns', [])
                
                # Convert dict to DataFrame
                predictions_df = pd.DataFrame(predictions_dict)
                
                print(f"Received predictions with shape {predictions_df.shape}")
                print(f"Stable columns: {stable_columns}")
                
                return predictions_df, stable_columns
            except Exception as e:
                print(f"Error deserializing response from Pi D: {e}")
                return None, []
            
    except Exception as e:
        print(f"Error communicating with Pi D: {e}")
        return None, []

def run_iterative_prediction(pi_d_address, input_file=None, output_file=None, worker_ips=None):
    # Set default file paths if not provided
    if input_file is None:
        input_file = os.path.join(DEFAULT_DATA_DIR, DEFAULT_INPUT_FILE)
    
    if output_file is None:
        output_file = os.path.join(DEFAULT_DATA_DIR, DEFAULT_OUTPUT_FILE)
    
    # Make sure the data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Loading dataset from {input_file}...")
    try:
        original_df = pd.read_excel(input_file)
        print(f"Successfully loaded data with shape {original_df.shape}")
    except Exception as e:
        print(f"Failed to load the dataset: {e}")
        return
    
    # Keep only numerical data
    df_numeric = original_df.select_dtypes(include=['float', 'int'])
    
    # Initialize variables for iterative prediction
    current_df = df_numeric.copy()
    all_predictions = pd.DataFrame()
    all_stable_columns = set()
    iteration = 1
    
    # Start the prediction loop
    while True:
        print(f"\n--- Iteration {iteration}: Current predictions count: {len(all_predictions)} ---")
        
        # Step 1: Make predictions with Pi A's workers
        start_time = time.time()
        print("Step 1: Making predictions with Pi A's workers...")
        local_results = make_local_predictions_with_workers(current_df, worker_ips)
        local_pred_time = time.time() - start_time
        print(f"Local prediction completed in {local_pred_time:.2f} seconds")
        
        # Step 2: Send to Pi D and get extended predictions
        start_time = time.time()
        print("Step 2: Sending to Pi D for extended predictions...")
        pi_d_predictions_df, pi_d_stable_columns = send_data_to_pi_d(pi_d_address, current_df, local_results)
        pi_d_time = time.time() - start_time
        print(f"Pi D processing completed in {pi_d_time:.2f} seconds")
        
        if pi_d_predictions_df is None or len(pi_d_predictions_df) == 0:
            print("No valid predictions received from Pi D, ending iteration")
            break
            
        # Add new predictions to our collection
        if all_predictions.empty:
            all_predictions = pi_d_predictions_df
        else:
            all_predictions = pd.concat([all_predictions, pi_d_predictions_df], ignore_index=True)
        
        # Update stable columns
        all_stable_columns.update(pi_d_stable_columns)
        
        print(f"Total predictions so far: {len(all_predictions)}")
        
        # Check if we've reached the target
        if len(all_predictions) >= TARGET_PREDICTION_COUNT:
            print(f"Reached target of {TARGET_PREDICTION_COUNT} predictions")
            # Trim to exactly TARGET_PREDICTION_COUNT
            all_predictions = all_predictions.iloc[:TARGET_PREDICTION_COUNT]
            break
            
        # Update current_df for next iteration
        # Append the predictions to the original data
        current_df = pd.concat([df_numeric, pi_d_predictions_df], ignore_index=True)
        
        # Keep the most recent data points to avoid exponential growth
        if len(current_df) > len(df_numeric) * 2:
            # Keep original data plus most recent predictions
            current_df = pd.concat([
                df_numeric, 
                current_df.iloc[-(len(pi_d_predictions_df)):]
            ], ignore_index=True)
        
        print(f"Updated dataset for next iteration: {current_df.shape}")
        
        # Increment iteration counter
        iteration += 1
    
    # Save the full results
    print(f"Saving {len(all_predictions)} predictions to {output_file}")
    
    with pd.ExcelWriter(output_file) as writer:
        # All columns sheet
        all_components = [original_df]
        header_row = pd.DataFrame([["Predicted Voltages"] + [""] * (len(original_df.columns) - 1)], 
                                 columns=original_df.columns)
        all_components.append(header_row)
        
        # Convert predictions to same format as original
        formatted_predictions = pd.DataFrame(index=range(len(all_predictions)), 
                                           columns=original_df.columns)
        for col in df_numeric.columns:
            if col in all_predictions.columns:
                formatted_predictions[col] = all_predictions[col]
        
        all_components.append(formatted_predictions)
        pd.concat(all_components, ignore_index=True).to_excel(writer, 
                                                            sheet_name='All Columns', 
                                                            index=False)
        
        # Stable columns sheet
        stable_columns_list = list(all_stable_columns)
        if stable_columns_list:
            stable_df = original_df[stable_columns_list]
            stable_components = [stable_df]
            stable_header = pd.DataFrame([["Predicted Voltages"] + [""] * (len(stable_columns_list) - 1)], 
                                        columns=stable_columns_list)
            stable_components.append(stable_header)
            
            stable_predictions = all_predictions[stable_columns_list]
            stable_components.append(stable_predictions)
            pd.concat(stable_components, ignore_index=True).to_excel(writer, 
                                                                    sheet_name='Stable Columns', 
                                                                    index=False)
    
    print(f"Results saved to {output_file}")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Stable columns: {len(all_stable_columns)}")
    return all_predictions, list(all_stable_columns)

if __name__ == "__main__":
    # Default addresses
    PI_D_ADDRESS = "10.24.194.11"  # Pi D's IP address
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        PI_D_ADDRESS = sys.argv[1]
    
    # Optional input and output file paths
    input_file = None
    output_file = None
    
    if len(sys.argv) > 2:
        input_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    # Optional target prediction count
    if len(sys.argv) > 4:
        try:
            TARGET_PREDICTION_COUNT = int(sys.argv[4])
            print(f"Custom prediction target: {TARGET_PREDICTION_COUNT}")
        except ValueError:
            print(f"Invalid prediction count: {sys.argv[4]}, using default: {TARGET_PREDICTION_COUNT}")
    
    run_iterative_prediction(PI_D_ADDRESS, input_file, output_file, WORKER_IPS)
