from quantumfed.data.datasets import get_dataset

def main():
    """Hàm chính để chạy thử nghiệm."""
    print("--- QuantumFed Framework: Data Module Test ---")

    # Đây là cách chúng ta sẽ sử dụng nó, giống như đọc từ file config
    dataset_name = "simple_nslkdd"
    data_path = "data/raw/simple_nslkdd.csv"

    print(f"\nAttempting to load dataset: '{dataset_name}'")
    
    # Sử dụng hàm factory để lấy đúng đối tượng dataset
    try:
        dataset_loader = get_dataset(name=dataset_name, data_path=data_path)
        
        # Lấy dữ liệu đã được xử lý
        X_train, y_train, X_test, y_test = dataset_loader.get_data()

        print("\n--- Data Loading and Processing Successful ---")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # In ra một vài mẫu dữ liệu đã được chuẩn hóa
        print("\nSample of preprocessed X_train data:")
        print(X_train[:2])

    except (ValueError, FileNotFoundError) as e:
        print(f"\n--- An Error Occurred ---")
        print(e)

if __name__ == "__main__":
    main()
