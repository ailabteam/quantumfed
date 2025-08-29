import abc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class BaseDataset(abc.ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các bộ dữ liệu.
    Bất kỳ bộ dữ liệu mới nào cũng phải kế thừa từ lớp này và implement
    phương thức _load_data.

    This abstract base class defines the interface for all datasets.
    Any new dataset must inherit from this class and implement the _load_data method.
    """
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self._load_and_process()

    @abc.abstractmethod
    def _load_data(self):
        """
        Phương thức riêng tư để tải dữ liệu thô.
        Mỗi lớp con phải tự định nghĩa cách đọc file dữ liệu của mình (ví dụ: csv, npz).

        Private method to load raw data.
        Each subclass must implement this to handle its specific file format (e.g., csv, npz).

        Returns:
            pd.DataFrame: DataFrame chứa toàn bộ dữ liệu.
        """
        raise NotImplementedError

    def _preprocess(self, df):
        """
        Tiền xử lý dữ liệu: xử lý các cột categorical và chuẩn hóa.

        Preprocesses the data: handles categorical features and scales numerical ones.

        Args:
            df (pd.DataFrame): The raw dataframe.

        Returns:
            (np.ndarray, np.ndarray): Features (X) and labels (y).
        """
        # Giả định cột cuối cùng là nhãn (label)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Chuyển đổi các cột categorical thành dummy variables
        X = pd.get_dummies(X, drop_first=True)

        # Chuẩn hóa các feature về khoảng [0, 1]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        return X, y.values

    def _split_data(self, X, y):
        """
        Phân chia dữ liệu thành tập train và test.
        Tự động kiểm tra điều kiện để sử dụng `stratify`.

        Splits the data into training and testing sets.
        Automatically checks if stratification is possible.
        """
        num_samples = len(y)
        num_classes = len(pd.unique(y))
        test_samples = int(num_samples * self.test_size)

        # Kiểm tra xem có nên dùng stratify hay không
        # Điều kiện: số mẫu trong tập test phải lớn hơn hoặc bằng số lớp
        should_stratify = (test_samples >= num_classes)
        
        stratify_option = y if should_stratify else None

        if not should_stratify:
            print(f"Warning: Test set size ({test_samples}) is smaller than number of classes ({num_classes})."
                " Disabling stratification for this split.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_option  # Dùng y nếu có thể, nếu không thì dùng None
        )
        print("Data split completed.")
        print(f"Train set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")



    def _load_and_process(self):
        """
        Pipeline hoàn chỉnh: Tải -> Tiền xử lý -> Phân chia.

        The complete pipeline: Load -> Preprocess -> Split.
        """
        print(f"Loading data from {self.data_path}...")
        raw_df = self._load_data()
        print("Preprocessing data...")
        X, y = self._preprocess(raw_df)
        self._split_data(X, y)
        print("Dataset ready.")

    def get_data(self):
        """
        Trả về dữ liệu đã được xử lý.

        Returns the processed and split data.
        """
        return self.X_train, self.y_train, self.X_test, self.y_test

# --- Lớp cụ thể đầu tiên của chúng ta ---
class SimpleNSLKDD(BaseDataset):
    """
    Một phiên bản đơn giản hóa của bộ dữ liệu NSL-KDD để làm ví dụ.

    A simplified version of the NSL-KDD dataset for demonstration.
    """
    def _load_data(self):
        """
        Tải dữ liệu từ một file CSV.

        Loads data from a CSV file.
        """
        # Đây chỉ là ví dụ, chúng ta sẽ tạo một file CSV giả ở bước sau
        # This is just for demonstration, we will create a dummy CSV later
        try:
            df = pd.read_csv(self.data_path, header=None)
            # Giả định rằng file gốc không có header, chúng ta sẽ thêm tên cột giả
            num_features = len(df.columns) - 1
            df.columns = [f'feature_{i}' for i in range(num_features)] + ['label']
            return df
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            print("Please create a dummy data file to proceed.")
            # Trả về một DataFrame trống để tránh crash chương trình
            return pd.DataFrame()


def get_dataset(name, data_path):
    """
    Hàm "Factory" để tạo đối tượng dataset dựa trên tên.
    Đây là chìa khóa cho sự linh hoạt.

    A factory function to create a dataset object based on its name.
    This is the key to flexibility.
    """
    dataset_map = {
        "simple_nslkdd": SimpleNSLKDD
        # Khi có dataset mới, chỉ cần thêm một dòng ở đây
        # "new_dataset": NewDatasetClass
    }

    if name not in dataset_map:
        raise ValueError(f"Dataset '{name}' not recognized. Available datasets: {list(dataset_map.keys())}")

    return dataset_map[name](data_path=data_path)
