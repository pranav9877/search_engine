import pandas as pd
from typing import Tuple, List

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and preprocess laptop CSV data.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Preprocessed DataFrame and text descriptions.
    """
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Validate price column
        if 'price' not in df.columns:
            print("Error: 'price' column not found in CSV. Available columns:", df.columns.tolist())
            raise KeyError("'price' column missing")
        
        # Handle missing values
        df.fillna({
            'name': 'Unknown',
            'Processor': 'Unknown',
            'Processor_Gen': 'Unknown',
            'RAM': 'Unknown',
            'RAM_Type': 'Unknown',
            'Operating_System': 'Unknown',
            'SSD': 'Unknown',
            'Display_Size': 0.0,
            'Warranty': 'Unknown',
            'Graphics': 'Unknown',
            'Other_Specs': '',
            'price': 0,
            'rating': 0.0,
            'url': ''
        }, inplace=True)
        
        # Convert price to numeric
        def clean_price(price):
            try:
                if isinstance(price, (int, float)):
                    return float(price)
                price_str = str(price).replace('₹', '').replace(',', '').strip()
                return float(price_str) if price_str else 0.0
            except (ValueError, TypeError):
                print(f"Warning: Invalid price format '{price}', defaulting to 0.0")
                return 0.0
        
        df['price'] = df['price'].apply(clean_price)
        
        # Process display size
        def clean_display_size(size):
            try:
                # Handle numeric values directly
                if isinstance(size, (int, float)):
                    return float(size)
                # Handle string values
                size_str = str(size).strip().lower()
                # Extract numeric part (e.g., "39.62 cm" -> 39.62)
                match = pd.Series([size_str]).str.extract(r'(\d+\.?\d*)')[0][0]
                return float(match) if match else 0.0
            except (ValueError, TypeError, AttributeError):
                print(f"Warning: Invalid display size format '{size}', defaulting to 0.0")
                return 0.0
        
        df['Display_Size'] = df['Display_Size'].apply(clean_display_size)
        
        # Create text description
        df['Description'] = df.apply(
            lambda row: (
                f"{row['name']} with {row['Processor']} {row['Processor_Gen']}, "
                f"{row['RAM']} {row['RAM_Type']} RAM, {row['SSD']} storage, "
                f"{row['Operating_System']} operating system, {row['Display_Size']} cm display, "
                f"{row['Graphics']} graphics. {row['Other_Specs']}"
            ).strip(),
            axis=1
        )
        
        # Validate descriptions
        if df['Description'].str.strip().eq('').any():
            print("Warning: Some descriptions are empty.")
        
        descriptions = df['Description'].tolist()
        print(f"Preprocessed {len(df)} rows successfully.")
        return df, descriptions
    
    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}.")
        raise
    except KeyError as e:
        print(f"Error: {str(e)}")
        raise
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        raise