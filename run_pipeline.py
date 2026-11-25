"""Run the whole project:  python run_pipeline.py"""
from src.analysis import run_basic_analysis, run_text_analysis
from src.data_loader import load_raw_data, save_processed_data
from src.preprocessing import preprocess_dataframe
from src.train import train_and_save


def main():
    print("1/4 Loading raw data...")
    df = load_raw_data()

    print("2/4 Analyzing raw data...")
    run_basic_analysis(df)

    print("3/4 Cleaning text (may take a minute)...")
    df = preprocess_dataframe(df)
    save_processed_data(df)
    run_text_analysis(df)

    print("4/4 Training models...")
    results = train_and_save(df)
    print(results)
    print("\nDone! Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
