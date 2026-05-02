from load_ibs_2015 import load_ibs_2015
from load_kchs_2021 import load_kchs_2021

def load_knbs_datasets() -> None :
    print("\nLoading KCHS 2021 dataset...")
    load_kchs_2021() 
    print("\nKCHS 2021 dataset loaded.")

    print("\nLoading IBS 2015 dataset...")
    load_ibs_2015()
    print("\nIBS 2015 dataset loaded.")

    print("\nAll KNBS datasets loaded and saved.")

def main() : 
    load_knbs_datasets() 

if __name__ == "__main__" : 
    main()