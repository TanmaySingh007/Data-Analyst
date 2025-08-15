#!/usr/bin/env python3
"""
Main execution script for the Data Analyst Project
This script orchestrates all phases of the project setup
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import pandas as pd
        import numpy as np
        import sqlalchemy
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_dataset():
    """Check if the dataset file exists"""
    dataset_path = Path("data/online_gaming_behavior.csv")
    if dataset_path.exists():
        print(f"✅ Dataset found: {dataset_path}")
        return True
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset")
        print("And save it as 'online_gaming_behavior.csv' in the data/ directory")
        return False

def run_phase(phase_name, script_name, description):
    """Run a specific phase of the project"""
    print(f"\n{'='*60}")
    print(f"RUNNING PHASE: {phase_name}")
    print(f"{'='*60}")
    print(f"Description: {description}")
    print(f"Script: {script_name}")
    
    if not os.path.exists(script_name):
        print(f"❌ Script {script_name} not found!")
        return False
    
    try:
        # Import and run the script
        if script_name == "database_setup.py":
            from database_setup import setup_database, load_data_from_sql
            engine = setup_database()
            if engine:
                df_players = load_data_from_sql(engine)
                df_players.to_pickle("df_players.pkl")
                print("✅ Phase completed successfully!")
                return True
            else:
                print("❌ Phase failed!")
                return False
        elif script_name == "data_inspection.py":
            from data_inspection import inspect_data
            inspect_data()
            print("✅ Phase completed successfully!")
            return True
        else:
            print(f"❌ Unknown script: {script_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 DATA ANALYST PROJECT - PHASE 1 SETUP")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check dataset
    if not check_dataset():
        return
    
    print("\n📋 PHASE SUMMARY:")
    print("Phase 1.1: Environment Setup ✅ (Completed)")
    print("Phase 1.2: Data Download ⏳ (Manual step required)")
    print("Phase 1.3: SQL Database Setup and Data Ingestion ⏳")
    print("Phase 1.4: Data Loading from SQL to Pandas ⏳")
    print("Phase 1.5: Initial Data Inspection ⏳")
    
    # Run Phase 1.3 & 1.4 (Database setup and data ingestion)
    success = run_phase(
        "1.3 & 1.4: Database Setup and Data Ingestion",
        "database_setup.py",
        "Set up SQLite database, create table schema, and ingest CSV data"
    )
    
    if not success:
        print("\n❌ Database setup failed. Please check the error messages above.")
        return
    
    # Run Phase 1.5 (Data inspection)
    success = run_phase(
        "1.5: Initial Data Inspection",
        "data_inspection.py",
        "Display data structure, statistics, and missing values analysis"
    )
    
    if success:
        print("\n🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("- Review the data inspection output above")
        print("- The DataFrame is saved as 'df_players.pkl'")
        print("- The database is saved as 'gaming_data.db'")
        print("- You're ready to proceed with data analysis!")
    else:
        print("\n❌ Some phases failed. Please review the error messages above.")

if __name__ == "__main__":
    main()
