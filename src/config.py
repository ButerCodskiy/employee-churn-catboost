"""Централизованная конфигурация и константы проекта."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ARCHIVE_NAMES = [
    "archive (10) (1).zip",
    "archive (10).zip",
    "archive.zip",
]

RANDOM_SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.2

TARGET_COL = "Attrition"
TARGET_MAPPING = {"Yes": 1, "No": 0, 1: 1, 0: 0}

DROP_COLS = [
    "Over18",
    "StandardHours",
    "EmployeeCount",
    "EmployeeNumber",
]

RAW_CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "MaritalStatus",
    "Gender",
    "OverTime",
]

RAW_NUMERICAL_COLS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

MEDIAN_MONTHLY_INCOME = 6500.0
COST_FN = 7.5 * MEDIAN_MONTHLY_INCOME
COST_FP = 0.5 * MEDIAN_MONTHLY_INCOME
COST_TP = 1.0 * MEDIAN_MONTHLY_INCOME
COST_TN = 0.0
