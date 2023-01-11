# Gene Expression Analysis

## Data Source

The dataset was aggregated from many source files from a publicly available database.
https://portal.gdc.cancer.gov/

## Project Navigation
* csv
    * data.csv | raw complimation of every patient and every data field
    * SurvivalNormal.csv | normalized ovarian cancer patients with just gene fields and survival outcome
    * SurvivalNormalBoth.csv | normalized ovarian and breast cancer patients with gene fields, age, cancer information, and prognosis.
    * ProgressionNormal.csv | normalized ovarian cancer patients with gene fields and progression outcome
    * RecurrenceNormal.csv | normalized ovarian cancer patients with gene fields and recurrence outcome
* R_Analysis
    * analysis.Rmd | R markdown script that conducts analysis of dataset, creating different types of models
    * gen_csv.Rmd | R markdown script that generates SurvivalNormalBoth.csv from data.csv
* Hiearchical_Clustering
    * hierarchical_clustering.py | python script that generates the figures for hierarchical clustering
* NeuralNetwork
    * choosing_genes.py | the python script for selecting the optimal features
    * graphing_feature_selection.py | python script that graphs the results of choosing_genes.py
    * graphing_neural_performance.py | python script that graphs the performance of the neural networks
    * neural_model.py | python script that includes functions for building a model on the dataset. Also includes a general evaluation of the a model
    * SurvivalExperiment.txt | the results of choosing_genes.py optimizing survival predictions    
    * ProgressionExperiment.txt | the results of choosing_genes.py optimizing progression predictions
    * RecurrenceExperiment.txt | the results of choosing_genes.py optimizing recurrence predictions
* Learning_Curve
    * graphing_learning_curve.py | graphs the learning curves from output files (hard-coded names)
    * learning_curve.py | generates a learning curve datafile on neural network
    * learning_curve.Rmd | generates a learning curve datafile on logistic regression model
* README.md | a project description
* LICENSE | the license for this project


## Dataset Fields

##### General Info
> Patient ID, Cancer Type, Survival Status, Survival in Days (7300 represents continued survival), Progression Status, Recurrence Status, Age, Race

##### Genes
> XRCC6, XRCC5, XRCC7, LIG4, LIG3, LIG1, XRCC4, NHEJ1, XRCC1, DCLRE1C, TP53BP1, BRCA1, BRCA2, EXO1, EXD2, POLM, POLL, POLQ, RAD50, MRE11, NBN, TDP1, RBBP8, CTBP1, APLF, PARP1, PARP3, PNKP, APTX, WRN, PAXX, RIF1, RAD52, RAD51, ATM, ATR, TP53, H2AX, ERCC1, ERCC4, RPA1, MSH2, MSH3, RAD1, MSH6, PMS1, MLH1, MLH3

##### Tumor Information
> ICD10 Code, Figo Stage, Neoplasm Histologic Grade, Tumor Size, Lymphatic Invasion Status

##### Treatment Information
> Total Dose, Number of Treatment Cycles, Treatment Day Range, Therapy Type, Drug Name, Regimen Indication

##### Null Fields
> Fields are null because they are either missing, unavailable, or irrelevant to the original study.



## Dataset Composition
- Ovarian Count: 374
- Breast Count: 279
- Ovarian Count Alive: 145 Deceased: 229
- Breast Count Alive: 137 Deceased: 142
- Mean: Ovarian Length Survival (yr): 9.741966285945132
- Mean: Breast Length Survival (yr): 12.21966448282708
- Standard Deviation: Ovarian Length Survival (yr): 8.349660685645034
- Standard Deviation: Breast Length Survival (yr): 8.12829931819985
- Mean: Ovarian Age (yr): 59.58288770053476
- Standard Deviation: Ovarian Age (yr): 11.343556969016111

##### Ovarian Cancer Aggregated Figo Stage
| Type | Total | Percentage |
| --- | --- | --- |
| Stage IIIC | 271 | 0.72460 |
| Stage IIC | 15 | 0.04011 |
| Stage IIB | 3 | 0.00802 |
| Stage IV | 57 | 0.15241 |
| Stage IIIA | 7 | 0.01872 |
| UNKNOWN | 3 | 0.00802 |
| Stage IIIB | 14 | 0.03743 |
| Stage IC | 1 | 0.00267 |
| Stage IIA | 3 | 0.00802 |

##### Breast Cancer Aggregated Figo Stage
| Type | Total | Percentage |
| --- | --- | --- |
| UNKNOWN | 279 | 1.00000 |

##### Ovarian Cancer Aggregated Race
| Type | Total | Percentage |
| --- | --- | --- |
| white | 324 | 0.86631
| asian | 11 | 0.02941
| african american | 25 | 0.06684
| UNKNOWN | 11 | 0.02941
| american indian or alaska native | 2 | 0.00535
| native hawaiian or other pacific islander | 1 | 0.00267

##### Breast Cancer Aggregated Race
| Type | Total | Percentage |
| --- | --- | --- |
| UNKNOWN | 279 | 1.00000 |

##### Ovarian Cancer Aggregated ICD 10 Code
| Type | Total | Percentage |
| --- | --- | --- |
| C56.9 | 369 | 0.98663 |
| C48.1 | 4 | 0.01070 |
| C48.2 | 1 | 0.00267 |

##### Breast Cancer Aggregated ICD 10 Code
| Type | Total | Percentage |
| --- | --- | --- |
| UNKNOWN | 279 | 1.00000 |

##### Ovarian Cancer Aggregated Neoplasm Histologic Grade
| Type | Total | Percentage |
| --- | --- | --- |
| G3 | 320 | 0.85561 |
| G2 | 42 | 0.11230 |
| G1 | 1 | 0.00267 |
| GB | 2 | 0.00535 |
| GX | 6 | 0.01604 |
| UNKNOWN |2 | 0.00535 |
| G4 | 1 | 0.00267 |

##### Breast Cancer Aggregated Neoplasm Histologic Grade
| Type | Total | Percentage |
| --- | --- | --- |
| UNKNOWN | 279 | 1.00000 |

##### Ovarian Cancer Aggregated Tumor Residual Disease
| Type | Total | Percentage |
| --- | --- | --- |
| Microscopic | 64 | 0.17112 |
| 11-20 mm | 26 | 0.06952 |
| UNKNOWN | 43 | 0.11497 |
| 1-10 mm | 171 | 0.45722 |
| >20 mm | 70 | 0.18717 |

##### Breast Cancer Aggregated Tumor Residual Disease
| Type | Total | Percentage |
| --- | --- | --- |
| UNKNOWN | 279 | 1.00000 |

##### Ovarian Cancer Aggregated Lymphatic Invasion
| Type | Total | Percentage |
| --- | --- | --- |
| NO | 47 | 0.12567 |
| UNKNOWN | 227 | 0.60695 |
| YES | 100 | 0.26738 |

##### Breast Cancer Aggregated Lymphatic Invasion
| Type | Total | Percentage |
| --- | --- | --- |
| UNKNOWN | 279 | 1.00000 |

##### Progression Aggregated
| Type | Total | Percentage |
| --- | --- | --- |
| FALSE | 222 | 0.59358 |
| TRUE | 100 | 0.26738 |
| UNKNOWN | 52 | 0.13904 |

##### Recurrence Aggregated
| Type | Total | Percentage |
| --- | --- | --- |
| FALSE | 190 | 0.50802 |
| TRUE | 132 | 0.35294 |
| UNKNOWN | 52 | 0.13904 |
