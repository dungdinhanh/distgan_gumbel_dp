## **Differential Privacy Generative Adversarial Networks**
This is the code for differential private GAN for generating private synthetic data.

**
**

## Requirements

 - Python 2.7 or 3.x, Numpy, scikit-learn, Tensorflow, Keras (1D demo)  
	For Python 3.x - `import pickle` in the file `distgan_image/modules/dataset.py`
 - For data imputation, need Python 3.5 and scikit-learn 0.21rc2 

**

## Running Process

 - Data Imputation: $python cervical_cancer_data_impute.py (only work on python3.5)
 - Change the paramters in the file distgan_cervical_cancer.py 
	 out_dir =  'output/' # output directory

	db_name =  'cervical_cancer' # datasetname for experiment

	data_source =  './data/cervical_cancer2/' # folder contains data 
	
	categorical_softmax_use = 3 (0: No softmax, 1:, 2:, 3: Gumbel softmax)
						
 - For Training $ python2 distgan_cervical_cancer.py  --train 0 --step 100000
 - Result will appear in the folder './output/cervical_cancer_distgan_fcnet_log_.../cervical_cancer/'
 - For Testing $ python2 distgan_cervical_cancer.py  --train 4 --step 100000

**