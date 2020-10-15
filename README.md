# HPOFiller

**HPOFiller: identifying missing protein-phenotype associations by graph convolutional network**

*Please run the programs in order.*


## Dependencies

Our model is implemented by Python 3.6 with Pytorch 1.4.0 and Pytorch-geometric 1.5.0, and run on Nvidia GPU with CUDA 10.0.

## Preprocessing

- `extract_gene_id.py`: First, please download gene annotations file from [http://compbio.charite.de/jenkins/job/hpo.annotations.monthly/](http://compbio.charite.de/jenkins/job/hpo.annotations.monthly/) with all sources and all frequencies: `ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt`. Then run the script, you will get a .txt file containing all gene ids. Finally, please upload this file to [http://www.uniprot.org/mapping/](http://www.uniprot.org/mapping/) to map Entrez Gene ID to UniProt ID.

### Cross-validation

- `create_annotation.py`: After generating Gene ID mapping file, you can run this script to generate protein-HPO annotations file without propagation. The output json file contains leaf annotations of each protein, like

	```
	{ protein_id1: [ hpo_term1, hpo_term2, ... ],
  	  protein_id2: [ hpo_term1, hpo_term2, ... ],
     ...
    }
	```

- `create_auxiliary_file_pa.py `: Now, you can create necessary auxiliary files, including: 
	- **protein list**: a json file containing all protein IDs
	- **term list**: a json file containing all HPO terms (used to annotate at least one protein)
	Note that we only keep HPO terms in PA sub-ontology.

- `split_train_test_pa.py `: Run this script to split `n_folds` folds and then generate `n_folds` mask files which contain train and test mask.

### Temporal validation

- `split_temporal_dataset_pa.py`: make necessary datasets along with the time. Note that we only consider HPO terms in PA.

## Similarity Networks Generation
### STRING

- `string.py`: Please firstly open [https://string-db.org/cgi/download.pl](https://string-db.org/cgi/download.pl) and choose "organism" as "Homo sapiens", then download "9606.protein.links.v11.0.txt.gz" (version number may change). Meanwhile, download mapping file under "ACCESSORY DATA" category, or open website
[https://string-db.org/mapping\_files/uniprot\_mappings/](https://string-db.org/mapping_files/uniprot_mappings/) to download it. After downloading, you can run this code to get a json file containing PPI data organized as

	```
	{
		protein1: {
			protein1a: score1a, 
			protein1b: score1b, 
			...
		}, 
		...
	}
	```
Here the scores are scaled to [0, 1].

### HPO Semantic Similarity

- `hpo_sim.py`: We choose **Information coefficient measure** (IC) as HPO similarity measure. Please set `'method'` in config file as `'ic'`. The generated json file containing IC similarity is organized as

	```
	{
		term1: {
			term1a: score1a, 
			term1b: score1b, 
			...
		}, 
		...
	}
	```

## Run the model

- `train.py`: We provide two modes:
	- Cross-validation: The program will conduct 10-folds CV and output corresponding predictions. Please set `'mode'` in config file as `'cv'`.
	- Temporal validation: This is a simulated real scene. The model will predict missing protein-HPO term associations based on current HPO annotations. Please set `'mode'` in config file as `'single'`.

## Evaluation

- `evaluation.py`: We provide three kinds of metrics to evaluate the performance of our model:
	- AUC: calculated on the whole pairs of protein-HPO term in the test set
	- AUPR: calculated on the whole pairs of protein-HPO term in the test set
	- AP@K: average precision at k, where k = 5000, 10000, 20000, 50000, also calculated on the whole pairs of protein-HPO term in the test set
	
	Please note that there are two running modes:
	
	- Cross-validation: Evaluate on each fold. Please set `'mode'` in config file as `'cv'`.
	- Temporal validation: Evaluate on single prediction result. Please set `'mode'` in config file as `'single'`.

