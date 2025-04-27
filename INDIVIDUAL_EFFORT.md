INDIVIDUAL EFFORT: The “INDIVIDUAL EFFORT” is a file that contains: (a) team
name, (b) list of every member in the team, (c) Description of work done by every
member. Overlapping text is discouraged. For example, even if two members
worked on ‘model training’, individual contributions that highlight individual
efort must be clearly stated.

# Team: How To Rob A Bank
Project name: HoudiniLM

## Chris Yoo
I've implemented the entirety of the rule-based model (WordNet synonyms + jailbreaking prefix/suffix). 

I've written a significant portion of the report, 

Cleaned the entire codebase

StrongREJECT evaluation 

## Rahul Markasserithodi
- Implemented OpenAI_response_detection for format scoring the prompts to give more accurate reward scores for approach 2
- Enhanced System prompt for HoudiniLM
- Worked on improving the judge prompt for OpenAI Similarity judge with few-shot prompting and additional rules
- Populated and labelled eval_judge.csv with 40+ diverse prompts for eval from Beavertails.
- Created and labelled additional data for similarity judge eval dataset
- Made the entire presentation 


## Alan Niu
 - Part of inter-annotator agreement in labelling custom eval_judge.csv which was used to evaluate StrongREJECT’s alignment with human beliefs.
 - Also created and labelled examples in similarity_evals.csv which includes evaluation for our proxy reward function used in approaches 1 and 2. 
 - Part of setting up structured responses LLM call and narrated original iteration of the prompt for similarity judge LLM used in OpenAI_similarity_judge.py. 
 - Setup initial StrongREJECT evaluation pipeline for Nous-Hermes base model using Unsloth and vLLM inference, which was also used to evaluate StrongREJECT on subsequent finetuned models. 
 - Created evaluation script similarity_evals.ipynb to test responses generated after each model’s prompt augmentation to the target response to assess effectiveness of the models’ prompt augmentation. Experimented with a range of similarity metrics including vector similarity, BERT score, BLEURT, and found that BLEURT was the most appropriate metric. 	
 - Wrote explanations for our reinforcement learning finetuning process, and outlined how the different reward functions trained our HoudiniLM model in the report. 
 - Filmed demo and key insights section of the presentation. 


## Ishmanbir Singh

