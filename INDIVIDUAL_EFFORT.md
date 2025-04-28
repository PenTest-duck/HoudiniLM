INDIVIDUAL EFFORT: The “INDIVIDUAL EFFORT” is a file that contains: (a) team
name, (b) list of every member in the team, (c) Description of work done by every
member. Overlapping text is discouraged. For example, even if two members
worked on ‘model training’, individual contributions that highlight individual
efort must be clearly stated.

# Team: How To Rob A Bank
Project name: HoudiniLM

## Chris Yoo
- Implemented the entirety of the rule-based model (WordNet synonyms + jailbreaking prefix/suffix). 
- Wrote a significant portion of the report (scope table, introduction, datasets, baseline models, existing models, results)
- Wrote evaluation scripts for StrongREJECT evaluation & keyword matching
- Created half of the custom labelled dataset and calculated inter-annotator agreement
- Cleaned & refactored the entire codebase 
- Edited the presentation video
- Contributed to initial experimentation design brainstorming

## Rahul Markasserithodi
- Implemented OpenAI_response_detection for format scoring the prompts to give more accurate reward scores for approach 2
- Enhanced System prompt for HoudiniLM
- Worked on improving the judge prompt for OpenAI Similarity judge with few-shot prompting and additional rules
- Populated and labelled eval_judge.csv with 40+ diverse prompts for eval from Beavertails.
- Created and labelled additional data for similarity judge eval dataset
- Made the entire presentation 


## Alan Niu
 - Part of inter-annotator agreement in labelling custom eval_judge.csv, used to evaluate StrongREJECT’s alignment with human beliefs.
 - Created and labelled examples in similarity_evals.csv which includes evaluation for our proxy reward function. 
 - Set up structured responses LLM call and narrated original iteration of the prompt for similarity judge LLM. 
 - Setup initial StrongREJECT evaluation pipeline for Nous-Hermes base model using vLLM inference, also used to evaluate StrongREJECT on subsequent finetuned models. 
 - Created evaluation script similarity_evals.ipynb to assess models using BLEURT.
 - Wrote explanations for our entire RL finetuning process in the report, including the different reward functions. 
 - Filmed demo and key insights section of the presentation. 


## Ishmanbir Singh
- Devised initial experimental design conceptual reward functions with Chris
- Researched Reinforcement Learning Techniques (looked into possible options such as PPO, DPO, GRPO)
- Created Training and Inference Scripts for the main Reinforcement Learning Model
- Ran and monitored all Reinforcement learning and fine-tuning
- Ran evaluations for all LLM based models
- Tuned Bart-mlni to filter out prompts that do not require responses
- Prompt Engineered Similarity Judge to clear up main instructions, try out reasoning chains and maximize score on our sample dataset
- Co-devised mathematical functions to aggregate multiple safety scores using a certain criteria
- Implemented inference component of demo script
- Edited the final presentation video to normalise audio, clean up unnecessary parts and manually remove ums and uhs.
- Hosted Lora Model Weights and Final Presentation Video

