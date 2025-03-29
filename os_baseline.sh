#!/bin/bash

#$ -q gpu.q
#$ -S /bin/bash
#$ -cwd
#$ -m ea
#$ -l gpu_mem=26G
#$ -l h_rt=01:55:59

module load cuda
LD_LIBRARY_PATH="/usr/local/cuda-11.0.3/lib64:/wynton/protected/home/ichs/bmiao/lib64"
export LD_LIBRARY_PATH

source ~/.bashrc
conda activate llama

export CUDA_VISIBLE_DEVICES=$SGE_GPU

gpuprof=$(dcgmi group -c mygpus -a $SGE_GPU | awk '{print $10}')
dcgmi stats -g $gpuprof -es
dcgmi stats -g $gpuprof -s $JOB_ID

starling_beta_path='starling-7b-beta'
starling_alpha_path='starling-7b-alpha'
openhermes_mistral_path="OpenHermes-2.5-Mistral-7B"
snorkel_mistral_path="Snorkel-Mistral-PairRM-DPO"
biomistral_path="BioMistral-7B"
yi_path="Yi-6B-Chat"
llama3_chat_path="llama-3-8b-chat-hf"
llama2_chat_path="llama-2-7b-chat-hf"
gemma_path="gemma-7b-it"
zephyr_path="zephyr-7b-gemma-v01"
jsl_path="JSL-MedMNX-7B-SFT"

models=($llama3_chat_path
        $starling_beta_path
        $starling_alpha_path
        $openhermes_mistral_path
        $snorkel_mistral_path
        $biomistral_path
        $yi_path
        $gemma_path
        $llama2_chat_path
        $jsl_path
        $zephyr_path)
        
for model in "${models[@]}"; do
  echo "Current model: ${model}"
  python -u os_baseline.py \
    --model_config_fpath="./utils/configs/${model}.json"\
    --data_fpath="./data/ncdmard/gpt4/validation.parquet.gzip"\
    --out_dir="./data/ncdmard/baseline" \
    --out_file_name="2024-04-21_${model}_reasons-provided_validation.csv"\
    --task='Task: Tumor necrosis factor inhibitors (TNFis) describe biologic drugs targeting TNF proteins. Using the clinical note provided, extract the following information into this JSON format: {"new_TNFi":"What new TNFi was prescribed or started? If the patient is not starting a new TNFi, write "NA"","last_TNFi":"What was the last TNFi the patient used? If none, write "NA"","reason_type_last_TNFi_stopped":"Which best describes why the last TNFi was stopped or planned to be stopped? "Adverse event", "Drug resistance", "Insurance/Cost","Lack of efficacy","Patient preference","Other", "NA"","full_reason_last_TNFi_stopped":"Provide a description for why the last TNFi was stopped or planned to be stopped?"}\nAnswer:'\
    --batch_size=1 \
    --truncate_note_to=6000
done

# stats
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"

dcgmi stats -g $gpuprof -x $JOB_ID
dcgmi stats -g $gpuprof -v -j $JOB_ID
dcgmi group -d $gpuprof

