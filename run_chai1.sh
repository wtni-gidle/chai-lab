#!/bin/bash
# Exit immediately if a command fails.
set -e

# Allowed value: "current"
# current: use the chai-lab command from the environment that is already active.
server="current"

echo "Server: $server"

usage() {
    echo ""
    echo "Please make sure all required parameters are given"
    echo "Usage: $0 <OPTIONS>"
    echo "Required Parameters:"
    echo "-i <input_path>                 Input JSON file."
    echo "-o <output_dir>                 Directory in which results will be saved."
    echo "Optional Parameters:"
    echo "-d <gpu_device>                 CUDA device ID, for example 0. (default: 0)"
    echo "-D <run_data_pipeline>          Run the data pipeline. (default: true)"
    echo "-P <run_inference>              Run model inference. (default: true)"
    echo "-J <write_input_json>           Write/update prepared JSON and resources. (default: true)"
    echo "-z <compress_fold_input>        Write prepared resources as zstd. (default: false)"
    echo "-f <compress_full_confidence>   Write detailed confidence as compressed NPZ. (default: false)"
    echo "-r <model_seeds>                One seed or comma-separated seeds, e.g. 1,2,3."
    echo "-n <diffusion_samples>          Number of samples per seed. (default: 5)"
    echo "-c <recycling_steps>            Number of trunk recycling steps. (default: 3)"
    echo "-p <sampling_steps>             Number of diffusion sampling steps. (default: 200)"
    echo "-M <use_msa_server>             Search missing protein MSAs. (default: true)"
    echo "-T <use_templates_server>       Search missing templates. (default: false)"
    echo "-m <max_template_date>          Latest searched-template release date. (default: no date filter)"
    echo "-E <use_esm_embeddings>         Generate ESM embeddings in inference. (default: true)"
    echo "-x <constraint_path>            Optional native Chai restraint CSV."
    echo "-C <fasta_names_as_cif_chains> Use input IDs as CIF chain names. (default: false)"
    echo "-S <skip>                       Skip seeds whose expected outputs exist. (default: false)"
    echo "-h                              Show this help."
    echo ""
    echo "Examples:"
    echo "  # Run only the data pipeline."
    echo "  $0 -i seq.json -o result -D true -P false"
    echo ""
    echo "  # Read seq_data.json and predict five seeds."
    echo "  $0 -i result/seq/seq_data.json -o result -D false -P true -r 1,2,3,4,5 -S true"
    exit 1
}

normalize_boolean() {
    local value="$2"
    # Match the Python CLI's case-insensitive, whitespace-trimmed spellings.
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    case "$value" in
        [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]|[Oo][Nn]) printf 'true\n' ;;
        [Ff][Aa][Ll][Ss][Ee]|0|[Nn][Oo]|[Oo][Ff][Ff]) printf 'false\n' ;;
        *) echo "Error: $1 must be true or false (got '$2')." >&2; return 2 ;;
    esac
}

# region: Parse command line arguments
while getopts "i:o:d:D:P:J:z:f:r:n:c:p:M:T:m:E:x:C:S:h" opt; do
    case "${opt}" in
    i) input_path=$OPTARG ;;
    o) output_dir=$OPTARG ;;
    d) gpu_device=$OPTARG ;;
    D) run_data_pipeline=$OPTARG ;;
    P) run_inference=$OPTARG ;;
    J) write_input_json=$OPTARG ;;
    z) compress_fold_input=$OPTARG ;;
    f) compress_full_confidence=$OPTARG ;;
    r) model_seeds=$OPTARG ;;
    n) diffusion_samples=$OPTARG ;;
    c) recycling_steps=$OPTARG ;;
    p) sampling_steps=$OPTARG ;;
    M) use_msa_server=$OPTARG ;;
    T) use_templates_server=$OPTARG ;;
    m) max_template_date=$OPTARG ;;
    E) use_esm_embeddings=$OPTARG ;;
    x) constraint_path=$OPTARG ;;
    C) fasta_names_as_cif_chains=$OPTARG ;;
    S) skip=$OPTARG ;;
    h) usage ;;
    *) usage ;;
    esac
done
# endregion

# region: Check required parameters
if [[ "$input_path" == "" || "$output_dir" == "" ]]; then
    usage
fi

if [[ ! -f "$input_path" ]]; then
    echo "Error: input JSON does not exist: $input_path"
    exit 1
fi
# endregion

# region: Set default values
if [[ "$gpu_device" == "" ]]; then gpu_device="0"; fi
if [[ "${run_data_pipeline+x}" != x ]]; then run_data_pipeline="true"; fi
if [[ "${run_inference+x}" != x ]]; then run_inference="true"; fi
if [[ "${write_input_json+x}" != x ]]; then write_input_json="true"; fi
if [[ "${compress_fold_input+x}" != x ]]; then compress_fold_input="false"; fi
if [[ "${compress_full_confidence+x}" != x ]]; then compress_full_confidence="false"; fi
if [[ "$diffusion_samples" == "" ]]; then diffusion_samples="5"; fi
if [[ "$recycling_steps" == "" ]]; then recycling_steps="3"; fi
if [[ "$sampling_steps" == "" ]]; then sampling_steps="200"; fi
if [[ "${use_msa_server+x}" != x ]]; then use_msa_server="true"; fi
if [[ "${use_templates_server+x}" != x ]]; then use_templates_server="false"; fi
if [[ "${use_esm_embeddings+x}" != x ]]; then use_esm_embeddings="true"; fi
if [[ "${fasta_names_as_cif_chains+x}" != x ]]; then fasta_names_as_cif_chains="false"; fi
if [[ "${skip+x}" != x ]]; then skip="false"; fi

run_data_pipeline=$(normalize_boolean -D "$run_data_pipeline")
run_inference=$(normalize_boolean -P "$run_inference")
write_input_json=$(normalize_boolean -J "$write_input_json")
compress_fold_input=$(normalize_boolean -z "$compress_fold_input")
compress_full_confidence=$(normalize_boolean -f "$compress_full_confidence")
use_msa_server=$(normalize_boolean -M "$use_msa_server")
use_templates_server=$(normalize_boolean -T "$use_templates_server")
use_esm_embeddings=$(normalize_boolean -E "$use_esm_embeddings")
fasta_names_as_cif_chains=$(normalize_boolean -C "$fasta_names_as_cif_chains")
skip=$(normalize_boolean -S "$skip")

if [[ "$run_data_pipeline" == "false" && "$run_inference" == "false" ]]; then
    echo "Error: run_data_pipeline and run_inference cannot both be false."
    exit 1
fi
# endregion

# region: Select executable
if [[ "$server" == "current" ]]; then
    # Activate your Chai-1 environment before running this script.
    chai_bin="chai-lab"
else
    echo "Error: server is invalid: $server"
    exit 1
fi

if ! command -v "$chai_bin" >/dev/null 2>&1; then
    echo "Error: Chai-1 executable does not exist: $chai_bin"
    exit 1
fi
# endregion

if [[ "$run_inference" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_device"
fi

#### Command arguments
command_args=(
    fold "$input_path" "$output_dir"
    --run-data-pipeline "$run_data_pipeline"
    --run-inference "$run_inference"
    --compress-fold-input "$compress_fold_input"
    --compress-full-confidence "$compress_full_confidence"
    --diffusion-samples "$diffusion_samples"
    --recycling-steps "$recycling_steps"
    --sampling-steps "$sampling_steps"
    --use-msa-server "$use_msa_server"
    --use-templates-server "$use_templates_server"
    --use-esm-embeddings "$use_esm_embeddings"
    --fasta-names-as-cif-chains "$fasta_names_as_cif_chains"
    --skip "$skip"
)

if [[ "$max_template_date" != "" ]]; then
    command_args+=(--max-template-date "$max_template_date")
fi
if [[ "$write_input_json" != "" ]]; then
    command_args+=(--write-input-json "$write_input_json")
fi
if [[ "$model_seeds" != "" ]]; then
    command_args+=(--seeds "$model_seeds")
fi
if [[ "$constraint_path" != "" ]]; then
    command_args+=(--constraint-path "$constraint_path")
fi

# Run Chai-1 with the requested parameters.
echo "$chai_bin ${command_args[*]}"
"$chai_bin" "${command_args[@]}"
