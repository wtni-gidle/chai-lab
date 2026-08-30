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
    echo "-r <model_seeds>                One seed or comma-separated seeds, e.g. 1,2,3."
    echo "-n <diffusion_samples>          Number of samples per seed. (default: 5)"
    echo "-c <recycling_steps>            Number of trunk recycling steps. (default: 3)"
    echo "-p <sampling_steps>             Number of diffusion sampling steps. (default: 200)"
    echo "-M <use_msa_server>             Search missing protein MSAs. (default: true)"
    echo "-T <use_templates_server>       Search missing templates. (default: false)"
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

# region: Parse command line arguments
while getopts "i:o:d:D:P:r:n:c:p:M:T:S:h" opt; do
    case "${opt}" in
    i) input_path=$OPTARG ;;
    o) output_dir=$OPTARG ;;
    d) gpu_device=$OPTARG ;;
    D) run_data_pipeline=$OPTARG ;;
    P) run_inference=$OPTARG ;;
    r) model_seeds=$OPTARG ;;
    n) diffusion_samples=$OPTARG ;;
    c) recycling_steps=$OPTARG ;;
    p) sampling_steps=$OPTARG ;;
    M) use_msa_server=$OPTARG ;;
    T) use_templates_server=$OPTARG ;;
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
if [[ "$run_data_pipeline" == "" ]]; then run_data_pipeline="true"; fi
if [[ "$run_inference" == "" ]]; then run_inference="true"; fi
if [[ "$diffusion_samples" == "" ]]; then diffusion_samples="5"; fi
if [[ "$recycling_steps" == "" ]]; then recycling_steps="3"; fi
if [[ "$sampling_steps" == "" ]]; then sampling_steps="200"; fi
if [[ "$use_msa_server" == "" ]]; then use_msa_server="true"; fi
if [[ "$use_templates_server" == "" ]]; then use_templates_server="false"; fi
if [[ "$skip" == "" ]]; then skip="false"; fi

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
    --diffusion-samples "$diffusion_samples"
    --recycling-steps "$recycling_steps"
    --sampling-steps "$sampling_steps"
    --use-msa-server "$use_msa_server"
    --use-templates-server "$use_templates_server"
    --skip "$skip"
)

if [[ "$model_seeds" != "" ]]; then
    command_args+=(--seeds "$model_seeds")
fi

# Run Chai-1 with the requested parameters.
echo "$chai_bin ${command_args[*]}"
"$chai_bin" "${command_args[@]}"
