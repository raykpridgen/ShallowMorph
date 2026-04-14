#!/bin/bash

#SBATCH --job-name="data_loading_pde_rkp"

#SBATCH --output="rkp_data.%j.%N.out"

#SBATCH --partition=shared

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=4G

#SBATCH --account=ccu108

#SBATCH --export=ALL

#SBATCH -t 06:00:00



module purge

module load cpu/0.15.4

module load aria2

# Fail on unset vars, but allow command failures (we handle them manually)

set -u

# Completed
#    "https://darus.uni-stuttgart.de/api/access/datafile/268187"
#    "https://darus.uni-stuttgart.de/api/access/datafile/133021"


URLS=(
    "https://darus.uni-stuttgart.de/api/access/datafile/133017"
)


# 	"1D.hdf5"
#	"2D.hdf5"

names=(
	"2D_diff.h5"
)


download_file() {

    local url="$1"

    local filename="$2"

    local token_header=""



    if [ -n "${API_TOKEN:-}" ]; then

        token_header="--header=X-Dataverse-key:${API_TOKEN}"

    fi



    echo "=== Downloading: $filename from $url ==="



    # aria2c with explicit redirect following and more connections

    aria2c --max-connection-per-server=16 --split=16 --min-split-size=1M -c --allow-overwrite=true --follow-metalink=true --follow-torrent=true -o "$filename" "$url"



    if [ $? -eq 0 ]; then

        echo "✓ aria2c succeeded"

        return 0

    fi



    echo "⚠ aria2c failed, trying wget with redirect following..."

    wget --continue --tries=5 --retry-connrefused \

         --max-redirect=10 \

         ${API_TOKEN:+--header="X-Dataverse-key:${API_TOKEN}"} \

         -O "$filename" "$url"



    if [ $? -eq 0 ]; then

        echo "✓ wget succeeded"

        return 0

    fi



    echo "✗ Both failed for $filename"

    return 1

}

for i in {0..2};
do

    download_file "${URLS[i]}" "${names[i]}" || echo "Error downloading $url"

done

echo "All downloads attempted."
