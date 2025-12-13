#!/bin/bash

################################################################################
# Plastic-Degrading Microorganism Genome Download Pipeline
# Data outputs: /data/bacteria
# Logs & metadata: $HOME
################################################################################

############################
# Configuration
############################

# Input data
TSV_URL="https://plasticdb.org/static/degraders_list.tsv"
TSV_FILE="$HOME/degraders_list.tsv"

# Biological data output directory
DATA_DIR="/data/bacteria"
ORGANISM_LIST="$DATA_DIR/organism_names.txt"
GENOME_DIR="$DATA_DIR"

# Logs & tracking (remain in HOME)
LOG_FILE="$HOME/genome_download.log"
ERROR_LOG="$HOME/genome_download_errors.log"
SUCCESS_LOG="$HOME/successful_downloads.txt"
FAILED_LOG="$HOME/failed_downloads.txt"

# Assembly preference
ASSEMBLY_LEVEL="complete"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

################################################################################
# Logging function
################################################################################
log_message() {
    local level=$1
    shift
    local message="$@"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

################################################################################
# Dependency check
################################################################################
check_dependencies() {
    log_message "INFO" "Checking dependencies..."

    local missing=0
    for cmd in wget awk datasets unzip gzip; do
        if ! command -v "$cmd" &> /dev/null; then
            echo -e "${RED}ERROR: Required command '$cmd' not found${NC}"
            log_message "ERROR" "Required command '$cmd' not found"
            missing=1
        fi
    done

    if [[ $missing -eq 1 ]]; then
        log_message "ERROR" "Missing dependencies. Please install required tools."
        exit 1
    fi

    log_message "INFO" "All dependencies satisfied ✓"
}

################################################################################
# Step 1: Download TSV (if missing)
################################################################################
download_tsv() {
    if [[ ! -f "$TSV_FILE" ]]; then
        echo -e "${BLUE}Downloading PlasticDB TSV...${NC}"
        log_message "INFO" "Downloading TSV from $TSV_URL"

        if ! wget -O "$TSV_FILE" "$TSV_URL" 2>>"$ERROR_LOG"; then
            log_message "ERROR" "Failed to download TSV file"
            exit 1
        fi

        log_message "INFO" "TSV file downloaded successfully ✓"
    else
        log_message "INFO" "TSV file already exists, skipping download"
    fi
}

################################################################################
# Step 2: Extract organism names
################################################################################
extract_organisms() {
    if [[ ! -f "$ORGANISM_LIST" ]]; then
        echo -e "${BLUE}Extracting organism names...${NC}"
        log_message "INFO" "Extracting organism names to $ORGANISM_LIST"

        awk -F'\t' 'NR > 1 && $1 != "" {print $1}' "$TSV_FILE" \
            | sort | uniq > "$ORGANISM_LIST"

        local count
        count=$(wc -l < "$ORGANISM_LIST")
        log_message "INFO" "Extracted $count unique organisms"
        echo -e "${GREEN}Found $count unique organisms${NC}"
    else
        local count
        count=$(wc -l < "$ORGANISM_LIST")
        log_message "INFO" "Organism list already exists ($count organisms)"
    fi
}

################################################################################
# Step 3: Setup directories
################################################################################
setup_directories() {
    mkdir -p "$DATA_DIR"
    log_message "INFO" "Biological data directory: $DATA_DIR"

    touch "$SUCCESS_LOG" "$FAILED_LOG"
}

################################################################################
# Step 4: Download genomes
################################################################################
download_genomes() {
    local total
    total=$(wc -l < "$ORGANISM_LIST")

    log_message "INFO" "Total organisms queued for download: $total"
    echo -e "${BLUE}Total organisms queued for download: ${total}${NC}"

    local current=0 skipped=0 downloaded=0 failed=0

    while IFS= read -r organism; do
        [[ -z "$organism" ]] && continue
        ((current++))

        echo -e "\n${YELLOW}[${current}/${total}] Processing: $organism${NC}"
        log_message "INFO" "Processing organism $current/$total: $organism"

        sanitized="${organism// /_}"
        fasta_file="$GENOME_DIR/${sanitized}.fasta"
        zip_file="$GENOME_DIR/${sanitized}.zip"

        if [[ -f "$fasta_file" ]]; then
            echo -e "${GREEN}✓ Genome already exists, skipping${NC}"
            ((skipped++))
            continue
        fi

        datasets download genome taxon "$organism" \
            --include genome \
            --assembly-source refseq \
            --assembly-level "$ASSEMBLY_LEVEL" \
            --filename "$zip_file" 2>>"$ERROR_LOG"

        if [[ ! -s "$zip_file" ]]; then
            datasets download genome taxon "$organism" \
                --include genome \
                --assembly-source genbank \
                --assembly-level "$ASSEMBLY_LEVEL" \
                --filename "$zip_file" 2>>"$ERROR_LOG"
        fi

        if [[ -s "$zip_file" ]]; then
            temp_dir="$GENOME_DIR/tmp_${sanitized}"
            mkdir -p "$temp_dir"

            if unzip -q "$zip_file" -d "$temp_dir" 2>>"$ERROR_LOG"; then
                fna_file=$(find "$temp_dir" -type f -name "*.fna" | head -n 1)

                if [[ -f "$fna_file" ]]; then
                    mv "$fna_file" "$fasta_file"
                    gzip -c "$fasta_file" > "${fasta_file}.gz"

                    echo -e "${GREEN}✓ Saved:${NC} $(basename "$fasta_file") + .gz"
                    log_message "INFO" "Downloaded genome for '$organism'"
                    echo "$organism" >> "$SUCCESS_LOG"
                    ((downloaded++))
                else
                    echo "$organism - No .fna file" >> "$FAILED_LOG"
                    ((failed++))
                fi
            else
                echo "$organism - Extraction failed" >> "$FAILED_LOG"
                ((failed++))
            fi

            rm -rf "$temp_dir" "$zip_file"
        else
            echo "$organism - No genome available" >> "$FAILED_LOG"
            ((failed++))
        fi

    done < "$ORGANISM_LIST"

    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Download Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Total organisms: $total"
    echo -e "${GREEN}Downloaded: $downloaded${NC}"
    echo -e "${GREEN}Skipped: $skipped${NC}"
    echo -e "${RED}Failed: $failed${NC}"

    log_message "INFO" "Pipeline complete - Total:$total Downloaded:$downloaded Skipped:$skipped Failed:$failed"
}

################################################################################
# Main
################################################################################
main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║ Plastic-Degrading Microorganism Genome Pipeline       ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    log_message "INFO" "===== Pipeline Started ====="
    log_message "INFO" "Assembly level: $ASSEMBLY_LEVEL"

    check_dependencies
    download_tsv
    extract_organisms
    setup_directories
    download_genomes

    log_message "INFO" "===== Pipeline Completed Successfully ====="
    echo -e "${GREEN}✓ Pipeline completed successfully${NC}"
}

main "$@"
