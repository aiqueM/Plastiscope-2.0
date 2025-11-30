#!/bin/bash

################################################################################
# Plastic-Degrading Microorganism Genome Download Pipeline
################################################################################

# Define variables
TSV_URL="https://plasticdb.org/static/degraders_list.tsv"
TSV_FILE="degraders_list.tsv"
ORGANISM_LIST="organism_names.txt"
GENOME_DIR="$HOME/plasticDB/genomes"
LOG_FILE="genome_download.log"
ERROR_LOG="genome_download_errors.log"
SUCCESS_LOG="successful_downloads.txt"
FAILED_LOG="failed_downloads.txt"

# Assembly level preference (can be changed to "chromosome" or "scaffold" if needed)
ASSEMBLY_LEVEL="complete"

# Color codes for better terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Function: Log messages to file and console
################################################################################
log_message() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

################################################################################
# Function: Check if required commands exist
################################################################################
check_dependencies() {
    log_message "INFO" "Checking dependencies..."
    
    local missing_deps=0
    
    for cmd in wget awk datasets unzip; do
        if ! command -v $cmd &> /dev/null; then
            echo -e "${RED}ERROR: Required command '$cmd' not found${NC}"
            log_message "ERROR" "Required command '$cmd' not found"
            missing_deps=1
        fi
    done
    
    if [ $missing_deps -eq 1 ]; then
        log_message "ERROR" "Missing dependencies. Please install required tools."
        exit 1
    fi
    
    log_message "INFO" "All dependencies satisfied ✓"
}

################################################################################
# Step 1: Download the TSV file ONLY if it doesn't exist
################################################################################
download_tsv() {
    if [[ ! -f "$TSV_FILE" ]]; then
        echo -e "${BLUE}Downloading microorganism data from PlasticDB...${NC}"
        log_message "INFO" "Downloading TSV file from $TSV_URL"
        
        if wget --content-disposition -O "$TSV_FILE" "$TSV_URL" 2>> "$ERROR_LOG"; then
            log_message "INFO" "TSV file downloaded successfully ✓"
        else
            log_message "ERROR" "Failed to download TSV file"
            echo -e "${RED}ERROR: Failed to download TSV file. Check $ERROR_LOG${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}TSV file already exists. Skipping download.${NC}"
        log_message "INFO" "TSV file already exists, skipping download"
    fi
}

################################################################################
# Step 2: Extract organism names (only if the list is missing)
################################################################################
extract_organisms() {
    if [[ ! -f "$ORGANISM_LIST" ]]; then
        echo -e "${BLUE}Extracting organism names...${NC}"
        log_message "INFO" "Extracting unique organism names from TSV"
        
        # Extract, sort, remove duplicates, and filter out empty lines
        if awk -F'\t' 'NR > 1 && $1 != "" {print $1}' "$TSV_FILE" | sort | uniq > "$ORGANISM_LIST"; then
            local count=$(wc -l < "$ORGANISM_LIST")
            log_message "INFO" "Extracted $count unique organisms ✓"
            echo -e "${GREEN}Found $count unique organisms${NC}"
        else
            log_message "ERROR" "Failed to extract organism names"
            echo -e "${RED}ERROR: Failed to extract organism names${NC}"
            exit 1
        fi
    else
        local count=$(wc -l < "$ORGANISM_LIST")
        echo -e "${GREEN}Organism names file already exists ($count organisms). Skipping extraction.${NC}"
        log_message "INFO" "Organism list already exists with $count entries"
    fi
}

################################################################################
# Step 3: Create genome directory if it doesn't exist
################################################################################
setup_directories() {
    mkdir -p "$GENOME_DIR"
    log_message "INFO" "Genome directory: $GENOME_DIR"
    
    # Initialize tracking files
    touch "$SUCCESS_LOG" "$FAILED_LOG"
}

################################################################################
# Step 4: Download genomes with improved error handling
################################################################################
download_genomes() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Starting genome downloads...${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    
    local total=$(wc -l < "$ORGANISM_LIST")
    local current=0
    local skipped=0
    local downloaded=0
    local failed=0
    
    while IFS= read -r organism; do
        # Skip empty lines
        if [[ -z "$organism" ]]; then
            continue
        fi
        
        ((current++))
        
        echo -e "\n${YELLOW}[${current}/${total}] Processing: $organism${NC}"
        log_message "INFO" "Processing organism $current/$total: $organism"
        
        # Sanitize filename
        sanitized_name="${organism// /_}"
        genome_file="$GENOME_DIR/${sanitized_name}.fasta"
        
        # Check if already downloaded
        if [[ -f "$genome_file" ]]; then
            echo -e "${GREEN}✓ Genome already exists. Skipping...${NC}"
            log_message "INFO" "Genome already exists for '$organism', skipping"
            ((skipped++))
            continue
        fi
        
        zip_file="$GENOME_DIR/${sanitized_name}.zip"
        
        # Attempt 1: Download RefSeq genome
        echo -e "  ${BLUE}Attempting RefSeq download...${NC}"
        log_message "INFO" "Attempting RefSeq download for '$organism'"
        
        datasets download genome taxon "$organism" \
            --include genome \
            --assembly-source refseq \
            --assembly-level "$ASSEMBLY_LEVEL" \
            --filename "$zip_file" 2>> "$ERROR_LOG"
        
        # Attempt 2: If RefSeq unavailable, try GenBank
        if [[ ! -s "$zip_file" ]]; then
            echo -e "  ${YELLOW}RefSeq unavailable, trying GenBank...${NC}"
            log_message "INFO" "RefSeq unavailable, attempting GenBank for '$organism'"
            
            datasets download genome taxon "$organism" \
                --include genome \
                --assembly-source genbank \
                --assembly-level "$ASSEMBLY_LEVEL" \
                --filename "$zip_file" 2>> "$ERROR_LOG"
        fi
        
        # Process downloaded genome
        if [[ -s "$zip_file" ]]; then
            echo -e "  ${BLUE}Extracting genome...${NC}"
            
            # Extract to temporary directory to avoid conflicts
            temp_dir="$GENOME_DIR/temp_${sanitized_name}"
            mkdir -p "$temp_dir"
            
            if unzip -q -o "$zip_file" -d "$temp_dir" 2>> "$ERROR_LOG"; then
                # Find the .fna file
                fna_file=$(find "$temp_dir" -type f -name "*.fna" | head -n 1)
                
                if [[ -f "$fna_file" ]]; then
                    # Move to final location
                    mv "$fna_file" "$genome_file"
                    echo -e "  ${GREEN}✓ Genome saved: $(basename $genome_file)${NC}"
                    log_message "INFO" "Successfully downloaded genome for '$organism'"
                    echo "$organism" >> "$SUCCESS_LOG"
                    ((downloaded++))
                else
                    echo -e "  ${RED}✗ No .fna file found in archive${NC}"
                    log_message "WARNING" "No .fna file found for '$organism'"
                    echo "$organism - No .fna file in archive" >> "$FAILED_LOG"
                    ((failed++))
                fi
            else
                echo -e "  ${RED}✗ Failed to extract archive${NC}"
                log_message "ERROR" "Failed to extract archive for '$organism'"
                echo "$organism - Extraction failed" >> "$FAILED_LOG"
                ((failed++))
            fi
            
            # Cleanup
            rm -f "$zip_file"
            rm -rf "$temp_dir"
        else
            echo -e "  ${RED}✗ No genome found (RefSeq or GenBank)${NC}"
            log_message "WARNING" "No genome available for '$organism'"
            echo "$organism - Not available in NCBI" >> "$FAILED_LOG"
            ((failed++))
        fi
        
    done < "$ORGANISM_LIST"
    
    # Clean up any remaining empty directories
    find "$GENOME_DIR" -type d -empty -delete 2>/dev/null
    
    # Print summary
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Download Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Total organisms processed: ${total}"
    echo -e "${GREEN}Already existed: ${skipped}${NC}"
    echo -e "${GREEN}Successfully downloaded: ${downloaded}${NC}"
    echo -e "${RED}Failed: ${failed}${NC}"
    echo -e "\nGenome directory: ${GENOME_DIR}"
    echo -e "Success log: ${SUCCESS_LOG}"
    echo -e "Failed log: ${FAILED_LOG}"
    echo -e "Full log: ${LOG_FILE}"
    
    log_message "INFO" "Pipeline complete - Total: $total, Skipped: $skipped, Downloaded: $downloaded, Failed: $failed"
}

################################################################################
# Main execution
################################################################################
main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Plastic-Degrading Microorganism Genome Pipeline          ║"
    echo "║  Enhanced Version with Error Handling & Progress Tracking ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
    
    log_message "INFO" "===== Pipeline Started ====="
    log_message "INFO" "Assembly level: $ASSEMBLY_LEVEL"
    
    # Execute pipeline steps
    check_dependencies
    download_tsv
    extract_organisms
    setup_directories
    download_genomes
    
    echo -e "\n${GREEN}✓ Pipeline completed successfully!${NC}"
    log_message "INFO" "===== Pipeline Completed Successfully ====="
}

# Run main function
main "$@"
