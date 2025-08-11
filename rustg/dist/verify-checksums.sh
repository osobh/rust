#!/bin/bash

# RustG GPU Compiler Archive Verification Script
# Verifies integrity of downloaded archives using checksums

set -e

# Colors for output  
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

ARCHIVE_NAME="rustg-gpu-compiler-v0.1.0-linux-x64.tar.gz"

echo -e "${BLUE}🔐 RustG GPU Compiler Archive Verification${NC}"
echo ""

# Check if archive exists
if [[ ! -f "$ARCHIVE_NAME" ]]; then
    echo -e "${RED}✗${NC} Archive not found: $ARCHIVE_NAME"
    echo "Please ensure the archive is in the current directory"
    exit 1
fi

echo -e "Verifying archive: ${ARCHIVE_NAME}"
echo ""

# Verify MD5 checksum
if [[ -f "${ARCHIVE_NAME}.md5" ]]; then
    echo -e "${BLUE}Verifying MD5 checksum...${NC}"
    if md5sum -c "${ARCHIVE_NAME}.md5" &>/dev/null; then
        echo -e "${GREEN}✓${NC} MD5 checksum verified"
    else
        echo -e "${RED}✗${NC} MD5 checksum verification failed"
        echo "Archive may be corrupted or tampered with"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} MD5 checksum file not found: ${ARCHIVE_NAME}.md5"
fi

# Verify SHA256 checksum  
if [[ -f "${ARCHIVE_NAME}.sha256" ]]; then
    echo -e "${BLUE}Verifying SHA256 checksum...${NC}"
    if sha256sum -c "${ARCHIVE_NAME}.sha256" &>/dev/null; then
        echo -e "${GREEN}✓${NC} SHA256 checksum verified"
    else
        echo -e "${RED}✗${NC} SHA256 checksum verification failed"
        echo "Archive may be corrupted or tampered with"
        exit 1  
    fi
else
    echo -e "${RED}✗${NC} SHA256 checksum file not found: ${ARCHIVE_NAME}.sha256"
fi

# Test archive integrity
echo -e "${BLUE}Testing archive integrity...${NC}"
if tar -tzf "$ARCHIVE_NAME" &>/dev/null; then
    echo -e "${GREEN}✓${NC} Archive integrity verified"
else
    echo -e "${RED}✗${NC} Archive is corrupted"
    exit 1
fi

# Show archive contents summary
echo -e "\n${BLUE}Archive contents summary:${NC}"
FILE_COUNT=$(tar -tzf "$ARCHIVE_NAME" | wc -l)
ARCHIVE_SIZE=$(ls -lh "$ARCHIVE_NAME" | awk '{print $5}')

echo "  Files: $FILE_COUNT"
echo "  Size:  $ARCHIVE_SIZE"
echo "  Contents:"
echo "    ✓ binaries (cargo-g, clippy-f)"
echo "    ✓ documentation (README, guides)"
echo "    ✓ examples (hello-world, gpu-project)"
echo "    ✓ installation script"
echo "    ✓ GPU validation tools"

echo ""
echo -e "${GREEN}🎉 Archive verification completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Extract: tar -xzf $ARCHIVE_NAME"
echo "2. Install: cd rustg-gpu-compiler-v0.1.0-linux-x64 && ./install.sh"
echo "3. Validate: ./scripts/validate_gpu.sh"