#!/bin/bash

# Configuration
STUDENT_NAME="ASEP_HARYANA_SAPUTRA"
ZIP_NAME="SMSML_${STUDENT_NAME}.zip"
PROJECT_ROOT=$(dirname "$0")

# Move to the project root
cd "$PROJECT_ROOT" || exit 1

echo "🧹 Membersihkan file ZIP lama jika ada..."
rm -f "${ZIP_NAME}"

echo "📦 Memulai proses zipping file untuk submission..."

# Create the ZIP archive containing only the specified files/folders at the root level
# We enclose folder names with spaces in quotes.
zip -r "${ZIP_NAME}" \
    "Eksperimen_SML_Full_${STUDENT_NAME}.ipynb" \
    "Eksperimen_SML_${STUDENT_NAME}.txt" \
    "Membangun_model" \
    "Monitoring dan Logging" \
    "Workflow-CI.txt" \
    -x "*.pyc" "*/__pycache__/*" "*/.ipynb_checkpoints/*" "*/.DS_Store" "*/.dvc/*" "*/.git/*"

# Check if the zip command was successful
if [ $? -eq 0 ]; then
    echo "✅ Berhasil membuat ${ZIP_NAME} sesuai ketentuan Dicoding!"
else
    echo "❌ Gagal membuat file ZIP."
    exit 1
fi
