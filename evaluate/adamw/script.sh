#!/usr/bin/bash
OUTPUT_FILE_PATH=./evaluate/adamw/output.txt
echo "----------------------------CVS-------------------------------" > $OUTPUT_FILE_PATH
python evaluate/adamw/evaluate_cvs.py >> $OUTPUT_FILE_PATH
echo "----------------------------PENDULUM-------------------------------" >> $OUTPUT_FILE_PATH
python evaluate/adamw/evaluate_pendulum.py >> $OUTPUT_FILE_PATH
echo "----------------------------DOUBLE_PENDULUM-------------------------------" >> $OUTPUT_FILE_PATH
python evaluate/adamw/evaluate_double_pendulum.py >> $OUTPUT_FILE_PATH
cp "$OUTPUT_FILE_PATH" evaluate/adamw_output.txt