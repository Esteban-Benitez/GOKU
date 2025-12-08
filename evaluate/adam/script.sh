#!/usr/bin/bash
OUTPUT_FILE_PATH=evaluate/adam/output.txt
echo "----------------------------CVS-------------------------------" > $OUTPUT_FILE_PATH
python evaluate/adam/evaluate_cvs.py >> $OUTPUT_FILE_PATH
echo "----------------------------PENDULUM-------------------------------" >> $OUTPUT_FILE_PATH
python evaluate/adam/evaluate_pendulum.py >> $OUTPUT_FILE_PATH
echo "----------------------------DOUBLE_PENDULUM-------------------------------" >> $OUTPUT_FILE_PATH
python evaluate/adam/evaluate_double_pendulum.py >> $OUTPUT_FILE_PATH

ls -l "$OUTPUT_FILE_PATH"    # <== Will tell you if the file exists
cp ${OUTPUT_FILE_PATH} evaluate/adam_output.txt