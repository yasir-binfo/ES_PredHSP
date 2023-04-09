#!/bin/bash



echo 'Feature Calculation'
echo -e '-------------------------------------------------------\n'

file_name=`echo $1 | cut -d "." -f1`
iFeature/iFeature.py --file $1 --type CKSAAP --out $file_name.csv

echo -e "\n"
echo -e 'Feature Calculation Completed'
echo -e '-------------------------------------------------------\n'




echo -e 'Start Prediction'
echo -e '-------------------------------------------------------\n'

echo "import pandas as pd
import numpy as np
from pickle import load
import os


df = pd.read_table('$file_name.csv')


df.drop('#', axis=1, inplace=True)


model = load(open('hsp_prediction_model.pkl', 'rb'))

decoder = load(open('label_encoder.pkl', 'rb'))

pred = model.predict(df)

decoding = decoder.inverse_transform(pred)

n = 1
for x in decoding:

	print('Prediction for Sequence ',n, x)
	n = n + 1


" >> python_file.py

python python_file.py

rm python_file.py

echo -e "Process Completed \a"
echo -e '-------------------------------------------------------\n'
