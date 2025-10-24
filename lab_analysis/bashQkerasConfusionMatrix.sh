#################
#bash program to run analyze the dnn profiles from the folder path that was input  
#using dynamic_parseDNNresults and then make a confusion matrix with just_qkeras_predict
#################
#usage: 
#chmod +x bashQkerasConfusionMatrix.sh
#./bashQkerasConfusionMatrix.sh /local/d1/smartpixLab/scurveData/ChipVersion1_ChipID16_SuperPix1/2025.10.24_12.10.37_DNN_vth0-0.080_vth1-0.160_vth2-0.320
#################
python dynamic_parseDNNresults.py -s -r $1/readout.csv
cp final_results.npy $1/
python just_qkeras_predict.py -d $1