#!/bin/bash

ACUITY_PATH="/home/huangrq/Verisilicon_Tool_Acuity_Toolkit_Binary_Whl_Src_5_1_0/acuity-toolkit-binary-5.1.0/bin"
cd ../$1
OVXGENERATOR=$ACUITY_PATH/ovxgenerator
if [ ! -e "$OVXGENERATOR" ]; then
    OVXGENERATOR="python3 $OVXGENERATOR.py"
fi


function export_ovx_network()
{
    NAME=$1
    QUANTIZED=$2
    
    
    if [ ${QUANTIZED} = 'float' ]; then 
        TYPE=float;
        quantization_type="none_quantized"

    elif [ ${QUANTIZED} = 'uint8' ]; then 
        quantization_type="asymmetric_quantized-u8"

        TYPE=quantized;
    elif [ ${QUANTIZED} = 'int8' ]; then 
        quantization_type="dynamic_fixed_point-8"

        TYPE=quantized;
    elif [ ${QUANTIZED} = 'int16' ]; then       
        quantization_type="dynamic_fixed_point-16"

        TYPE=quantized;

    fi

    
    if [ ${QUANTIZED} = 'float' ]; then 
	mkdir -p "${NAME}_${QUANTIZED}"
        cmd="$OVXGENERATOR \
            --model-input ${NAME}.json \
            --data-input  ${NAME}.data \
            --export-dtype   ${TYPE} \
            --save-fused-graph \
            --reorder-channel    '$(cat reorder_channel.txt)'    \
            --channel-mean-value '$(cat channel_mean_value.txt)' \
            --model-output ${NAME}_${QUANTIZED}/${NAME}_float"
    else
    	mkdir -p "${NAME}_${QUANTIZED}"
        cmd="$OVXGENERATOR \
            --model-input ${NAME}.json \
            --data-input  ${NAME}.data \
            --export-dtype   ${TYPE} \
       	    --batch-size 1 \
            --model-quantize ${NAME}_${quantization_type}.quantize \
            --reorder-channel    '$(cat reorder_channel.txt)'    \
            --channel-mean-value '$(cat channel_mean_value.txt)' \
            --model-output ${NAME}_${QUANTIZED}/${NAME}_${QUANTIZED}"  
    fi  
        
    echo $cmd
    eval $cmd  
     
}

if [ "$#" -lt 2 ]; then
    echo "Enter a network name and quantized type ( float / uint8 / int8 / int16 )"
    exit -1
fi

export_ovx_network ${1%/} ${2%/}
