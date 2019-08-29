#!/bin/bash
ACUITY_PATH="/home/huangrq/Verisilicon_Tool_Acuity_Toolkit_Binary_Whl_Src_5_1_0/acuity-toolkit-binary-5.1.0/bin"
cd ../$1
TENSORZONEX=$ACUITY_PATH/tensorzonex
if [ ! -e "$TENSORZONEX" ]; then
    TENSORZONEX="python3 $TENSORZONEX.py"
fi

DATASET=./dataset.txt

function quantize_network()
{
    NAME=$1
    
    QUANTIZED=$2
    
    if [ ${QUANTIZED} = 'float' ]; then 
        echo "=========== do not need quantied==========="
        exit -1 
    elif [ ${QUANTIZED} = 'uint8' ]; then 
        quantization_type="asymmetric_quantized-u8"
    elif [ ${QUANTIZED} = 'int8' ]; then 
        quantization_type="dynamic_fixed_point-8"
    elif [ ${QUANTIZED} = 'int16' ]; then       
        quantization_type="dynamic_fixed_point-16"
    else
        echo "=========== wrong quantization_type ! ( uint8 / int8 / int16 )==========="
        exit -1 
    fi
    
    if [ -f ${NAME}_${quantization_type}.quantize ]; then
        echo -e "\033[31m rm  ${NAME}_${quantization_type}.quantize \033[0m" 
        rm ${NAME}_${quantization_type}.quantize
    fi  
# model-quantize :nerual network quantized tensors 
    cmd="$TENSORZONEX \
        --action quantization \
        --batch-size 1 \
        --dtype float \
        --quantized-rebuild  \
        --quantized-dtype ${quantization_type} \
        --model-quantize  ${NAME}_${quantization_type}.quantize \
        --model-input ${NAME}.json \
        --model-data  ${NAME}.data \
        --reorder-channel    '$(cat reorder_channel.txt)'    \
        --channel-mean-value '$(cat channel_mean_value.txt)' \
        --source text \
        --source-file ${DATASET}"
    
    echo $cmd
    eval $cmd
    
}

if [ "$#" -lt 2 ]; then
    echo "Enter a network name and quantized type ( uint8 / int8 / int16 )"
    exit -1
fi

quantize_network ${1%/} ${2%/}
