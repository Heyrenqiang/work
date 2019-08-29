#!/bin/bash

ACUITY_PATH="/home/huangrq/Verisilicon_Tool_Acuity_Toolkit_Binary_Whl_Src_5_1_0/acuity-toolkit-binary-5.1.0/bin"

if [ -z "$ACUITY_PATH" ]; then
    echo "Need to set enviroment variable ACUITY_PATH"
        exit 1
fi

cd ../$1
function import_caffe_network()
{
    NAME=$1
    CONVERTCAFFE=$ACUITY_PATH/convertcaffe
    if [ ! -e "$CONVERTCAFFE" ]; then
        CONVERTCAFFE="python3 $CONVERTCAFFE.py"
    fi  
    
    if [ -f ${NAME}.json ]; then
        echo -e "\033[31m rm  ${NAME}.json \033[0m" 
        rm ${NAME}.json
    fi
    
    if [ -f ${NAME}.data ]; then
        echo -e "\033[31m rm  ${NAME}.data \033[0m" 
        rm ${NAME}.data
    fi  
    
    echo "=========== Converting $NAME Caffe model ==========="
    if [ -f ${NAME}.caffemodel ]; then
    cmd="$CONVERTCAFFE \
        --caffe-model ${NAME}.prototxt \
        --caffe-blobs ${NAME}.caffemodel \
        --net-output  ${NAME}.json \
        --data-output ${NAME}.data"
    else
    echo "=========== fake Caffe model data file==========="
    cmd="$CONVERTCAFFE \
        --caffe-model ${NAME}.prototxt \
        --net-output  ${NAME}.json \
        --data-output ${NAME}.data"
    fi  
}

function import_tensorflow_network()
{
    NAME=$1
    CONVERTF=$ACUITY_PATH/convertensorflow
    if [ ! -e "$CONVERTF" ]; then
        CONVERTF="python3 $CONVERTF.py"
    fi
    
    if [ -f ${NAME}.json ]; then
        echo -e "\033[31m rm  ${NAME}.json \033[0m" 
        rm ${NAME}.json
    fi
    
    if [ -f ${NAME}.data ]; then
        echo -e "\033[31m rm  ${NAME}.data \033[0m" 
        rm ${NAME}.data
    fi  
    
    echo "=========== Converting $NAME Tensorflow model ==========="
    cmd="$CONVERTF \
        --tf-pb ${NAME}.pb \
        --data-output ${NAME}.data \
        --net-output ${NAME}.json \
        $(cat inputs_outputs.txt)"
}

function import_onnx_network()
{
    NAME=$1
    CONVERTONNX=$ACUITY_PATH/convertonnx
    if [ ! -e "$CONVERTONNX" ]; then
        CONVERTONNX="python3 $CONVERTONNX.py"
    fi
    
    if [ -f ${NAME}.json ]; then
        echo -e "\033[31m rm  ${NAME}.json \033[0m" 
        rm ${NAME}.json
    fi
    
    if [ -f ${NAME}.data ]; then
        echo -e "\033[31m rm  ${NAME}.data \033[0m" 
        rm ${NAME}.data
    fi
    
    echo "=========== Converting $NAME ONNX model ==========="
    cmd="$CONVERTONNX \
        --onnx-model  ${NAME}.onnx \
        --net-output  ${NAME}.json \
        --data-output ${NAME}.data"
}

function import_tflite_network()
{
    NAME=$1
    CONVERTTFLITE=$ACUITY_PATH/convertflite
    if [ ! -e "$CONVERTTFLITE" ]; then
        CONVERTTFLITE="python3 $CONVERTTFLITE.py"
    fi
    
    if [ -f ${NAME}.json ]; then
        echo -e "\033[31m rm  ${NAME}.json \033[0m" 
        rm ${NAME}.json
    fi
    
    if [ -f ${NAME}.data ]; then
        echo -e "\033[31m rm  ${NAME}.data \033[0m" 
        rm ${NAME}.data
    fi  
    
    echo "=========== Converting $NAME TFLite model ==========="
    cmd="$CONVERTTFLITE \
        --tflite-model ${NAME}.tflite \
        --net-output  ${NAME}.json \
        --data-output ${NAME}.data"
}

function import_darknet_network()
{
    NAME=$1
    CONVERTDARKNET=$ACUITY_PATH/convertdarknet
    if [ ! -e "$CONVERTDARKNET" ]; then
        CONVERTDARKNET="python3 $CONVERTDARKNET.py"
    fi
    
    if [ -f ${NAME}.json ]; then
        echo -e "\033[31m rm  ${NAME}.json \033[0m" 
        rm ${NAME}.json
    fi
    
    if [ -f ${NAME}.data ]; then
        echo -e "\033[31m rm  ${NAME}.data \033[0m" 
        rm ${NAME}.data
    fi
    
    echo "=========== Converting $NAME darknet model ==========="
    cmd="$CONVERTDARKNET \
        --net-input ${NAME}.cfg \
        --weight-input ${NAME}.weights \
        --net-output ${NAME}.json \
        --data-output ${NAME}.data"
}

function import_network()
{
    NAME=$1
    pushd $NAME
    
    
    if [ -f ${NAME}.prototxt ]; then
        import_caffe_network ${1%/}
    elif [ -f ${NAME}.pb ]; then
        import_tensorflow_network ${1%/}
    elif [ -f ${NAME}.onnx ]; then
        import_onnx_network ${1%/}
    elif [ -f ${NAME}.tflite ]; then
        import_tflite_network ${1%/}
    elif [ -f ${NAME}.weights ]; then
        import_darknet_network ${1%/}
    else
        echo "=========== can not find suitable model files ==========="
    fi

    echo $cmd
    eval $cmd
    
    if [ -f ${NAME}.data -a -f ${NAME}.json ]; then
        echo -e "\033[31m SUCCESS \033[0m" 
    else
        echo -e "\033[31m ERROR ! \033[0m" 
    fi  
    popd
}

if [ "$#" -ne 1 ]; then
    echo "Enter a network name !"
    exit -1
fi

import_network ${1%/}
