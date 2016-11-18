#!/bin/bash 

returnVal=()
convertArgsStrToArray() {
    local concat=""
    local t=""
    returnVal=()

    for word in $@; do
        local len=`expr "$word" : '.*"'`

        [ "$len" -eq 1 ] && concat="true"

        if [ "$concat" ]; then
            t+=" $word"
        else
            word=${word#\"}
            word=${word%\"}
            returnVal+=("$word")
        fi

        local count=$(grep -o "\"" <<< "$t" | wc -l)
        if [ "$count" -eq 2 ]; then
            t=${t# }
            t=${t#\"}
            t=${t%\"}
            returnVal+=("$t")
            t=""
            concat=""
          fi
        #fi
    done
}

function compare {

  file1=$1
  file2=$2
  digits=$3
  keyString=$4
  offset=$5
  lines=$6
  delimeter=$7
  field=$8
  errorString=$9
  removeMatch=${10}

  #echo "${1}"
  #echo "${2}"
  #echo "${3}"
  #echo "${4}"
  #echo "${5}"
  #echo "${6}"
  #echo "${7}"
  #echo "${8}"
  #echo "${9}"
  #echo "${10}"
  
  errorCode=0
  correctOffset=$((offset))
  size=$((offset + lines))
  
  fileResults1="$(cat $file1 | grep -a -A$size "$keyString" | tail -n +$correctOffset | grep -v $removeMatch | tr -s " "  | cut -d "$delimeter" -f$field  | tr -d " " | cut -b-$digits)"
  fileResults2="$(cat $file2 | grep -a -A$size "$keyString" | tail -n +$correctOffset | grep -v $removeMatch | tr -s " "  | cut -d "$delimeter" -f$field  | tr -d " " | cut -b-$digits)"

  diffResult=$(diff <(echo "$fileResults1") <(echo "$fileResults2") )

  if [ "$diffResult" != "" ] 
  then
    echo "ERROR: $errorString"
    echo "file1=$fileResults1"
    echo "file2=$fileResults2"
    errorCode=1
  fi

  return $errorCode
}

runCommand=$1
correctOutput=$2
tests=$3
errorCode=0

a="$($runCommand &> tmpResult)"
b="$(ls -l)"
echo "ran $a"
echo "files: $b"

while IFS='' read -r line || [[ -n "$line" ]]; do
  convertArgsStrToArray $line 
  compare $correctOutput tmpResult "${returnVal[@]}" 
  res=$?
  if [ "$res" != 0 ] 
  then
    errorCode=$res
  fi
done < "$tests"

exit $errorCode
