#!/bin/sh

if [ -d '/home/wuqi/' ]; then
  python="/home/wuqi/.conda/envs/krtorch/bin/python -u"
  rootpath="/home/wuqi/"
else
	echo 'This server is unavailable!'
  exit 1
fi

c='ca'
type='train' # train  test

project_dir='kr/kr1/CA/'
config_file=${rootpath}${project_dir}'configs/'${c}'.yml'  # base agw rga ca
script=${rootpath}${project_dir}${type}".py"

echo "Start running the "${type}" script..."
${python} ${script} --config_file=${config_file} SYS.ROOT_PATH "${rootpath}"
